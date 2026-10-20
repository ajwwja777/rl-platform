"""Native asynchronous Actor: background snapshots and transactional rollout RNG.

All public mutations run under ``lock`` (the HTTP handler owns it). Restore only
holds the lock to publish a completed immutable agent; it never warms inference.
"""
import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from actor_snapshot import restore
from native_online import source_path
from online_actor import Actor
from online_protocol import atomic_json, episode_id
from wire import decode


def configure_native():
    import continuous_native
    import continuous_snapshot
    import online_actor
    continuous_native.install()
    online_actor.contract = continuous_snapshot.contract


def wait_ready(agent):
    import jax
    jax.block_until_ready(agent._infer_cache)


class ContinuousActor(Actor):
    max_pending_tokens = 4

    def __init__(self, args):
        self.stage_phase = 'idle'
        self.stage_error = None
        self.stage_version = None
        self.stage_directory = None
        self.stage_thread = None
        self.staged = None
        self.pending_tokens = OrderedDict()
        self.starting_version = None
        self.policy_versions = []
        configure_native()
        super().__init__(args)

    def status(self):
        return {**super().status(), 'update_layout': 'native_continuous', 'async_mode': 'native_async_v1',
                'stage_phase': self.stage_phase, 'stage_error': self.stage_error,
                'stage_version': self.stage_version, 'stage_directory': self.stage_directory,
                'pending_inferences': len(self.pending_tokens),
                'starting_actor_version': self.starting_version,
                'policy_versions': list(self.policy_versions)}

    def install(self, directory):
        if self.stage_phase in ('loading', 'ready'):
            raise ValueError('Actor staging is already pending')
        return super().install(directory)

    def stage(self, directory):
        directory = Path(directory).resolve()
        if directory.parent != self.args.snapshot_root.resolve():
            raise ValueError('Actor path outside configured snapshots')
        if self.stage_phase in ('loading', 'ready'):
            if str(directory) == self.stage_directory:
                return self.status()
            raise ValueError('Another Actor snapshot is already staging')
        # Capture the immutable template while the caller holds the inference lock.
        template = self.agent
        self.stage_directory = str(directory)
        self.stage_phase, self.stage_error, self.stage_version = 'loading', None, None
        self.stage_thread = threading.Thread(target=self._restore, args=(directory, template), daemon=True)
        self.stage_thread.start()
        return self.status()

    def _restore(self, directory, template):
        try:
            manifest = json.loads((directory / 'manifest.json').read_text())
            version = int(manifest['actor_version'])
            with self.lock:
                if version < self.meta['updates']:
                    raise ValueError('Actor version rollback is forbidden')
                self.stage_version = version
            candidate, manifest = restore(directory, template, self.identity)
            wait_ready(candidate)
            with self.lock:
                if int(manifest['actor_version']) < self.meta['updates']:
                    raise ValueError('Actor version rollback is forbidden')
                self.staged = (candidate, manifest)
                self.stage_phase = 'ready'
        except Exception as exc:
            with self.lock:
                self.staged = None
                self.stage_phase, self.stage_error = 'stage_failed', str(exc)

    def _adopt(self):
        if self.staged is None:
            return
        candidate, manifest = self.staged
        self.agent = candidate.replace(rng=self.agent.rng)
        self.meta['updates'] = int(manifest['actor_version'])
        self.staged = None
        self.stage_phase = 'adopted'

    def adopt(self):
        if self.active is not None:
            raise ValueError('Idle adoption requires no active episode')
        self._adopt()
        return self.status()

    def begin(self, data):
        identity = episode_id(data['episode_uuid'])
        if self.active == identity and self.phase == 'rollout':
            return self.status()
        if self.active or self.phase != 'ready':
            raise ValueError('Local Actor is not ready for this version')
        if 'actor_version' in data and int(data['actor_version']) != self.meta['updates']:
            raise ValueError('Local Actor is not ready for this version')
        if self._receipt(identity).exists():
            raise ValueError('Episode already finished')
        self._adopt()
        # Native async keeps rollout RNG across episodes; learner RNG is unrelated.
        self.pending_tokens.clear()
        self.active, self.phase = identity, 'rollout'
        self.starting_version = self.meta['updates']
        self.policy_versions = [self.starting_version]
        return self.status()

    def infer(self, route, data):
        if self.phase != 'rollout' or str(np.asarray(data['episode_uuid']).item()) != self.active:
            raise ValueError('Inference requires the active online episode')
        if route not in ('/precache', '/select'):
            raise ValueError('Unknown inference operation')
        token = None
        if route == '/select':
            token_id = str(np.asarray(data.get('inference_token', '')).item())
            token = self.pending_tokens.get(token_id)
            if token is None or token['episode'] != self.active:
                raise ValueError('Invalid or retired inference token')
            if int(data['actor_version']) != token['version'] or int(data['delay']) != token['delay']:
                raise ValueError('Inference token candidate metadata mismatch')
        self._adopt()
        live = self.agent
        try:
            if token is not None:
                # Native loop_utils carries only old precache RNG into latest agent.
                self.agent = live.replace(rng=token['rng'])
                data = {**data, 'actor_version': self.meta['updates']}
            result = super().infer(route, data)
            if route == '/precache':
                token_id = str(uuid.uuid4())
                self.pending_tokens[token_id] = dict(rng=self.agent.rng, episode=self.active,
                    version=result['actor_version'], delay=result['delay'])
                self.agent = live
                while len(self.pending_tokens) > self.max_pending_tokens:
                    self.pending_tokens.popitem(last=False)
                result['inference_token'] = token_id
            else:
                # Any other transaction started from the old RNG and is now stale.
                self.pending_tokens.clear()
                result['precache_actor_version'] = token['version']
                for version in (token['version'], result['actor_version']):
                    if version not in self.policy_versions:
                        self.policy_versions.append(version)
            return result
        except Exception:
            self.agent = live
            raise

    def _receipt(self, identity):
        return self.args.snapshot_root.parent / 'actor_receipts' / (identity + '.json')

    def finish(self, data):
        identity = episode_id(data['episode_uuid'])
        receipt = self._receipt(identity)
        if receipt.exists():
            return json.loads(receipt.read_text())
        if identity != self.active:
            raise ValueError('Stale local Actor episode')
        value = dict(rollout_rng=np.asarray(self.agent.rng).tolist(),
                     actor_version=self.starting_version, last_actor_version=self.meta['updates'],
                     policy_versions=list(self.policy_versions))
        atomic_json(receipt, value)
        self.pending_tokens.clear()
        self.active, self.phase = None, 'ready'
        return value


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'bundle', 'snapshot-root', 'snapshot'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--rtc-delay', type=int, default=5)
    parser.add_argument('--port', type=int, default=8043)
    parser.add_argument('--edit-scale', type=float, default=.01)
    args = parser.parse_args()
    source_path(args.project)
    actor = ContinuousActor(args)

    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm = True

        def log_message(self, *args):
            pass

        def send(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(200, actor.status())

        def do_POST(self):
            inference = self.path in ('/precache', '/select')
            acquired = actor.lock.acquire(blocking=False) if inference else actor.lock.acquire(timeout=5)
            if not acquired:
                return self.send(409, {'error': 'Actor is busy'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length < 2 * 1024 * 1024:
                    raise ValueError('Invalid body')
                body = self.rfile.read(length)
                if inference:
                    started = time.monotonic()
                    result = actor.infer(self.path, decode(body, self.headers.get('Content-Encoding', 'identity')))
                    result['compute_seconds'] = time.monotonic() - started
                else:
                    data = json.loads(body)
                    if self.path == '/episode/start':
                        result = actor.begin(data)
                    elif self.path == '/episode/cancel-start' and actor.active is None:
                        receipt = actor._receipt(episode_id(data['episode_uuid']))
                        result = json.loads(receipt.read_text()) if receipt.exists() else actor.status()
                    elif self.path in ('/episode/end', '/episode/cancel-start'):
                        result = actor.finish(data)
                    elif self.path == '/actor/install':
                        result = actor.install(data['directory'])
                    elif self.path == '/actor/adopt':
                        result = actor.adopt()
                    elif self.path == '/actor/stage':
                        result = actor.stage(data['directory'])
                    else:
                        raise ValueError('Unknown Actor operation')
                self.send(200, result)
            except Exception as exc:
                self.send(409, {'error': str(exc)})
            finally:
                actor.lock.release()

    print('EXPO continuous inference-only Actor ready; no optimizer updates or ROS publishers', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
