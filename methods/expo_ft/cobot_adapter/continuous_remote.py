"""Continuous cross-host transport; local inference never waits for a publication."""
import json
from pathlib import Path
import threading
import time

from async_remote import AsyncRemote
from async_state import Outbox
from online_protocol import atomic_json
from remote_runtime_support import RemoteSupport


AUDIT_FIELDS = ('rollout_rng', 'last_actor_version', 'policy_versions')


def seal_receipt(meta, receipt):
    if receipt['actor_version'] != meta['actor_version']:
        raise ValueError('Actor receipt policy mismatch')
    return {**meta, **{key: receipt[key] for key in AUDIT_FIELDS if key in receipt}}


class ContinuousOutbox(Outbox):
    def pending(self):
        with self.lock:
            paths = sorted(self.episodes.glob('*/finished.json'),
                           key=lambda p: (json.loads(p.read_text())['finished_at'], p.parent.name))
            pending = []
            for path in paths:
                identity = path.parent.name
                if (self.receipts / (identity + '.json')).exists():
                    continue
                meta = json.loads(path.read_text())
                receipt = self.episodes.parent / 'actor_receipts' / (identity + '.json')
                if not meta.get('actor_receipt_missing_after_restart') and receipt.is_file():
                    recovered = seal_receipt(meta, json.loads(receipt.read_text()))
                    if recovered != meta:
                        atomic_json(path, recovered)
                pending.append(identity)
            return pending


class ContinuousRemote(AsyncRemote):
    def remote_init(self, args):
        RemoteSupport.remote_init(self, args)
        if not self.remote or self.remote.get('async_mode') != 'native_async_v1':
            raise ValueError('Native continuous profile required')
        if self.remote.get('critic_prior', 'seeded') != 'seeded' or self.remote.get('episodes_per_update', 1) != 1:
            raise ValueError('Native continuous requires seeded sampling and immediate episodes')
        self.transfer = None
        self.outbox = ContinuousOutbox(args.run_dir / 'episodes')
        self.installing, self.install_error, self.staged = False, None, None
        self.download_state, self.upload_state, self.actor_status = {}, {}, {}
        self.network_error = None
        self.upload_busy = threading.Event()
        self.stage_retry_after = 0.
        self.publication_pending = False
        self.publication_requested_at = 0.

    def validate_identity(self, info, role):
        if (info.get('deployment_id') != self.args.model_id or info.get('rtc_delay') != self.args.rtc_delay
                or info.get('update_layout') != 'native_continuous'):
            raise RuntimeError('%s identity mismatch' % role)
        if 'edit_scale' in self.remote and info.get('edit_scale') != self.remote['edit_scale']:
            raise RuntimeError('%s edit scale mismatch' % role)

    def validate_remote_status(self, info):
        RemoteSupport.validate_remote_status(self, info)
        if info.get('async_mode') != 'native_async_v1' or info.get('critic_prior') != 'seeded':
            raise RuntimeError('Remote learner is not native continuous')

    def remote_online_status(self):
        value = dict(self.remote_status)
        local_pending = self.outbox.pending()
        pending = set(value.get('pending_episode_ids', [])) | set(local_pending)
        pending.update(self.outbox.unconfirmed(value.get('admission_revision', 0)))
        reason = 'learner_fault' if value.get('phase') == 'fault' else None
        if self.actor_status.get('phase') == 'fault':
            reason = 'actor_fault'
        ready = bool(self.ready and self.local_actor_version >= 0 and reason is None)
        value.update(ready=ready, phase='rollout' if self.episode else 'ready' if ready else reason or 'loading',
            learner_phase=self.remote_status.get('phase'), learner_active_episode=value.get('active_episode'),
            active_episode=self.episode, served_actor_version=self.local_actor_version,
            published_actor_version=self.remote_status.get('actor_version', -1),
            downloaded_actor_version=self.staged[1]['actor_version'] if self.staged else self.local_actor_version,
            pending_upload_episodes=len(local_pending), pending_episodes=len(pending),
            collection_waiting_for=reason, download=dict(self.download_state), upload=dict(self.upload_state),
            network_error=self.network_error, install_error=self.install_error,
            actor_stage_phase=self.actor_status.get('stage_phase'), actor_stage_error=self.actor_status.get('stage_error'),
            compute_host='Learner %s / Actor Cobot' % self.remote.get('learner_host', 'HPC'))
        return value

    def json_rpc(self, path, body):
        if path == '/episode/start':
            if not self.remote_online_status()['ready']:
                raise ValueError('Continuous Actor is not ready')
            # The Actor atomically adopts and reports the actual starting version.
            value = self.local_rpc(path, {'episode_uuid': body['episode_uuid']})
            self.local_actor_version = max(self.local_actor_version, value['actor_version'])
            return value
        if path == '/episode/cancel-start':
            return self.local_rpc(path, body)
        return super().json_rpc(path, body)

    def transfer_episode(self, identity):
        path = self.journal.root / identity / 'finished.json'
        meta = json.loads(path.read_text())
        if not meta.get('actor_receipt_missing_after_restart'):
            receipt = self.local_rpc('/episode/end', {'episode_uuid': identity})
            atomic_json(path, seal_receipt(meta, receipt))
            self.local_actor_version = max(self.local_actor_version, receipt.get('last_actor_version', receipt['actor_version']))
        return self.remote_online_status()

    def policy_command(self):
        directory = self.start_remote()
        command = [self.remote['python'], str(Path(__file__).with_name('continuous_actor.py')),
            '--project', self.remote['native_project'], '--bundle', str(self.args.bundle),
            '--model-id', self.args.model_id, '--snapshot-root', str(self.actor_root), '--snapshot', str(directory),
            '--port', str(self.args.port), '--rtc-delay', str(self.args.rtc_delay)]
        return command + (['--edit-scale', str(self.remote['edit_scale'])] if 'edit_scale' in self.remote else [])

    def local_status(self):
        with self.rpc.open('http://127.0.0.1:%d/status' % self.args.port, timeout=2) as response:
            value = json.load(response)
        self.validate_server_status(value)
        self.actor_status = value
        self.local_actor_version = max(self.local_actor_version, value['actor_version'])
        return value

    def download_loop(self):
        while not self.stop.wait(.5):
            try:
                self.download_once()
            except RuntimeError as exc:
                self.actor_status = dict(phase='fault', error=str(exc))
                self.fail(str(exc))
                return
            except (OSError, ValueError) as exc:
                self.download_state = dict(self.download_state, state='retrying', error=str(exc))
                self.stop.wait(2)

    def download_once(self):
        if not self.transfer_lock.acquire(blocking=False):
            return False
        try:
            return self._download_once()
        finally:
            self.transfer_lock.release()

    def _download_once(self):
        if self.staged:
            return self.maybe_install()
        # One outstanding demand, coalesced by the Learner at a safe update point.
        latest = self.remote_status.get('actor_version', -1)
        if self.publication_pending and latest <= self.local_actor_version:
            if time.monotonic() - self.publication_requested_at < 30:
                return False
            self.publication_pending = False  # Retry demand after a disconnected ACK.
        if not self.publication_pending:
            requested = self.remote_rpc('/async/actor/request', {})
            self.publication_pending = bool(requested.get('publication_pending'))
            self.publication_requested_at = time.monotonic()
            latest = requested.get('actor_version', latest)
        if latest <= self.local_actor_version:
            return False
        started = time.monotonic()
        self.download_state = dict(state='downloading', actor_version=latest, started_at=time.time())
        directory, manifest = self.download_actor()
        self.staged = (directory, manifest)
        self.publication_pending = False
        atomic_json(self.args.run_dir / 'downloaded.json', dict(directory=str(directory), actor_version=manifest['actor_version']))
        self.download_state = dict(self.download_state, state='downloaded', actor_version=manifest['actor_version'],
                                   downloaded_at=time.time(), seconds=time.monotonic() - started)
        return self.maybe_install()

    def maybe_install(self):
        if self.staged is None:
            return False
        directory, manifest = self.staged
        version = manifest['actor_version']
        value = self.local_status()
        if value['actor_version'] < version:
            phase = value.get('stage_phase')
            matching = value.get('stage_directory') == str(directory)
            if matching and phase == 'stage_failed':
                self.install_error = value.get('stage_error')
                self.download_state = dict(self.download_state, state='stage_failed', actor_version=version, error=self.install_error)
            if not matching or phase in ('idle', 'adopted', 'stage_failed'):
                if time.monotonic() < self.stage_retry_after:
                    return False
                self.stage_retry_after = time.monotonic() + 5
                self.download_state.setdefault('stage_started_monotonic', time.monotonic())
                value = self.local_rpc('/actor/stage', {'directory': str(directory)})
                self.actor_status = value
                self.download_state = dict(self.download_state, state='staging', actor_version=version, error=None)
            if value.get('stage_phase') != 'ready':
                return False
            if not self.control_lock.acquire(blocking=False):
                return False
            try:
                if self.episode or self.journal.active or (self.future is not None and not self.future.done()):
                    return False
                value = self.local_rpc('/actor/adopt', {})
                self.actor_status = value
            finally:
                self.control_lock.release()
        if value['actor_version'] != version:
            raise ValueError('Adopted Actor identity mismatch')
        self.local_actor_version = max(self.local_actor_version, version)
        self.staged, self.install_error = None, None
        adopted_at = time.monotonic()
        self.download_state = dict(self.download_state, state='installed', actor_version=version, error=None,
            install_seconds=max(0., adopted_at-self.download_state.get('stage_started_monotonic', adopted_at)),
            installed_at=time.time())
        atomic_json(self.args.run_dir / 'installed.json', self.download_state)
        from actor_snapshot import prune
        prune(self.actor_root, version)
        return True
