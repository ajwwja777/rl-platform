"""Native continuous EXPO with durable independent Replay and remote snapshots."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from online_learner import Learner
from online_protocol import atomic_json
from async_state import retire_uncommitted_snapshots
from continuous_state import Activity, ContinuousInbox, REPLAY_KEYS
from continuous_learning import work


class ContinuousLearner(Learner):
    work = work

    def __init__(self, args):
        if not (args.recipe == 'native_sync' and args.learner_only and args.native_batch_update):
            raise ValueError('Continuous mode requires native full-batch remote learner')
        if args.critic_prior != 'seeded' or args.episodes_per_update != 1:
            raise ValueError('Continuous mode uses native sampling without episode batching')
        from continuous_native import install
        install()
        self.replay_lock = threading.RLock()
        self.status_lock = threading.RLock()
        self.stop, self.wake, self.publish_requested = (threading.Event() for _ in range(3))
        self.activity = Activity(args.ep_timeout_secs)
        self.inbox = None
        self.publications = {}
        self.published_version = -1
        self.timings, self._status = {}, {}
        self.last_update_at = None
        self.last_checkpoint_at = time.time()
        self.committed_updates = -1
        self.checkpoint_writer = None
        self.checkpoint_phase = 'idle'
        self.checkpoint_publish = False
        self._replay_restored = False
        self._contract(args)
        retire_uncommitted_snapshots(args.run_dir)
        super().__init__(args)
        self.committed_updates = self.meta['updates']
        self.resume_learning = self.meta['checkpoint'] is not None or self.meta['completed_episodes'] > 0
        self.publish_committed()
        self.inbox = ContinuousInbox(self.root, self.known_versions, max_pending=0)
        self.inbox.recover()
        self.phase = 'ready'
        self.save_status()
        self.worker = threading.Thread(target=self.work, name='expo-continuous-learner', daemon=True)
        self.worker.start()

    def _contract(self, args):
        digest = hashlib.sha256()
        for name in ('continuous_learner.py', 'continuous_learning.py', 'continuous_state.py',
                     'continuous_native.py', 'continuous_snapshot.py', 'continuous_http.py',
                     'continuous_checkpoint.py', 'continuous_persistence.py'):
            digest.update(Path(__file__).with_name(name).read_bytes())
        value = dict(mode='native_async_v1', source_sha256=digest.hexdigest(),
                     async_script_sha256=hashlib.sha256((args.project/'train_pi_robo_async.py').read_bytes()).hexdigest(),
                     ep_timeout_secs=args.ep_timeout_secs, checkpoint_interval_secs=args.checkpoint_interval_secs,
                     sampling='native_BatchProcessor', independent_rollout_rng=True)
        path = args.run_dir/'continuous_contract.json'
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Continuous contract changed; use an explicit continuation')
        atomic_json(path, value)

    def rebuild(self):
        if not self._replay_restored:
            path = self.root/'replay-state.json'
            if path.exists():
                replay = json.loads(path.read_text())
                self.meta.update({k: replay[k] for k in REPLAY_KEYS if k in replay})
            self._replay_restored = True
        super().rebuild()

    def recover(self, identities):
        # Independent consumer owns admission and insertion, never base sync schedule.
        self.wake.set()

    def publish_actor(self):
        from actor_snapshot import export
        from continuous_snapshot import contract
        self.actor_manifest = export(self.root/'actors', self.agent, self.meta['updates'],
                                     contract(self.args.bundle, self.args.project, self.rtc_delay, self.edit_scale))

    def publish_committed(self, version=None):
        if version is None:version = self.meta['updates']
        if self.meta_path.exists():
            durable = json.loads(self.meta_path.read_text())['updates']
            if version > durable:raise ValueError('Cannot publish uncommitted optimizer state')
        path = self.root/'publications.json'
        versions = json.loads(path.read_text()) if path.exists() else {}
        if str(version) not in versions:
            versions[str(version)] = dict(generation=len(versions), published_at=time.time())
            atomic_json(path, versions)
        self.publications, self.published_version = versions, version

    def known_versions(self):
        return {int(v) for v in self.publications}

    def request_publish(self):
        # Coalesced demand from an idle downloader; never adds a training budget.
        writing_publication=getattr(self,'checkpoint_publish',False) and getattr(self,'checkpoint_phase','idle') in ('capturing','writing')
        if not writing_publication and (self.phase not in ('waiting_episode','waiting_replay') or self.meta['updates'] > self.published_version):
            self.publish_requested.set()
            self.wake.set()
        return dict(actor_version=self.published_version, published_actor_version=self.published_version,
                    learner_step=self.meta['updates'], publication_pending=self.publish_requested.is_set() or writing_publication)

    def save_status(self):
        with self.status_lock:
            if self.error is not None:self.phase='fault'
            with self.replay_lock:
                meta = dict(self.meta)
                replay_size = len(self.buffer)
            pending = self.inbox.pending() if self.inbox else []
            publication = self.publications.get(str(self.published_version), {})
            value = dict(ready=self.error is None and self.phase != 'fault' and self.published_version >= 0,
                phase=self.phase, learner_phase=self.phase, error=self.error, active_episode=None,
                actor_version=self.published_version, published_actor_version=self.published_version,
                learner_step=meta['updates'], actor_updates=meta['actor_updates'],
                critic_updates=meta['critic_updates'], replay_steps=replay_size,
                accepted_episodes=len(meta['accepted']), completed_episodes=meta['completed_episodes'],
                replay_online_episodes=len(meta['replay']),
                seed_episodes=25, seed_steps=self.seed['frames'], online_steps=meta['steps'],
                training_enabled=True, deployment_id=self.args.deployment_id,
                metrics=dict(self.metrics), updated_at=time.time(), pid=os.getpid(),
                recipe='native_async', async_mode='native_async_v1', learning_schedule='native_continuous',
                update_layout='native_continuous', batch_size=64, utd_ratio=20,
                critic_prior='seeded', critic_sampling='native_uniform', episodes_per_update=1,
                warmup_remaining=0, pending_update_calls=0, async_max_pending=0, async_max_policy_lag=0,
                rtc_delay=self.rtc_delay, edit_scale=self.edit_scale, learner_devices=self.num_devices,
                visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                pending_episode_ids=pending, pending_episodes=len(pending),
                publications=dict(self.publications), published_generation=publication.get('generation', 0),
                published_at=publication.get('published_at'), publication_pending=self.publish_requested.is_set() or
                (self.checkpoint_publish and self.checkpoint_phase in ('capturing','writing')),
                ep_timeout_secs=self.args.ep_timeout_secs,
                last_episode_at=meta.get('last_episode_at'), last_replay_insert_at=meta.get('last_replay_insert_at'),
                last_update_at=self.last_update_at, timings=dict(self.timings), checkpoint_phase=self.checkpoint_phase,
                committed_learner_step=self.committed_updates)
            try:
                from monitoring import LearnerHistory
                if not hasattr(self, '_monitor_history'):
                    self._monitor_history = LearnerHistory(self.root, meta['critic_updates'])
                self._monitor_history.observe(value, self.metrics)
                value['monitoring'] = self._monitor_history.payload()
            except Exception:
                logging.exception('Continuous telemetry failed; learning unaffected')
            self._status = value
            atomic_json(self.root/'status.json', value)
            return value

    def snapshot(self):
        value = dict(self._status)
        if self.inbox:value.update(self.inbox.transport_status())
        return value


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'bundle', 'run-dir', 'seed-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=18443)
    parser.add_argument('--deployment-id', required=True)
    parser.add_argument('--recipe', choices=['native_sync'], default='native_sync')
    parser.add_argument('--learner-only', action='store_true')
    parser.add_argument('--native-batch-update', action='store_true')
    parser.add_argument('--critic-prior', choices=['seeded'], default='seeded')
    parser.add_argument('--episodes-per-update', type=int, choices=[1], default=1)
    parser.add_argument('--num-devices', type=int, choices=[1,2,4], default=4)
    parser.add_argument('--rtc-delay', type=int, default=5)
    parser.add_argument('--edit-scale', type=float, default=.01)
    parser.add_argument('--ep-timeout-secs', type=float, default=120)
    parser.add_argument('--checkpoint-interval-secs', type=float, default=300)
    args = parser.parse_args()
    if args.ep_timeout_secs <= 0 or args.checkpoint_interval_secs <= 0:parser.error('Invalid persistence/timeout interval')
    args.batch_size=64;args.capacity=500000;args.warmup_steps=0;args.steps_per_update=0;args.max_updates=0
    from native_online import source_path
    source_path(args.project)
    learner = ContinuousLearner(args)
    from continuous_http import serve
    serve(learner, args.port)


if __name__ == '__main__':main()
