"""Single optimizer owner; journal reception continues while GPU updates run."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from online_protocol import atomic_json
from observed_learner import ObservedLearner
from async_learning import learn_episode
from async_state import Inbox,retire_uncommitted_snapshots


class AsyncLearner(ObservedLearner):
    learn_episode=learn_episode

    def __init__(self,args):
        if not (args.recipe=='native_sync' and args.learner_only and args.native_batch_update):
            raise ValueError('Async requires native full-batch learner-only mode')
        if args.async_max_pending<args.episodes_per_update:raise ValueError('Async bound smaller than episode batch')
        self.stop=threading.Event();self.wake=threading.Event()
        self.inbox=None;self.publications={};self.published_version=-1
        self.pending_updates=0;self.timings={};self._status={}
        self.max_pending=args.async_max_pending
        self.async_contract(args)
        retire_uncommitted_snapshots(args.run_dir)
        super().__init__(args)
        self.publish_committed()
        self.inbox=Inbox(self.root,self.known_versions,self.max_pending)
        self.inbox.recover()
        self.phase='ready';self.save_status()
        self.worker=threading.Thread(target=self.work,name='expo-async-learner',daemon=True)
        self.worker.start()

    def async_contract(self,args):
        signature=hashlib.sha256()
        for name in ('async_learner.py','async_learning.py','async_state.py','async_http.py'):
            signature.update(Path(__file__).with_name(name).read_bytes())
        value=dict(mode='bounded_async_v1',source_sha256=signature.hexdigest(),max_pending=args.async_max_pending,
                   max_policy_lag=args.async_max_policy_lag,independent_rollout_rng=True)
        path=args.run_dir/'async_contract.json'
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Async configuration changed; use an explicit continuation')
        atomic_json(path,value)

    def known_versions(self):
        return {int(v) for v in self.publications}

    def publish_committed(self):
        # Inference files can exist before commit; advertise only durable optimizer state.
        version=self.meta['updates']
        path=self.root/'publications.json'
        versions=json.loads(path.read_text()) if path.exists() else {}
        if str(version) not in versions:
            versions[str(version)]=dict(generation=len(versions),published_at=time.time())
            atomic_json(path,versions)
        self.publications=versions;self.published_version=version

    def recover(self,identities):
        # Base constructor detects lost finish RPCs; our inbox worker owns replay.
        self.wake.set()

    def save_status(self):
        value=super().save_status()
        pending=self.inbox.backlog() if self.inbox else []
        value.update(async_mode='bounded_async_v1',ready=self.phase!='fault' and self.published_version>=0,
            learner_phase=self.phase,actor_version=self.published_version,published_actor_version=self.published_version,
            published_generation=self.publications.get(str(self.published_version),{}).get('generation',0),
            publications=self.publications,pending_episode_ids=pending,pending_episodes=len(pending),
            pending_update_calls=self.pending_updates,async_max_pending=self.max_pending,
            async_max_policy_lag=self.args.async_max_policy_lag,timings=dict(self.timings))
        self._status=value
        atomic_json(self.root/'status.json',value)
        return value

    def snapshot(self):
        value=dict(self._status)
        if self.inbox:
            value.update(self.inbox.transport_status())
        return value

    def work(self):
        while not self.stop.is_set():
            pending=self.inbox.pending()
            if not pending:
                self.wake.wait(.5);self.wake.clear();continue
            identity=pending[0]
            try:
                self.active=identity;self.phase='updating';self.save_status()
                with self.lock:self.learn_episode(identity)
                self.active=None;self.phase='ready';self.save_status()
            except Exception as exc:
                self.error=str(exc);self.phase='fault';self.save_status()
                logging.exception('Async learner stopped; admitted journals retained')
                return


def main():
    p=argparse.ArgumentParser()
    for name in ('project','bundle','run-dir','seed-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=18443);p.add_argument('--deployment-id',required=True)
    p.add_argument('--recipe',choices=['native_sync'],default='native_sync')
    p.add_argument('--learner-only',action='store_true');p.add_argument('--native-batch-update',action='store_true')
    p.add_argument('--critic-prior',choices=['seeded','demo50','demo50_success30_failure20'],default='demo50_success30_failure20')
    p.add_argument('--episodes-per-update',type=int,default=5);p.add_argument('--num-devices',type=int,choices=[1,2,4],default=4)
    p.add_argument('--rtc-delay',type=int,default=5);p.add_argument('--edit-scale',type=float,default=.01)
    p.add_argument('--async-max-pending',type=int,default=10);p.add_argument('--async-max-policy-lag',type=int,default=2)
    args=p.parse_args()
    if not args.episodes_per_update<=args.async_max_pending<=50 or not 1<=args.async_max_policy_lag<=10:p.error('Invalid async queue bounds')
    args.batch_size=64;args.capacity=500000;args.warmup_steps=0;args.steps_per_update=30;args.max_updates=0
    from native_online import source_path
    source_path(args.project)
    learner=AsyncLearner(args)
    from async_http import serve
    serve(learner,args.port)

if __name__=='__main__':main()
