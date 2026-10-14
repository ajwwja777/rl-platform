"""Single-GPU native EXPO: episode-pinned inference and synchronous updates."""
import argparse
import io
import json
import os
from pathlib import Path
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from online_protocol import atomic_json, episode_id, transitions
from episode_journal import read_journal
from native_online import source_path, load_native, replay_buffer, insert_episode, valid_batch


class Learner:
    def __init__(self, args):
        import jax
        self.args=args
        self.aligned=getattr(args,'recipe','legacy')=='native_sync'
        self.native_batch_update=bool(getattr(args,'native_batch_update',False))
        self.rtc_delay=int(getattr(args,'rtc_delay',2))
        if not 1<=self.rtc_delay<=8:raise ValueError('Invalid RTC delay')
        self.seed=None
        if self.aligned:
            from success_seed import seed_manifest
            self.seed=seed_manifest(args.seed_dir)
        self.lock=threading.Lock()
        self.active=None
        self.phase='loading'
        self.error=None
        self.metrics={}
        self.root=args.run_dir
        self.root.mkdir(parents=True,exist_ok=True)
        atomic_json(self.root/'status.json',dict(ready=False,phase='loading',pid=os.getpid(),updated_at=time.time()))
        self.meta_path=self.root/'committed.json'
        import hashlib
        signature=hashlib.sha256()
        for name in ('deployment.json','norm_stats.json'):
            signature.update((args.bundle/name).read_bytes())
        source_signature=hashlib.sha256()
        for directory in ('expo_ft','configs'):
            for source in sorted((args.project/directory).rglob('*.py')):
                source_signature.update(str(source.relative_to(args.project)).encode());source_signature.update(source.read_bytes())
        contract=dict(native_source_sha256=source_signature.hexdigest(),bundle=str(args.bundle.resolve()),bundle_metadata_sha256=signature.hexdigest(),batch_size=args.batch_size,capacity=args.capacity,warmup_steps=args.warmup_steps,steps_per_update=args.steps_per_update,max_updates=args.max_updates,rtc_delay=self.rtc_delay,schema=1)
        if self.aligned:
            for name in ('train_pi_robo.py','scripts/dynamic_pick/run_server.sh'):
                source_signature.update((args.project/name).read_bytes())
            contract.update(native_source_sha256=source_signature.hexdigest(),recipe='native_sync',schema=2,utd_ratio=20,seed_manifest=self.seed,rtc_delay=self.rtc_delay)
            adapter_signature=hashlib.sha256()
            for name in ('native_online.py','memory_update.py','batch_memory.py','split_update.py','stream_sample.py','sync_schedule.py','success_seed.py','online_protocol.py','episode_journal.py','online_learner.py','stream_checkpoint.py','remote_journal.py'):
                adapter_signature.update(Path(__file__).with_name(name).read_bytes())
            contract['adapter_source_sha256']=adapter_signature.hexdigest()
            contract['update_layout']='native_full_batch' if self.native_batch_update else 'experimental_streamed'
        contract_path=self.root/'contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
            raise ValueError('Online run configuration changed; use an independent run directory')
        atomic_json(contract_path,contract)
        self.meta=json.loads(self.meta_path.read_text()) if self.meta_path.exists() else dict(
            checkpoint=None,updates=0,accepted=[],replay=[],steps=0,credit=0,actor_updates=0)
        self.meta.setdefault('completed_episodes',0)
        self.meta.setdefault('critic_updates',0)
        self.agent,self.config,self.prompt=load_native(args.bundle,self.rtc_delay,resume=False,aligned=self.aligned)
        if self.aligned:
            from batch_memory import install
            install()
        from memory_update import HostTargetUpdater
        self.updater=HostTargetUpdater(self.agent)
        self.updater.split_batches=self.aligned and not self.native_batch_update
        self.updater.progress=self.update_progress
        self.agent=self.updater.prepare(self.agent)
        if self.meta['checkpoint'] is not None:
            from stream_checkpoint import restore
            self.agent=restore(self.root/'checkpoints',self.agent,self.updater,self.meta['checkpoint'])
        if 'agent_rng' in self.meta:
            self.agent=self.agent.replace(rng=jax.device_put(np.asarray(self.meta['agent_rng'],np.uint32),self.agent.rng.sharding))
        self.agent=self.agent.cache_infer_params()
        self.buffer=replay_buffer(self.config,self.prompt,args.capacity)
        self.rebuild()
        if 'replay_rng' in self.meta:
            self.buffer.rng=jax.numpy.asarray(self.meta['replay_rng'],dtype=jax.numpy.uint32)
        self.warm_inference()
        self.phase='ready'
        self.save_status()
        # Completed journal with a lost finish RPC is recovered exactly once.
        pending=[p.parent.name for p in (self.root/'episodes').glob('*/finished.json')
                 if p.parent.name not in self.meta['accepted']]
        if pending:
            self.phase='updating'
            self.save_status()
            threading.Thread(target=self.recover,args=(pending,),daemon=True).start()

    def warm_inference(self):
        with np.load(self.args.bundle.parent/'common/dev_observation.npz',allow_pickle=False) as z:
            obs={k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
        # Compile both native inference paths before exposing readiness.
        saved_rng=self.agent.rng
        cached,self.agent=self.agent.sample_pre_cache(dict(obs))
        actions,self.agent,info=self.agent.sample_actions(dict(obs),precached=cached)
        prefix=np.asarray(info['executed_padded'])[-self.rtc_delay:]
        cached,self.agent=self.agent.sample_pre_cache(dict(obs),prefix)
        actions,self.agent,info=self.agent.sample_actions(dict(obs),precached=cached,delay=self.rtc_delay)
        assert np.isfinite(np.asarray(actions)).all()
        for d in (0,self.rtc_delay):
            actions,self.agent,info=self.agent.sample_actions(dict(obs),precached=np.asarray(cached),delay=d,only_base_actions=True)
            assert np.isfinite(np.asarray(actions)).all()
        for d in (0,self.rtc_delay):
            actions,self.agent,info=self.agent.sample_actions(dict(obs),precached=np.asarray(cached),delay=d,only_base_actions=False)
            assert np.isfinite(np.asarray(actions)).all()
        self.agent=self.agent.replace(rng=saved_rng)

    def update_progress(self,phase,step):
        if phase=='critic':self.meta['critic_updates']=step
        if phase=='actor':self.meta['actor_updates']=step
        self.save_status()

    def save_status(self):
        value=dict(ready=self.phase in ('ready','rollout'),phase=self.phase,error=self.error,
            active_episode=self.active,actor_version=self.meta['updates'],learner_step=self.meta['updates'],
            actor_updates=self.meta['actor_updates'],replay_steps=max(0,len(self.buffer)-(0 if self.aligned else len(self.meta['replay']))),
            accepted_episodes=len(self.meta['accepted']),training_enabled=True,deployment_id=getattr(self.args,'deployment_id',None),
            metrics=self.metrics,updated_at=time.time(),pid=os.getpid(),batch_size=self.args.batch_size,
            warmup_steps=self.args.warmup_steps,steps_per_update=self.args.steps_per_update)
        if self.aligned:
            value.update(recipe='native_sync',rtc_delay=self.rtc_delay,update_layout='native_full_batch' if self.native_batch_update else 'experimental_streamed',warmup_episodes=10,completed_episodes=self.meta['completed_episodes'],
                warmup_remaining=max(0,10-self.meta['completed_episodes']),critic_updates=self.meta['critic_updates'],
                utd_ratio=20,seed_episodes=25,seed_steps=self.seed['frames'],online_steps=self.meta['steps'])
        atomic_json(self.root/'status.json',value)
        return value

    def rebuild(self):
        self.buffer.clear()
        if self.seed:
            from success_seed import insert_seed
            insert_seed(self.buffer,self.args.seed_dir,self.seed)
        for identity in self.meta['replay']:
            _,data=read_journal(self.root/'episodes',identity)
            insert_episode(self.buffer,data,terminal_observation=not self.aligned)

    def begin(self, identity):
        identity=episode_id(identity)
        if self.active==identity and self.phase=='rollout':return self.save_status()
        if self.phase!='ready' or self.active or identity in self.meta['accepted']:
            raise ValueError('Learner is not ready for a new episode')
        import shutil
        if shutil.disk_usage(self.root).free < 16*1024**3:
            raise ValueError('EXPO online NVMe needs at least16GiB free for journal and checkpoint')
        self.active=identity
        self.phase='rollout'
        return self.save_status()

    def cancel_start(self,identity):
        identity=episode_id(identity)
        if (self.root/'episodes'/identity/'started.json').exists():raise ValueError('Existing journal must be finalized')
        if self.active==identity and self.phase=='rollout':
            self.active=None;self.phase='ready'
        elif self.active is not None or self.phase!='ready':raise ValueError('Stale start cancellation')
        return self.save_status()

    def finish(self,identity):
        identity=episode_id(identity)
        if identity in self.meta['accepted']:return self.save_status()
        if self.phase=='updating' and self.active==identity:return self.save_status()
        if self.active is None and self.phase=='ready' and (self.root/'episodes'/identity/'finished.json').is_file():
            self.active=identity
            self.phase='rollout'
        if self.active!=identity or self.phase!='rollout':raise ValueError('Stale episode finish')
        if not (self.root/'episodes'/identity/'finished.json').is_file():raise ValueError('Episode not durably finalized')
        self.phase='updating'
        self.save_status()
        threading.Thread(target=self.recover,args=([identity],),daemon=True).start()
        return self.save_status()

    def recover(self,identities):
        with self.lock:
            try:
                for identity in identities:self.learn_episode(identity)
                self.active=None
                self.phase='ready'
            except Exception as exc:
                self.error=str(exc)
                self.phase='fault'
                traceback.print_exc()
            self.save_status()

    def learn_episode(self,identity):
        import jax
        from stream_checkpoint import save
        if identity in self.meta['accepted']:return
        previous_checkpoint=self.meta['checkpoint']
        previous_updates=self.meta['updates']
        info,data=read_journal(self.root/'episodes',identity)
        if info['actor_version']!=self.meta['updates']:
            raise ValueError('Episode policy version differs from committed learner')
        accepted=data is not None and info.get('training_eligible',True) and info['outcome'] in ('success','failure')
        if accepted:
            self.meta['replay'].append(identity)
            sizes={x:json.loads((self.root/'episodes'/x/'finished.json').read_text())['count']+(0 if self.aligned else 1) for x in self.meta['replay']}
            evicted=False
            while sum(sizes[x] for x in self.meta['replay'])+(self.seed['frames'] if self.seed else 0)>self.args.capacity:
                self.meta['replay'].pop(0)
                evicted=True
            if self.aligned and not evicted:
                insert_episode(self.buffer,data,terminal_observation=False)
            else:self.rebuild()  # Whole episodes, chronological; never sample across ring wrap.
            if self.aligned:
                from sync_schedule import Schedule
                schedule,count=Schedule(self.meta['completed_episodes'],self.meta['steps'],self.meta['credit']).finish(len(data['actions']))
                self.meta.update(completed_episodes=schedule.completed_episodes,steps=schedule.env_steps,credit=schedule.credit)
            else:
                self.meta['steps']+=len(data['actions'])
                self.meta['credit']+=len(data['actions'])
                count=min(self.args.max_updates,self.meta['credit']//self.args.steps_per_update) if len(self.buffer)-len(self.meta['replay'])>=self.args.warmup_steps else 0
            for _ in range(count):
                if self.aligned:
                    if self.native_batch_update:
                        batch=self.buffer._convert_to_openpi_format(self.buffer.sample_jax(64*20))
                    else:
                        from stream_sample import Batch
                        batch=Batch(self.buffer,64*20,64)
                    raw_actor=self.buffer.sample_jax(64,success_only=True)
                    if raw_actor is None:raise ValueError('Confirmed success seed unavailable')
                    actor_batch=self.buffer._convert_to_openpi_format(raw_actor)
                    del raw_actor
                else:
                    batch=valid_batch(self.buffer,self.args.batch_size)
                    if batch is None:break
                    actor_batch=valid_batch(self.buffer,self.args.batch_size,success_only=True)
                self.agent=self.agent.replace(train_base_actor=actor_batch is not None)
                self.agent,metrics=self.updater.update(self.agent,batch,20 if self.aligned else 1,actor_batch)
                jax.block_until_ready(self.agent)
                self.metrics={k:float(np.asarray(v)) for k,v in metrics.items() if np.asarray(v).size==1}
                if not all(np.isfinite(v) for v in self.metrics.values()):raise ValueError('Non-finite native learner metrics')
                self.agent=self.agent.replace(train_base_actor=True)
                self.meta['updates']+=1
                if not self.aligned:self.meta['credit']-=self.args.steps_per_update
                self.meta['critic_updates']=int(self.agent.critic.step)
                self.meta['actor_updates']=int(self.agent.actor_train_state.step)
                self.save_status()
            if self.meta['updates'] != previous_updates:
                self.warm_inference()
                # Use a fresh disk step even after an interrupted uncommitted save.
                disk_step=max([int(p.name) for p in (self.root/'checkpoints').glob('*') if p.name.isdigit()] or [0])+1
                save(self.root/'checkpoints',self.agent,self.updater,disk_step)
                self.meta['checkpoint']=disk_step
        self.meta['accepted'].append(identity)
        self.meta['agent_rng']=np.asarray(self.agent.rng).tolist()
        if hasattr(self.buffer,'rng'):self.meta['replay_rng']=np.asarray(self.buffer.rng).tolist()
        atomic_json(self.meta_path,self.meta)
        # Metadata is committed before deleting older complete checkpoints.
        import shutil
        retained={self.meta['checkpoint'],previous_checkpoint}
        checkpoint_root=(self.root/'checkpoints').resolve()
        for old in checkpoint_root.glob('*'):
            if old.name.isdigit() and int(old.name) not in retained and old.resolve().parent==checkpoint_root:
                shutil.rmtree(old)

    def infer(self,route,data):
        if self.phase!='rollout' or str(data['episode_uuid'].item())!=self.active:
            raise ValueError('Inference requires the active online episode')
        obs={k:data[k] for k in ('state','left_wrist_image','right_wrist_image')}
        from core import validate_observation
        validate_observation(obs)
        version=self.meta['updates']
        if route=='/precache':
            prefix=data['prefix']
            if prefix.shape not in ((0,7),(self.rtc_delay,7)) or not np.isfinite(prefix).all():raise ValueError('Invalid RTC prefix')
            padded=None
            if len(prefix):
                value=self.agent.actor.input_transforms({**obs,'actions':prefix.copy()})['actions']
                padded=np.pad(np.asarray(value),((0,0),(0,25)))
            cache,self.agent=self.agent.sample_pre_cache(dict(obs),padded)
            return dict(precached=np.asarray(cache).tolist(),delay=len(prefix),actor_version=version)
        if int(data['actor_version'])!=version:raise ValueError('Stale actor version')
        cached=data['precached']
        if cached.shape!=(self.agent.N,16,7) or not np.isfinite(cached).all():raise ValueError('Invalid cached candidates')
        actions,self.agent,info=self.agent.sample_actions(dict(obs),precached=cached,delay=int(data['delay']),
            only_base_actions=(version==0 and not self.aligned))
        actions=np.asarray(actions)
        if actions.shape!=(8,7) or not np.isfinite(actions).all():raise ValueError('Non-finite native EXPO output')
        return dict(actions=actions.tolist(),actor_version=version)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--port',type=int,default=8043)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--capacity',type=int,default=6000)
    p.add_argument('--warmup-steps',type=int,default=64);p.add_argument('--steps-per-update',type=int,default=25)
    p.add_argument('--max-updates',type=int,default=8)
    p.add_argument('--recipe',choices=('legacy','native_sync'),default='legacy')
    p.add_argument('--seed-dir',type=Path)
    p.add_argument('--deployment-id')
    p.add_argument('--rtc-delay',type=int,default=2)
    p.add_argument('--native-batch-update',action='store_true')
    args=p.parse_args()
    if args.recipe=='native_sync':
        if args.seed_dir is None:p.error('native_sync requires --seed-dir')
        args.batch_size=64;args.capacity=500000;args.warmup_steps=0;args.steps_per_update=30;args.max_updates=0
    source_path(args.project)
    learner=Learner(args)
    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def send(self,status,value):
            body=json.dumps(value,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path not in ('/ready','/status'):return self.send(404,{'error':'not_found'})
            self.send(200,{**json.loads((learner.root/'status.json').read_text()),'step':2000})
        def do_POST(self):
            if self.path=='/episode/receipt':
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<16384:raise ValueError('Invalid receipt body')
                    identity=episode_id(json.loads(self.rfile.read(length))['episode_uuid'])
                    committed=json.loads(learner.meta_path.read_text()) if learner.meta_path.exists() else {}
                    return self.send(200,{'accepted':identity in committed.get('accepted',[])})
                except Exception as exc:return self.send(409,{'error':str(exc)})
            if not learner.lock.acquire(blocking=False):return self.send(409,{'error':'EXPO learner is updating; wait for the next episode'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2000000:raise ValueError('Invalid body size')
                body=self.rfile.read(length)
                if self.path.startswith('/journal/'):
                    from remote_journal import receive
                    parts=self.path.split('/')
                    if len(parts)!=4:raise ValueError('Invalid journal upload route')
                    result=receive(learner.root/'episodes',learner.active,learner.meta['accepted'],parts[2],parts[3],body,self.headers.get('X-Content-SHA256',''))
                elif self.path in ('/episode/start','/episode/end','/episode/cancel-start'):
                    payload=json.loads(body)
                    method={'/episode/start':learner.begin,'/episode/end':learner.finish,'/episode/cancel-start':learner.cancel_start}[self.path]
                    result=method(payload['episode_uuid'])
                elif self.path in ('/precache','/select'):
                    from wire import decode
                    data=decode(body,self.headers.get('Content-Encoding','identity'))
                    started=time.monotonic()
                    result=learner.infer(self.path,data)
                    result['compute_seconds']=time.monotonic()-started
                else:raise ValueError('Unknown EXPO endpoint')
                self.send(200,result)
            except Exception as exc:self.send(409,{'error':str(exc)})
            finally:learner.lock.release()
    print('EXPO native online Learner ready; no ROS publishers',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
