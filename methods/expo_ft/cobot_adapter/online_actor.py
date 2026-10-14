"""Cobot inference-only native EXPO actor. No Replay, updates or ROS publishers."""
import argparse
import io
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from native_online import source_path,load_native
from online_protocol import episode_id,atomic_json
from online_learner import Learner
from actor_snapshot import contract,restore,strip_training
from wire import decode

class Actor:
    infer=Learner.infer
    warm_inference=Learner.warm_inference

    def __init__(self,args):
        self.args=args;self.rtc_delay=args.rtc_delay;self.aligned=True
        self.active=None;self.phase='loading';self.error=None
        self.meta={'updates':-1};self.lock=threading.Lock()
        self.edit_scale=getattr(args,"edit_scale",None)
        self.agent,_,_=load_native(args.bundle,args.rtc_delay,aligned=True,edit_scale=self.edit_scale)
        self.agent=strip_training(self.agent)
        self.identity=contract(args.bundle,args.project,args.rtc_delay,self.edit_scale)
        self.install(args.snapshot)

    def status(self):
        return dict(ready=self.phase in ('ready','rollout'),phase=self.phase,error=self.error,active_episode=self.active,
                    actor_version=self.meta['updates'],deployment_id=self.args.model_id,rtc_delay=self.rtc_delay,
                    update_layout='native_full_batch',compute_host='Cobot',training_enabled=False,step=2000,edit_scale=float(self.agent.edit_scale))

    def install(self,directory):
        if self.active:raise ValueError('Cannot replace Actor during an episode')
        directory=Path(directory).resolve()
        if directory.parent!=self.args.snapshot_root.resolve():raise ValueError('Actor path outside configured snapshots')
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['actor_version']<self.meta['updates']:raise ValueError('Actor version rollback is forbidden')
        self.phase='loading'
        try:
            agent,manifest=restore(directory,self.agent,self.identity)
            self.agent=agent
            self.warm_inference()
            self.meta['updates']=manifest['actor_version'];self.phase='ready';self.error=None
        except Exception as exc:
            self.phase='fault';self.error=str(exc);raise
        return self.status()

    def begin(self,data):
        import jax
        identity=episode_id(data['episode_uuid'])
        if self.active==identity and self.phase=='rollout':return self.status()
        if self.active or self.phase!='ready' or int(data['actor_version'])!=self.meta['updates']:raise ValueError('Local Actor is not ready for this version')
        rng=np.asarray(data['rollout_rng'])
        if rng.shape!=(2,) or not np.issubdtype(rng.dtype,np.integer) or np.any(rng<0) or np.any(rng>2**32-1):raise ValueError('Invalid rollout RNG')
        self.agent=self.agent.replace(rng=jax.device_put(rng.astype(np.uint32),self.agent.rng.sharding))
        self.active=identity;self.phase='rollout'
        return self.status()

    def finish(self,data):
        identity=episode_id(data['episode_uuid'])
        receipt=self.args.snapshot_root.parent/'actor_receipts'/(identity+'.json')
        if receipt.exists():return json.loads(receipt.read_text())
        if identity!=self.active:raise ValueError('Stale local Actor episode')
        value=dict(rollout_rng=np.asarray(self.agent.rng).tolist(),actor_version=self.meta['updates'])
        atomic_json(receipt,value)
        self.active=None;self.phase='ready'
        return value


def main():
    p=argparse.ArgumentParser()
    for name in ('project','bundle','snapshot-root','snapshot'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-id',required=True);p.add_argument('--rtc-delay',type=int,default=5);p.add_argument('--port',type=int,default=8043)
    p.add_argument('--edit-scale',type=float)
    args=p.parse_args();source_path(args.project);actor=Actor(args)
    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def send(self,status,value):
            body=json.dumps(value,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Length',str(len(body)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
        def do_GET(self):self.send(200,actor.status())
        def do_POST(self):
            acquired=actor.lock.acquire(blocking=False) if self.path in ('/precache','/select') else actor.lock.acquire(timeout=5)
            if not acquired:return self.send(409,{'error':'Actor is busy'})
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<2*1024*1024:raise ValueError('Invalid body')
                body=self.rfile.read(length)
                if self.path in ('/precache','/select'):
                    started=time.monotonic();result=actor.infer(self.path,decode(body,self.headers.get('Content-Encoding','identity')))
                    result['compute_seconds']=time.monotonic()-started
                else:
                    data=json.loads(body)
                    if self.path=='/episode/start':result=actor.begin(data)
                    elif self.path=='/episode/cancel-start' and actor.active is None:result=actor.status()
                    elif self.path in ('/episode/end','/episode/cancel-start'):result=actor.finish(data)
                    elif self.path=='/actor/install':result=actor.install(data['directory'])
                    else:raise ValueError('Unknown Actor operation')
                self.send(200,result)
            except Exception as exc:self.send(409,{'error':str(exc)})
            finally:actor.lock.release()
    print('EXPO local inference-only Actor ready; no optimizer updates or ROS publishers',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
