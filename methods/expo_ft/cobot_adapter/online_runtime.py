"""Web/HIL coordinator for native synchronous EXPO. No motion until operator Start."""
import argparse
import io
import json
import os
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import Request
import numpy as np
from ros_runtime import Runtime
from episode_journal import EpisodeJournal
from online_protocol import episode_id
from remote_runtime_support import RemoteSupport


class OnlineRuntime(RemoteSupport, Runtime):
    def __init__(self,args,ros):
        self.remote_init(args)
        self.journal=EpisodeJournal(args.run_dir/'episodes')
        self.episode=None
        super().__init__(args,ros)
        if not self.options['rtc']:
            raise ValueError('Native online EXPO requires RTC for matching delayed training targets')
        ros.Subscriber('/master/joint_left',self.joint_type,self.sample_callback('executed'),queue_size=1)
        runtime=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send(self,status,value):
                body=json.dumps(value,allow_nan=False).encode();self.send_response(status)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
            def do_GET(self):
                self.send(200,runtime.status_extra())
            def do_POST(self):
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<16384:raise ValueError('Invalid body size')
                    body=json.loads(self.rfile.read(length))
                    if body.get('model_id')!=args.model_id:raise ValueError('Online model identity changed')
                    identity=episode_id(body['episode_uuid'])
                    if self.path=='/episode/start':
                        if not runtime.gate.paused:raise ValueError('Pause before opening an episode')
                        if runtime.transfer:raise ValueError('Previous remote episode transfer/update is pending')
                        if (runtime.journal.root/identity).exists() and not runtime.journal.active:
                            raise ValueError('Episode already exists; recover its outcome before restarting')
                        value=runtime.json_rpc('/episode/start',{'episode_uuid':identity})
                        runtime.journal.begin(identity,value['actor_version'])
                        runtime.episode=identity
                    elif self.path=='/episode/end':
                        if not runtime.gate.paused:raise ValueError('Pause before finishing an episode')
                        if body.get('outcome')=='aborted' and not (runtime.journal.root/identity/'started.json').exists():
                            value=runtime.json_rpc('/episode/cancel-start',{'episode_uuid':identity})
                        else:
                            runtime.journal.finish(identity,body.get('outcome','unknown'))
                            value=runtime.transfer_episode(identity) if runtime.remote else runtime.json_rpc('/episode/end',{'episode_uuid':identity})
                        runtime.episode=None
                    elif self.path in ('/episode/pause','/episode/resume'):
                        if identity!=runtime.episode:raise ValueError('Stale episode identity')
                        runtime.journal.pause(self.path.endswith('pause'))
                        value={'ok':True}
                    else:raise ValueError('Unknown episode operation')
                    with runtime.lock:runtime.save()
                    self.send(200,value)
                except Exception as exc:self.send(409,{'error':str(exc)})
        self.control=ThreadingHTTPServer(('127.0.0.1',args.control_port),Handler)
        threading.Thread(target=self.control.serve_forever,daemon=True).start()
        threading.Thread(target=self.collect,daemon=True).start()
        if self.remote:threading.Thread(target=self.remote_poll,daemon=True).start()

    def start_server(self):
        super().start_server()
        if self.remote:
            with self.rpc.open("http://127.0.0.1:%d/status"%self.args.port,timeout=2) as response:self.local_actor_version=json.load(response)["actor_version"]
        if self.remote and self.transfer:
            threading.Thread(target=self.upload_episode,args=(self.transfer['episode_uuid'],),daemon=True).start()

    def status_extra(self):
        if self.remote:return {'online':self.remote_online_status(),'episode_uuid':self.episode}
        try:value=json.loads((self.args.run_dir/'status.json').read_text())
        except (OSError,ValueError):value={'phase':'loading','ready':False}
        return {'online':value,'episode_uuid':self.episode}

    def policy_command(self):
        if self.remote:
            directory=self.start_remote()
            return [self.remote['python'],str(Path(__file__).with_name('online_actor.py')),
                '--project',self.remote['native_project'],'--bundle',str(self.args.bundle),'--model-id',self.args.model_id,
                '--snapshot-root',str(self.actor_root),'--snapshot',str(directory),'--port',str(self.args.port),'--rtc-delay',str(self.args.rtc_delay)]+(['--edit-scale',str(self.remote['edit_scale'])] if 'edit_scale' in self.remote else [])
        command=[str(self.args.project/'.venv/bin/python'),str(Path(__file__).with_name('online_learner.py')),
                '--project',str(self.args.project),'--bundle',str(self.args.bundle),
                '--run-dir',str(self.args.run_dir),'--port',str(self.args.port),'--rtc-delay',str(self.args.rtc_delay)]
        if getattr(self.args,'recipe','legacy')=='native_sync':
            command+=['--recipe','native_sync','--seed-dir',str(self.args.seed_dir)]
        return command

    def json_rpc(self,path,body):
        if self.remote:
            value=self.remote_rpc(path,body)
            if path=='/episode/start':
                try:self.local_rpc(path,{**body,'actor_version':value['actor_version'],'rollout_rng':value['rollout_rng']})
                except Exception:
                    self.remote_rpc('/episode/cancel-start',body)
                    raise
            elif path=='/episode/cancel-start':self.local_rpc(path,body)
            return value
        request=Request('http://127.0.0.1:%d%s'%(self.args.port,path),data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with self.rpc.open(request,timeout=15) as response:return json.load(response)

    def array_rpc(self,path,**values):
        from wire import encode
        payload,encoding=encode(values,compress=False)
        request=Request('http://127.0.0.1:%d%s'%(self.args.port,path),data=payload,
            headers={'Content-Type':'application/octet-stream','Content-Encoding':encoding})
        with self.rpc.open(request,timeout=30) as response:return json.load(response)

    def infer(self,observation,prefix):
        if not self.episode:raise ValueError('Start a Web collection episode before rollout')
        return self.array_rpc('/precache',**observation,prefix=np.asarray(prefix,np.float32).reshape(-1,7),episode_uuid=self.episode)

    def resolve_inference(self,value):
        if not self.episode:raise ValueError('Online episode has ended')
        payload=self.array_rpc('/select',**self.observation(),episode_uuid=self.episode,
            precached=np.asarray(value['precached'],np.float32),delay=value['delay'],actor_version=value['actor_version'])
        return np.asarray(payload['actions'],np.float32)

    def collect(self):
        deadline=time.monotonic()
        while not self.stop.is_set():
            now=time.monotonic()
            if now<deadline:self.stop.wait(deadline-now)
            if self.stop.is_set():break
            deadline=max(deadline+.1,time.monotonic())
            if not self.journal.active:continue
            try:
                with self.lock:
                    cmd=self.samples.get('executed')
                    mode=self.gate.mode
                    human=mode.startswith('manual:') and 'left' in mode
                    valid=(not self.gate.manual_pause and not self.gate.fault and
                           ((not self.gate.paused and self.chunk_ready) or human))
                if not valid:
                    self.journal.sample(None,None,time.monotonic(),human,False)
                    continue
                if cmd is None or time.monotonic()-cmd[1]>.15:
                    raise ValueError('Executed coordinator command is stale')
                msg=cmd[0]
                if list(msg.name)!=['joint%d'%i for i in range(7)]:raise ValueError('Coordinator command joint order changed')
                stamp=msg.header.stamp.to_sec()
                if stamp and self.ros.Time.now().to_sec()-stamp>.15:raise ValueError('Coordinator command timestamp expired')
                obs=self.observation()
                self.journal.sample(obs,np.asarray(msg.position,np.float32),time.monotonic(),human)
            except Exception as exc:
                self.journal.sample(None,None,time.monotonic(),False,False)
                self.fail('EXPO online capture: '+str(exc))

    def close(self):
        if hasattr(self,'control'):self.control.shutdown()
        super().close()
        self.close_remote()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--step',type=int,default=2000)
    p.add_argument('--model-id',required=True);p.add_argument('--port',type=int,default=8043)
    p.add_argument('--control-port',type=int,default=8044);p.add_argument('--shadow',action='store_true')
    p.add_argument('--recipe',choices=('legacy','native_sync'),default='legacy')
    p.add_argument('--seed-dir',type=Path)
    p.add_argument('--remote-config',type=Path)
    p.add_argument('--rtc-delay',type=int,default=2)
    args=p.parse_args()
    if args.recipe=='native_sync' and args.seed_dir is None and args.remote_config is None:p.error('native_sync requires --seed-dir')
    import rospy
    runtime=OnlineRuntime(args,rospy)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:runtime.stop.set())
    try:runtime.run()
    except Exception as exc:runtime.fail(exc);raise
    finally:runtime.close()

if __name__=='__main__':main()
