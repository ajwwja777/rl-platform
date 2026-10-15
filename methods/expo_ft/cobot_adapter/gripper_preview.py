"""Matched Actor11 original/feedback-opening previews; no Learner or Replay."""
import argparse
import json
from pathlib import Path
import signal
import threading
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import Request
import numpy as np
from ros_runtime import Runtime
from gripper_feedback import FeedbackPlan
import time

PREVIEW_SCALES={
    'expo-plug-actor11-gripper-original':.01,
    'expo-plug-actor11-gripper-feedback':.01,
}

def preview_scale(model_id):
    try:return PREVIEW_SCALES[model_id]
    except KeyError:raise ValueError('Unknown frozen EXPO preview identity')


class PreviewRuntime(Runtime):
    def __init__(self,args,ros):
        self.episode=str(uuid.uuid4())
        super().__init__(args,ros)

    def make_plan(self, actions, last):
        if not self.args.model_id.endswith('-feedback'):
            return super().make_plan(actions, last)
        self.feedback_plan = FeedbackPlan(actions,last,self.options['smoothing'],self.options['rtc'],self.rtc_delay)
        return self.feedback_plan.plan

    def prepare_plan(self, plan, index, last):
        if not self.args.model_id.endswith('-feedback'):
            return plan
        with self.lock:
            feedback = self.samples.get('state')
            if feedback is None or time.monotonic()-feedback[1] > .25:
                raise ValueError('Left feedback expired before gripper planning')
            stamp = feedback[0].header.stamp.to_sec()
            if stamp and self.ros.Time.now().to_sec()-stamp > .25:
                raise ValueError('Left feedback timestamp expired before gripper planning')
            actual = float(feedback[0].position[6])
        return self.feedback_plan.prepare(index,last,actual)

    def interpolate_command(self, last, target, fraction, index):
        if not self.args.model_id.endswith('-feedback'):
            return super().interpolate_command(last,target,fraction,index)
        return self.feedback_plan.interpolate(last,target,fraction,index)

    def policy_command(self):
        return [str(self.args.project/'.venv/bin/python'),str(Path(__file__)), 'server',
                '--bundle',str(self.args.bundle),'--project',str(self.args.project),
                '--native-project',str(self.args.native_project),'--snapshot',str(self.args.snapshot),
                '--model-id',self.args.model_id,'--port',str(self.args.port),'--episode',self.episode]

    def validate_server_status(self,info):
        if (info.get('deployment_id')!=self.args.model_id or info.get('actor_version')!=11 or
            info.get('rtc_delay')!=5 or info.get('edit_scale')!=preview_scale(self.args.model_id) or
            info.get('training_enabled') is not False or info.get('frozen_preview') is not True):
            raise RuntimeError('Frozen EXPO preview identity mismatch')

    def status_extra(self):
        return {'frozen_preview':True,'edit_scale':preview_scale(self.args.model_id),'rtc_delay':5,'training_enabled':False,'actor_version':11,'gripper_feedback':self.args.model_id.endswith('-feedback')}

    def array_rpc(self,path,**values):
        from wire import encode
        payload,encoding=encode(values,compress=False)
        request=Request('http://127.0.0.1:%d%s'%(self.args.port,path),data=payload,
            headers={'Content-Type':'application/octet-stream','Content-Encoding':encoding})
        with self.rpc.open(request,timeout=30) as response:return json.load(response)

    def infer(self,observation,prefix):
        return self.array_rpc('/precache',**observation,prefix=np.asarray(prefix,np.float32).reshape(-1,7),episode_uuid=self.episode)

    def resolve_inference(self,value):
        payload=self.array_rpc('/select',**self.observation(),episode_uuid=self.episode,
            precached=np.asarray(value['precached'],np.float32),delay=value['delay'],actor_version=value['actor_version'])
        return np.asarray(payload['actions'],np.float32)


def serve(args):
    from native_online import source_path
    source_path(args.native_project)
    from online_actor import Actor
    from wire import decode
    if json.loads((args.snapshot/'manifest.json').read_text())['actor_version']!=11:
        raise ValueError('Gripper comparison accepts only immutable Actor11')
    native_args=argparse.Namespace(bundle=args.bundle,project=args.native_project,
        snapshot=args.snapshot,snapshot_root=args.snapshot.parent,rtc_delay=5,model_id=args.model_id,edit_scale=.01)
    actor=Actor(native_args)
    assert float(actor.agent.edit_scale)==preview_scale(args.model_id)
    actor.active=args.episode;actor.phase='rollout'
    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def send(self,code,value):
            body=json.dumps(value,allow_nan=False).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path not in ('/ready','/status'):return self.send(404,{'error':'not_found'})
            self.send(200,{**actor.status(),'frozen_preview':True,'edit_scale':preview_scale(args.model_id)})
        def do_POST(self):
            if self.path not in ('/precache','/select'):return self.send(404,{'error':'Frozen preview has no training or installation endpoint'})
            if not actor.lock.acquire(blocking=False):return self.send(409,{'error':'Actor busy'})
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=2*1024*1024:raise ValueError('Invalid body')
                data=decode(self.rfile.read(length),self.headers.get('Content-Encoding','identity'))
                self.send(200,actor.infer(self.path,data))
            except Exception as exc:self.send(409,{'error':str(exc)})
            finally:actor.lock.release()
    print('EXPO frozen edit=%s preview ready; no Learner/Replay/ROS publishers in server' % preview_scale(args.model_id),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=('runtime','server'))
    for name in ('bundle','project','native-project','snapshot'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-id',required=True);p.add_argument('--port',type=int,default=8043)
    p.add_argument('--episode',default=str(uuid.uuid4()));p.add_argument('--shadow',action='store_true')
    args=p.parse_args();preview_scale(args.model_id);args.step=2000;args.rtc_delay=5
    if args.role=='server':return serve(args)
    import rospy
    runtime=PreviewRuntime(args,rospy)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:runtime.stop.set())
    try:runtime.run()
    except Exception as exc:runtime.fail(exc);raise
    finally:runtime.close()

if __name__=='__main__':main()
