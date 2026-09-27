"""Registered loopback CPU actor worker. No ROS imports or publishers."""
import argparse,json,os,threading,hashlib,time,queue
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from .upstream_actor import UpstreamActor,ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8023);args=parser.parse_args()
    cache={};lock=threading.RLock();jobs=queue.Queue(maxsize=1);decision=[0]
    telemetry=ROOT/'runs/plug_v2/learning/telemetry';telemetry.mkdir(parents=True,exist_ok=True)
    def append(path,row):
        if path.exists() and path.stat().st_size>2*1024*1024:
            lines=path.read_text(errors='replace').splitlines()[-500:]
            temporary=path.with_suffix(path.suffix+'.tmp')
            temporary.write_text('\n'.join(lines)+'\n')
            os.replace(str(temporary),str(path))
        with path.open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n')
    def diagnostic_worker():
        while True:
            item=jobs.get()
            if item is None:return
            actor,z,context,ref,plan,number=item
            try:
                diag=actor.diagnose(z,context,ref,plan);base={'decision':number,'time':time.time(),'actor_version':actor.version,'learner_step':actor.learner_step}
                append(telemetry/'decision_q.jsonl',{**base,'actor_q':diag['actor_q'],'reference_q':diag['reference_q']})
                append(telemetry/'actor_delta.jsonl',{**base,'horizon_rms':diag['horizon_rms'],'joint_rms':diag['joint_rms'],'overall_rms':diag['overall_rms']})
            except Exception as exc:
                append(telemetry/'diagnostic_errors.jsonl',{'time':time.time(),'decision':number,'error':str(exc)})
    threading.Thread(target=diagnostic_worker,name='critic-telemetry',daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):
            code=200
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=1024*1024:raise ValueError('invalid body')
                body=json.loads(self.rfile.read(n))
                if self.path=='/api/select':
                    path=Path(body['release']).resolve(strict=True);path.relative_to((ROOT/'runs/plug_v2/learning/rtc-v5/releases').resolve())
                    release=json.loads(path.read_text());key=hashlib.sha256(json.dumps(release,sort_keys=True).encode()).hexdigest()
                    with lock:
                        if key not in cache:
                            cache[key]=UpstreamActor(release)
                            while len(cache)>2:cache.pop(next(iter(cache)))
                        actor=cache[key]
                    result={'key':key,'actor_version':actor.version,'learner_step':actor.learner_step,'pid':os.getpid(),'protocol':'upstream-rtc-v1'}
                elif self.path=='/api/infer':
                    with lock:actor=cache[body['key']]
                    inputs=[np.asarray(body[k],np.float32) for k in ('z','context','ref')];action=actor.act(*inputs)
                    decision[0]+=1
                    if decision[0]%10==0:
                        try:jobs.put_nowait((actor,*[x.copy() for x in inputs],action.copy(),decision[0]))
                        except queue.Full:pass
                    result={'action':action.tolist(),'actor_version':actor.version}
                else:raise ValueError('unknown endpoint')
            except Exception as e:code=409;result={'error':str(e)}
            raw=json.dumps(result).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.send_header('Connection','close');self.end_headers();self.wfile.write(raw)
    print('UPSTREAM_ACTOR_SERVICE_READY',os.getpid(),args.port,flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
if __name__=='__main__':main()
