# Loopback CPU actor service: keep Torch outside the ROS control interpreter.
import argparse,json,hashlib,threading,os
from pathlib import Path
from urllib.request import Request,ProxyHandler,build_opener
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
URL='http://127.0.0.1:8021'
def request(path,data):
    req=Request(URL+path,data=json.dumps(data).encode(),method='POST',
                headers={'Content-Type':'application/json','Connection':'close'})
    with build_opener(ProxyHandler({})).open(req,timeout=3) as response:payload=json.load(response)
    if payload.get('error'):raise RuntimeError(payload['error'])
    return payload
class ActorClient:
    def __init__(self,key):self.key=key
    def act(self,z,context,ref):
        payload=request('/api/infer',{'key':self.key,'z':z.tolist(),'context':context.tolist(),'ref':ref.tolist()})
        action=np.asarray(payload['action'],np.float32)
        if action.shape!=(10,14) or not np.isfinite(action).all():raise ValueError('invalid actor result')
        return action
def select_actor(path):
    response=request('/api/select',{'path':str(Path(path).resolve())})
    return ActorClient(response['key']),int(response['actor_version'])
def main():
    import torch
    from methods.openpi_rlt.plug_v2.learning import load_actor
    torch.set_num_threads(2)
    actors={};lock=threading.RLock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            code=200
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=1024*1024:raise ValueError('invalid body')
                body=json.loads(self.rfile.read(n))
                if self.path=='/api/select':
                    path=Path(body['path']).resolve(strict=True)
                    path.relative_to((ROOT/'runs/plug_v2/learning').resolve())
                    key=hashlib.sha256(path.read_bytes()).hexdigest()
                    with lock:
                        if key not in actors:
                            actor,version=load_actor(path)
                            actors[key]=(actor,version)
                        actor,version=actors[key]
                    result={'key':key,'actor_version':version,'cohort':'plug_v2'}
                elif self.path=='/api/infer':
                    with lock:actor,version=actors[body['key']]
                    values=[np.asarray(body[k],np.float32) for k in ('z','context','ref')]
                    if [v.shape for v in values]!=[(2048,),(99,),(10,14)] or not all(np.isfinite(v).all() for v in values):
                        raise ValueError('invalid actor input')
                    with torch.no_grad():action=actor.act(*values)
                    result={'action':action.tolist(),'actor_version':version}
                else:code=404;result={'error':'not_found'}
            except Exception as error:code=409;result={'error':str(error)}
            raw=json.dumps(result).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.send_header('Connection','close');self.end_headers();self.wfile.write(raw)
    print('ACTOR_SERVICE_READY',os.getpid(),flush=True)
    ThreadingHTTPServer(('127.0.0.1',8021),Handler).serve_forever()
if __name__=='__main__':main()
