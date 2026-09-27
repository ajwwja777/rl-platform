"""Control-interpreter client; JAX/Torch stay in an owned CPU-only worker."""
import json,os,subprocess,time,socket,hashlib
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';REG=RUN/'backend/rtc-actor-process.json'
def request(path,body,port=8023):
    req=Request(f'http://127.0.0.1:{port}'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Connection':'close'},method='POST')
    with build_opener(ProxyHandler({})).open(req,timeout=20) as r:data=json.load(r)
    if data.get('error'):raise RuntimeError(data['error'])
    return data
class Client:
    def __init__(self,key,port=8023):self.key=key;self.port=port
    def act(self,z,context,ref):
        data=request('/api/infer',dict(key=self.key,z=z.tolist(),context=context.tolist(),ref=ref.tolist()),self.port)
        action=np.asarray(data['action'],np.float32)
        if action.shape!=(10,14) or not np.isfinite(action).all():raise ValueError('bad actor output')
        return action

def select(release,port=8023):
    data=request('/api/select',{'release':str(release)},port)
    if data.get('protocol')!='upstream-rtc-v1':raise ValueError('actor protocol mismatch')
    return Client(data['key'],port),int(data['actor_version'])

def ensure_worker():
    from .cli import alive,identity,occupied
    if occupied(8023):
        record=json.loads(REG.read_text())
        if not alive(record) or b'methods.openpi_rlt.plug_v2.rtc_actor_service' not in (Path('/proc')/str(record['pid'])/'cmdline').read_bytes():raise RuntimeError('port8023 is not our registered actor')
        return None
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    log=RUN/'backend/rtc-actor.log'
    with log.open('a') as f:p=subprocess.Popen(['/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python','-u','-m','methods.openpi_rlt.plug_v2.rtc_actor_service'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    from .cli import atomic
    atomic(REG,{'pid':p.pid,'start_ticks':identity(p.pid),'command':'methods.openpi_rlt.plug_v2.rtc_actor_service'})
    deadline=time.monotonic()+30
    while not occupied(8023):
        if p.poll() is not None:raise RuntimeError('upstream actor worker failed; see '+str(log))
        if time.monotonic()>deadline:p.terminate();p.wait(timeout=10);raise RuntimeError('upstream actor worker startup timeout')
        time.sleep(.1)
    return p
