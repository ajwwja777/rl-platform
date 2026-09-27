# Manual warmup and paused batch online updates, isolated from robot control.
import argparse,json,os,sys,time,subprocess,fcntl,socket
from pathlib import Path
from methods.openpi_rlt.plug_v2.storage import releases_for_phase
from urllib.request import ProxyHandler,build_opener
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/plug_v2'
BASE=ROOT.parent.parent/'data/rlt/plug_v2'
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
class LearningDeferred(RuntimeError):pass
def session_phase():
    try:
        with build_opener(ProxyHandler({})).open('http://127.0.0.1:8026/api/session',timeout=2) as r:
            return json.load(r)['phase']
    except OSError:
        with socket.socket() as sock:
            return 'unknown' if sock.connect_ex(('127.0.0.1',8026))==0 else 'offline'
def rtc_training_idle():
    phase=session_phase()
    if phase in ('stopped','offline'):return True
    # Only the batch coordinator holding the rollout lease may train between episodes.
    if phase not in ('waiting_scene','ready','disarmed'):return False
    try:
        owner=int(os.environ.get('RLT_RTC_LEASE_OWNER','0'))
        status=json.loads((RUN/'learning/operation.json').read_text())
        if owner<=0 or status.get('pid')!=owner:return False
        if b'methods.openpi_rlt.plug_v2.training_flow' not in (Path('/proc')/str(owner)/'cmdline').read_bytes():return False
        with (RUN/'learning/operation.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
        return False
    except (OSError,ValueError,KeyError):return False

def work(command,steps=20000,new_uuids=None):
    if session_phase() not in ('offline','disarmed','ready','paused','terminal_pending','waiting_scene','stopped','fault'):
        raise LearningDeferred('pause/end Session before learning')
    folder=RUN/'learning';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'operation.json'
    with (folder/'operation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise LearningDeferred('another operation or rollout start holds the lease')
        if session_phase() not in ('offline','disarmed','ready','paused','terminal_pending','waiting_scene','stopped','fault'):
            raise LearningDeferred('pause/end Session before learning')
        atomic(status,{'phase':'preparing','command':command,'pid':os.getpid(),'automatic_warmup':False})
        try:
            env=os.environ.copy()
            env['PYTHONPATH']=':'.join([str(ROOT/'envs/machine-a-py311-overlay'),str(ROOT),
                                      str(ROOT/'code/openpi-rlt/src')])
            env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
            # Preparation uses the already loaded frozen model over loopback.
            cmd=[sys.executable,'-u','-m','methods.openpi_rlt.plug_v2.replay','prepare']
            subprocess.run(cmd,env=env,check=True,cwd=ROOT)
            if new_uuids is not None:
                import numpy as np
                transitions=0
                for path in (RUN/'replay/v2').glob('*.npz'):
                    if path.name.endswith('.source.npz'):continue
                    with np.load(path,allow_pickle=False) as data:
                        meta=json.loads(str(data['metadata']))
                        if meta['source_uuid'] in new_uuids and meta['split']=='train':
                            transitions+=len(data['z'])
                if transitions<=0:raise RuntimeError('no new eligible training transitions')
                steps=min(20000,5*transitions)
            # Do not inherit CUDA_VISIBLE_DEVICES='' from the control supervisor.
            env['CUDA_VISIBLE_DEVICES']='0'
            atomic(status,{'phase':'training','command':command,'pid':os.getpid(),'steps':steps,
                           'automatic_warmup':False})
            cmd=[sys.executable,'-u','-m','methods.openpi_rlt.plug_v2.learning',command,'--steps',str(steps)]
            subprocess.run(cmd,env=env,check=True,cwd=ROOT)
            ready=json.loads((folder/command/'ready.json').read_text())
            atomic(status,{'phase':'accepted','command':command,**ready,'updates':steps,'automatic_warmup':False})
            return steps
        except Exception as error:
            atomic(status,{'phase':'rejected','command':command,'error':str(error),'automatic_warmup':False})
            raise
def cycle(args):
    path=RUN/'learning/online_cycle.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    consumed=set(previous.get('consumed_uuids',[]))
    while True:
        phase=session_phase()
        if phase not in ('offline','disarmed','ready','paused','terminal_pending','waiting_scene','stopped','fault'):
            time.sleep(1);continue
        if not (RUN/'learning/warmup/actor.pt').exists():time.sleep(1);continue
        available={}
        for p in releases_for_phase('online'):
            item=json.loads(p.read_text())
            if item['cohort']=='plug_v2' and item['training_frames']>=7:
                available[item['source_uuid']]=item
        new=set(available)-consumed
        if len(new)<args.episodes:
            atomic(path,{'phase':'idle','pending_episodes':len(new),'episodes_per_update':args.episodes,
                         'consumed_uuids':sorted(consumed),'automatic_warmup':False})
            time.sleep(1);continue
        selected=sorted(new)
        # Source transition count is computed after actual validated replay preparation.
        atomic(path,{'phase':'preparing','new_uuids':selected,'episodes_per_update':args.episodes,
                     'consumed_uuids':sorted(consumed),'automatic_warmup':False})
        started=time.monotonic()
        try:
            updates=work('online',new_uuids=selected)
            outcome='accepted'
        except LearningDeferred:
            time.sleep(1);continue
        except Exception as error:
            updates=0
            print('ONLINE_BATCH_REJECTED',str(error),flush=True);outcome='rejected'
        consumed.update(selected)
        atomic(path,{'phase':outcome,'new_uuids':selected,'updates':updates,
                     'seconds':time.monotonic()-started,'consumed_uuids':sorted(consumed),
                     'episodes_per_update':args.episodes,'automatic_warmup':False})
        print('ONLINE_BATCH',outcome,len(selected),updates,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['warmup','cycle'])
    p.add_argument('--episodes',type=int,default=5)
    args=p.parse_args()
    if not 1<=args.episodes<=20:raise ValueError('invalid rollout batch')
    if args.command=='warmup':
        manifest=json.loads((ROOT/'deployments/plug_v2/manifest.json').read_text())
        receipt=json.loads((RUN/'model-server/validation.json').read_text())
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',8020))!=0:
                raise RuntimeError('model is offline; run ui_up/rlt_up --reference, wait ready, then End Session without rlt_down')
        if manifest.get('status')!='offline_validated' or receipt.get('status')!='passed' or receipt['metadata']['checkpoint']!=manifest['checkpoint']:
            raise RuntimeError('validated matching frozen Stage-1 model required')
        work('warmup')
    else:
        if (RUN/'learning/rtc-v5/current.json').exists():
            from .rtc_online_cycle import cycle as cycle_v5
            cycle_v5(args.episodes)
        elif (RUN/'learning/v4/current.json').exists():
            from .online_cycle_v4 import cycle as cycle_v4
            cycle_v4(args.episodes)
        else:cycle(args)
if __name__=='__main__':main()
