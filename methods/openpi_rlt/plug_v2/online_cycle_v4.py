"""Paused five-episode batches, versioned publication, durable rejection accounting."""
import json,time,os,fcntl,datetime,subprocess,sys
from pathlib import Path
from .training_flow import atomic,session_phase,LearningDeferred
from .storage import releases_for_phase
from .online_release import selected,CONTRACT,digest
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
SAFE_PHASES=('waiting_scene','stopped','offline','disarmed','ready')
def publish(path,report):
    path=Path(path).resolve()
    import torch
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if (checkpoint.get('learning_algorithm')!='supported_iql_residual_v1' or checkpoint.get('status')!='accepted' or checkpoint.get('execution_contract')!=CONTRACT
        or checkpoint.get('global_step')!=report.get('selected_step') or not report.get('accepted')):
        raise ValueError('unvalidated checkpoint publication')
    atomic(RUN/'learning/v4/current.json',{'status':'offline_validated_onsite_pending',
        'execution_contract':CONTRACT,'learning_algorithm':'supported_iql_residual_v1','path':str(path),'sha256':digest(path),
        'actor_version':report['selected_step'],'parent_version':report.get('parent_version'),
        'autonomous_improvement_verified':False,'report':str(path.parent/'report.json'),
        'published_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
def available():
    out={}
    for p in releases_for_phase('online'):
        d=json.loads(p.read_text())
        if d.get('status')=='validated' and d.get('cohort')=='plug_v2' and d.get('training_frames',0)>=7:
            if d.get('metadata',{}).get('outcome') in ('success','failure'):out[d['source_uuid']]=p
    return out
def work(uuids):
    folder=RUN/'learning';status=folder/'operation.json'
    if session_phase() not in SAFE_PHASES:raise LearningDeferred('wait for completed episode')
    with (folder/'operation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise LearningDeferred('training/start lease busy')
        if session_phase() not in SAFE_PHASES:raise LearningDeferred('wait for completed episode')
        initial,release=selected()
        atomic(status,{'phase':'preparing','pid':os.getpid(),'command':'online-v4','new_uuids':uuids,
                       'execution_contract':CONTRACT,'actor_version':release['actor_version']})
        try:
            subprocess.run([sys.executable,'-u','-m','methods.openpi_rlt.plug_v2.replay','prepare'],cwd=ROOT,check=True)
            import numpy as np
            counts={}
            for p in (RUN/'replay/v2').glob('*.npz'):
                with np.load(p,allow_pickle=False) as f:
                    m=json.loads(str(f['metadata']))
                    if m['source_uuid'] in uuids and m['split']=='train':counts[m['source_uuid']]=len(f['z'])
            if not sum(counts.values()):raise ValueError('new batch has no eligible training transitions; no update')
            steps=min(2000,max(100,5*sum(counts.values())))
            run=folder/'v4/batches'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            atomic(status,{'phase':'training','pid':os.getpid(),'command':'online-v4','steps':steps,
                           'new_training_episodes':len(counts),'new_uuids':uuids,'execution_contract':CONTRACT})
            from .supported_learner import train
            path,report=train('online',steps,run,initial)
            if selected()[1]['sha256']!=release['sha256']:raise ValueError('base version changed during update')
            report.update(parent_version=release['actor_version'],new_uuids=uuids,new_training_episodes=len(counts))
            (run/'report.json').write_text(json.dumps(report,indent=2))
            publish(path,report)
            receipt={'phase':'accepted','command':'online-v4','actor_version':report['selected_step'],
                     'learner_final_step':release['actor_version']+steps,'new_training_episodes':len(counts),
                     'updates':steps,'seconds':report['seconds'],'execution_contract':CONTRACT,'report':str(run/'report.json')}
            atomic(status,receipt);return receipt
        except Exception as e:
            atomic(status,{'phase':'rejected','command':'online-v4','error':str(e),'new_uuids':uuids,
                           'retained_actor_version':release['actor_version'],'execution_contract':CONTRACT})
            raise
def cycle(episodes=5):
    p=RUN/'learning/v4/cycle.json'
    previous=json.loads(p.read_text()) if p.exists() else {}
    consumed=set(previous.get('consumed_uuids',[]));rejected=set(previous.get('rejected_uuids',[]))
    # Recover a crash after pointer publication but before cycle-accounting commit.
    _,release=selected()
    receipt=json.loads(Path(release['report']).read_text())
    consumed.update(receipt.get('new_uuids',[]));rejected.difference_update(consumed)
    while True:
        # End Session never starts a background update.
        if session_phase()!='waiting_scene':time.sleep(1);continue
        sources=available();fresh=set(sources)-consumed-rejected
        if len(fresh)<episodes:
            atomic(p,{'phase':'idle','pending_episodes':len(fresh),'episodes_per_update':episodes,
                      'consumed_uuids':sorted(consumed),'rejected_uuids':sorted(rejected)})
            time.sleep(1);continue
        batch=sorted(fresh|(rejected&set(sources)))
        atomic(p,{'phase':'preparing','new_uuids':batch,'consumed_uuids':sorted(consumed),'rejected_uuids':sorted(rejected)})
        try:
            receipt=work(batch)
        except LearningDeferred:time.sleep(1);continue
        except Exception as e:
            rejected.update(batch);receipt={'phase':'rejected','error':str(e)}
        else:
            consumed.update(batch);rejected.difference_update(batch)
        atomic(p,{**receipt,'consumed_uuids':sorted(consumed),'rejected_uuids':sorted(rejected),
                  'episodes_per_update':episodes})
        print('ONLINE_V4',json.dumps(receipt),flush=True)
