"""Paused five-episode original-RLT updates, fixed normalization, atomic releases."""
import os,json,time,fcntl,subprocess,shutil,hashlib,math
from pathlib import Path
from .training_flow import ROOT,RUN,atomic,session_phase,LearningDeferred
from .rtc_release import selected,digest,CURRENT
from .online_value_gate import fixed_value_gate
IDLE=('waiting_scene','stopped','offline','ready','disarmed')
PY='/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python'

def burnin_for_release(release,steps,required=1000):
    completed=max(0,int(release.get('critic_burnin_updates',0)))
    if bool(release.get('critic_calibrated',False)):completed=max(completed,int(required))
    return min(int(steps),max(0,int(required)-completed))

def pending(available,consumed,attempted,minimum):
    # Rejected batches remain auditable but are quarantined from automatic
    # retries.  Train one fixed-size batch from UUIDs never attempted before.
    candidates=sorted(set(available)-set(consumed)-set(attempted))
    return candidates[:minimum] if len(candidates)>=minimum else []

def regression_gate(baseline,after):
    reasons=[]
    for delay in ('0','6'):
        a=baseline['summary'][delay];b=after['summary'][delay]
        for key in ('deployment_matched_mse','deployment_accel_rms'):
            if not math.isfinite(b[key]) or b[key]>a[key]*1.05+1e-10: reasons.append(delay+':'+key)
        if b['deployment_matched_mse']>=b['ref_matched_conditioner_mse']: reasons.append(delay+':reference_regression')
    reasons.extend(fixed_value_gate(after,min_count=3))
    return reasons

def available_uuids():
    from .storage import releases_for_phase
    result=set()
    for phase in ('warmup','online'):
        for p in releases_for_phase(phase):
            d=json.loads(p.read_text())
            if d.get('cohort')=='plug_v2' and d.get('status')=='validated' and d.get('metadata',{}).get('outcome') in ('success','failure'):
                result.add(d['source_uuid'])
    return result

def run_batch(release_path,release,uuids):
    folder=RUN/'learning';status=folder/'operation.json'
    with (folder/'operation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise LearningDeferred('rollout or learning lease busy')
        if session_phase() not in IDLE:raise LearningDeferred('wait for completed episode')
        tag=time.strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid())
        data=RUN/'diagnostics'/('rtc-online-'+tag);data.mkdir();(data/'episodes').mkdir()
        # Immutable old replay files remain shared; only new source UUIDs are prepared.
        for p in (Path(release['dataset'])/'episodes').glob('*.npz'):os.link(p,data/'episodes'/p.name)
        env=os.environ.copy();env.update(RLT_RTC_LEASE_OWNER=str(os.getpid()),JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.10',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',XDG_CACHE_HOME=str(folder/'rtc-v5/cache'))
        def execute(module,args):
            with (data/'operation.log').open('a') as log:subprocess.run([PY,'-u','-m','methods.openpi_rlt.plug_v2.'+module]+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        telemetry=folder/'telemetry';telemetry.mkdir(exist_ok=True)
        env['RLT_TELEMETRY_PROGRESS_PATH']=str(telemetry/'progress.json')
        atomic(status,dict(phase='preparing',pid=os.getpid(),new_uuids=uuids,automatic_warmup=False))
        atomic(telemetry/'progress.json',dict(phase='replay_build',completed=0,total=1,message='正在构建固定 replay'))
        try:
            execute('rtc_upstream_data',['--output',str(data)])
            manifest=json.loads((data/'manifest.json').read_text())
            transitions=sum(m['td_valid_rows'] for m in manifest if m['uuid'] in uuids and m['split']=='train')
            if transitions<=0:raise ValueError('no new training-split factual transitions')
            steps=min(2000,5*transitions);out=folder/('rtc-online-'+tag)
            atomic(status,dict(phase='training',pid=os.getpid(),updates=steps,new_transitions=transitions,new_uuids=uuids,automatic_warmup=False))
            atomic(telemetry/'progress.json',dict(phase='training',completed=0,total=steps,message='critic / actor training'))
            burnin=burnin_for_release(release,steps);execute('rtc_upstream_train',['--data',str(data),'--output',str(out),'--steps',str(steps),'--q-burnin-steps',str(burnin),'--delta-weight',str(release.get('rl_config',{}).get('delta_weight',10.)),'--critic-target',str(release.get('critic_target','td')),'--resume-release',str(release_path)])
            end=release['global_step']+steps
            before=json.loads((out/f'evaluation_{release["global_step"]}.json').read_text());after=json.loads((out/f'evaluation_{end}.json').read_text())
            fixed_path=release.get('fixed_gate_baseline')
            baseline=json.loads(Path(fixed_path).read_text()) if fixed_path and Path(fixed_path).is_file() else before
            reasons=regression_gate(baseline,after)
            atomic(telemetry/'episode_q.json',after['episode_q'])
            atomic(telemetry/'progress.json',dict(phase='evaluation',completed=1,total=1,message='固定 autonomous gate 已完成'))
            # Finite critic predictions are necessary; this is not a success-rate estimate.
            import numpy as np
            for p in out.glob(f'eval_{end}_*.npz'):
                with np.load(p) as f:
                    if not np.isfinite(f['q']).all() or not np.isfinite(f['pred']).all():reasons.append('nonfinite:'+p.name)
            audit=dict(accepted=not reasons,reasons=reasons,updates=steps,new_transitions=transitions,new_uuids=uuids,parent_release=str(release_path),before=before['summary'],after=after['summary'],value_summary=after['value_summary'])
            atomic(out/'release_audit.json',audit)
            if reasons:
                atomic(telemetry/'progress.json',dict(phase='rejected',completed=1,total=1,message='candidate rejected: '+', '.join(reasons[:3])))
                atomic(status,dict(phase='rejected',pid=os.getpid(),**audit));return None
            # Revalidate current parent immediately before publish; never overwrite another release.
            current_path,current=selected()
            if current_path!=release_path:raise RuntimeError('parent release changed during batch')
            new=dict(release,name='rtc-online-'+tag,global_step=end,actor_updates=after['actor_version'],checkpoint=str(out/f'checkpoints/step_{end}.pkl'),norm_stats=str(out/'norm_stats_delta.json'),rl_config=json.loads((out/'config.json').read_text())['rl_config'],dataset=str(data),parent_release=str(release_path),consumed_uuids=sorted(set(release.get('consumed_uuids',[]))|set(uuids)),offline_audit=str(out/'release_audit.json'),fixed_gate_baseline=release.get('fixed_gate_baseline') or str(out/f'evaluation_{release["global_step"]}.json'),critic_target=str(json.loads((out/'config.json').read_text()).get('critic_target','td')),critic_burnin_updates=int(release.get('critic_burnin_updates',0))+burnin,critic_calibrated=(int(release.get('critic_burnin_updates',0))+burnin)>=1000,onsite_validated=False,online_improvement_validated=False)
            for key in ('checkpoint','norm_stats'):new[key+'_sha256']=digest(new[key])
            if new['norm_stats_sha256']!=release['norm_stats_sha256']:raise ValueError('online normalization changed')
            path=folder/'rtc-v5/releases'/('rtc-online-'+tag+'.json')
            if path.exists():raise FileExistsError(path)
            atomic(path,new);atomic(CURRENT,dict(release=str(path),release_sha256=digest(path)))
            atomic(telemetry/'progress.json',dict(phase='accepted',completed=1,total=1,message='candidate accepted at learner step '+str(end)))
            atomic(status,dict(phase='accepted',pid=os.getpid(),actor_version=after['actor_version'],learner_final_step=end,release=str(path),**audit));return new
        except Exception as e:
            atomic(telemetry/'progress.json',dict(phase='error',completed=1,total=1,message='update failed: '+str(e)))
            atomic(status,dict(phase='rejected',pid=os.getpid(),error=str(e),new_uuids=uuids,automatic_warmup=False));raise

def cycle(episodes=5):
    if episodes!=5:raise ValueError('validated batch size is five episodes')
    ledger=RUN/'learning/rtc-v5/online_cycle.json'
    previous=json.loads(ledger.read_text()) if ledger.exists() else {};attempted=set(previous.get('attempted_uuids',[]))
    while True:
        # Servo/HIL/finalize phases do no replay scan, checkpoint hashing or ledger writes.
        if session_phase() not in IDLE:
            time.sleep(2);continue
        path,release=selected();consumed=set(release.get('consumed_uuids',[]));available=available_uuids();batch=pending(available,consumed,attempted,episodes)
        if not batch or session_phase() not in IDLE:
            eligible=available-consumed-attempted
            quarantined=(available-consumed)&attempted
            atomic(ledger,dict(phase='waiting',pending_episodes=len(eligible),new_since_attempt=len(eligible),quarantined_episodes=len(quarantined),episodes_per_update=episodes,consumed_uuids=sorted(consumed),attempted_uuids=sorted(attempted)))
            telemetry=RUN/'learning/telemetry';telemetry.mkdir(exist_ok=True)
            atomic(telemetry/'progress.json',dict(phase='waiting',completed=min(len(eligible),episodes),total=episodes,message=str(len(eligible))+'/'+str(episodes)+' new verified episodes'))
            time.sleep(2);continue
        try:
            result=run_batch(path,release,batch);outcome='accepted' if result else 'rejected'
        except LearningDeferred:time.sleep(1);continue
        except Exception as e:print('RTC_ONLINE_REJECTED',str(e),flush=True);outcome='rejected'
        attempted.update(batch)
        atomic(ledger,dict(phase=outcome,new_uuids=batch,attempted_uuids=sorted(attempted),episodes_per_update=episodes))
        print('RTC_ONLINE',outcome,'episodes',len(batch),flush=True)
