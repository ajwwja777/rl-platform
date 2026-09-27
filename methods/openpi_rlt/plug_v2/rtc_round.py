"""Round-based offline updates for plug_v2, replacing the paused five-episode online cycle.

- Every validated episode enters the dataset; nothing is quarantined. A failed gate rejects the
  parameter update only, never the data.
- A round trains an AWBC candidate from the deployed release on all data and writes a candidate
  release. It never moves the deployed pointer: promotion is an explicit on-robot A/B decision.
- Training requires the Session to be stopped (rlt_stop.sh keeps the model loaded), so it never
  runs between episodes.

  status                       episodes since the last round, per-actor on-robot success
  critic [--targets ...]       critic-only runs (actor frozen) comparing critic targets on held-out episodes
  train [--critic-target ...]  one round: rebuild dataset, train AWBC candidate, gate, write candidate release
  compare A B                  on-robot comparison of two actor versions (Wilson CI, Fisher exact)
  switch RELEASE               point the deployment at a release (backup kept); restart the Session to load it
"""
import argparse,json,os,sys,time,math,glob,subprocess,fcntl,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2';LEARN=RUN/'learning';RELEASES=LEARN/'rtc-v5/releases'
ROUNDS=LEARN/'rounds';ONLINE=ROOT.parent.parent/'data/rlt/plug_v2/online'
PY='/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python'
IDLE_FOR_SWITCH=('disarmed','ready','stopped','offline','waiting_scene')
DEPLOY_KEYS=('deployment_matched_mse','deployment_accel_rms')

# ---------- statistics ----------
def wilson(k,n,z=1.96):
    if n<=0:return (float('nan'),float('nan'))
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0.,c-h),min(1.,c+h))

def fisher_two_sided(a,b,c,d):
    """2x2 table [[a,b],[c,d]] (success,failure per arm)."""
    n=a+b+c+d;r1=a+b;c1=a+c
    def p(k):return math.comb(r1,k)*math.comb(n-r1,c1-k)/math.comb(n,c1)
    p0=p(a);lo=max(0,c1-(n-r1));hi=min(r1,c1)
    return min(1.,sum(p(k) for k in range(lo,hi+1) if p(k)<=p0*(1+1e-9)))

def auc(pos,neg):
    pos=np.asarray(pos,float);neg=np.asarray(neg,float)
    if not len(pos) or not len(neg):return float('nan')
    diff=pos[:,None]-neg[None,:];return float(np.mean((diff>0)+.5*(diff==0)))

def bootstrap_auc(values,labels,iters=2000,seed=0):
    """Episode-level bootstrap: point AUC and 95% interval."""
    v=np.asarray(values,float);y=np.asarray(labels,bool);rng=np.random.default_rng(seed);point=auc(v[y],v[~y]);out=[]
    for _ in range(iters):
        i=rng.integers(0,len(v),len(v));yy=y[i]
        if yy.all() or not yy.any():continue
        out.append(auc(v[i][yy],v[i][~yy]))
    lo,hi=(np.quantile(out,[.025,.975]) if out else (float('nan'),)*2)
    return point,float(lo),float(hi)

# ---------- gates ----------
def deployment_reasons(base,after,tol=.05):
    """Hard safety gate: the fixed deployment-MSE/acceleration part of rtc_online_cycle.regression_gate.
    Value metrics are reported with intervals instead of gating: with ~25 held-out episodes they are noise-level."""
    reasons=[]
    for delay in ('0','6'):
        a=base['summary'][delay];b=after['summary'][delay]
        for key in DEPLOY_KEYS:
            if not math.isfinite(b[key]) or b[key]>a[key]*(1+tol)+1e-10:reasons.append(delay+':'+key)
        if b['deployment_matched_mse']>=b['ref_matched_conditioner_mse']:reasons.append(delay+':reference_regression')
    return reasons

def value_report(evaluation,group='autonomous'):
    rows=[r for r in evaluation.get('value_episodes',[]) if group in r.get('groups',())]
    y=[bool(r['success']) for r in rows];out={'n_success':int(sum(y)),'n_failure':int(len(y)-sum(y))}
    for field in ('q_data','q_data_early','q_actor'):
        keep=[(r[field],s) for r,s in zip(rows,y) if r.get(field) is not None and math.isfinite(float(r[field]))]
        if keep and any(s for _,s in keep) and not all(s for _,s in keep):
            p,lo,hi=bootstrap_auc([v for v,_ in keep],[s for _,s in keep]);out[field]={'auc':p,'ci95':[lo,hi]}
    q=[r['q_actor'] for r in rows if r.get('q_actor') is not None];d=[r['q_data'] for r in rows if r.get('q_data') is not None]
    if q and d:out['q_actor_minus_q_data']=float(np.mean(q)-np.mean(d))  # overestimation of the actor's own actions
    return out

def choose_checkpoint(evaluations,parent_actor_version,tol=.05):
    """Latest evaluated checkpoint that changed the actor and passes the deployment gate, else None."""
    ordered=sorted(evaluations,key=lambda e:e['step']);base=ordered[0];chosen=None;log=[]
    for e in ordered[1:]:
        if int(e['actor_version'])<=int(parent_actor_version):continue
        reasons=deployment_reasons(base,e,tol);log.append({'step':e['step'],'actor_version':int(e['actor_version']),'reasons':reasons})
        if not reasons:chosen=e
    return chosen,log

# ---------- data ----------
def blue_centroid(img):
    a=img.astype(np.int16);r,g,b=a[...,0],a[...,1],a[...,2];m=(b>150)&(b-r>40)&(g>110)&(g<230);ys,xs=np.nonzero(m)
    return (float(xs.mean()),float(ys.mean())) if len(xs)>=50 else (float('nan'),float('nan'))

def online_episodes(scene=False):
    out=[]
    for f in glob.glob(str(ONLINE/'episode_*.rlt.json')):
        d=json.loads(Path(f).read_text());row=dict(index=d['episode_index'],uuid=d['episode_uuid'],actor_version=d.get('actor_version'),
            outcome=d.get('outcome'),hil=int(d.get('hil_frames') or 0)>0,commands=d.get('policy_command_count'),snapshot=d.get('scene_snapshot'))
        if scene and row['snapshot'] and os.path.exists(row['snapshot']):
            with np.load(row['snapshot']) as z:row['wrist_socket']=blue_centroid(z['camera_right']);row['start']=float(z['timestamp'])
        out.append(row)
    return sorted(out,key=lambda r:r['index'])

def validated_uuids():
    from .storage import releases_for_phase
    result=set()
    for phase in ('warmup','online'):
        for p in releases_for_phase(phase):
            d=json.loads(p.read_text())
            if d.get('cohort')=='plug_v2' and d.get('status')=='validated' and d.get('metadata',{}).get('outcome') in ('success','failure'):result.add(d['source_uuid'])
    return result

def latest_round():
    p=ROUNDS/'latest.json';return json.loads(p.read_text()) if p.exists() else None

def cache_dataset():
    """Most recent completed dataset directory, used only as a hard-link cache for per-episode features."""
    best=None
    for d in list((RUN/'diagnostics').glob('rtc-online-*'))+list((RUN/'diagnostics').glob('round-*')):
        p=d/'progress.json'
        try:
            if json.loads(p.read_text()).get('phase')=='completed' and (best is None or p.stat().st_mtime>best[0]):best=(p.stat().st_mtime,d)
        except (OSError,ValueError):continue
    return best[1] if best else None

SWITCH_LOG=ROUNDS/'switch-log.jsonl'

def read_switch_log(path=None):
    path=Path(path or SWITCH_LOG);out=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:out.append(json.loads(line))
            except ValueError:continue
    return sorted(out,key=lambda e:e['time'])

def assign_releases(rows,log):
    """Release active when each episode started: the latest switch before its start; None before the first logged switch."""
    times=[e['time'] for e in log]
    for r in rows:
        t=r.get('start');i=int(np.searchsorted(times,t,side='right'))-1 if t is not None else -1
        r['release']=log[i]['release'] if i>=0 else None
    return rows

def arm_stats(rows,version,key='actor_version'):
    L=[r for r in rows if r.get(key)==version and r['outcome'] in ('success','failure','aborted')]
    auto=[r for r in L if not r['hil']];k=sum(1 for r in auto if r['outcome']=='success')
    ws=[r['wrist_socket'] for r in L if 'wrist_socket' in r and math.isfinite(r['wrist_socket'][0])]
    return dict(version=version,n=len(L),n_auto=len(auto),auto_success=k,rate=k/len(auto) if auto else float('nan'),ci=wilson(k,len(auto)),
                hil=sum(1 for r in L if r['hil']),wrist=tuple(np.mean(ws,axis=0)) if ws else None)

# ---------- commands ----------
def session_phase():
    from .training_flow import session_phase as phase
    return phase()

def require_stopped():
    from .training_flow import rtc_training_idle
    if not rtc_training_idle():raise SystemExit('Session must be stopped first: ./scripts/rlt_stop.sh (the model stays loaded)')

def env():
    e=os.environ.copy();e.update(PYTHONPATH=':'.join([str(ROOT/'envs/machine-a-py311-overlay'),str(ROOT)]),JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='0',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.10',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',XDG_CACHE_HOME=str(LEARN/'rtc-v5/cache'))
    return e

def run(module,args,log):
    with open(log,'a') as f:
        f.write('$ '+module+' '+' '.join(map(str,args))+'\n');f.flush()
        subprocess.run([PY,'-u','-m','methods.openpi_rlt.plug_v2.'+module]+[str(a) for a in args],cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)

def build_dataset(tag,log):
    data=RUN/'diagnostics'/('round-'+tag);(data/'episodes').mkdir(parents=True)
    cache=cache_dataset()
    if cache:
        for p in (cache/'episodes').glob('*.npz'):os.link(p,data/'episodes'/p.name)
    print('dataset: building',data,'(feature cache from '+(cache.name if cache else 'none')+')',flush=True)
    run('rtc_upstream_data',['--output',data],log);return data

def lease():
    f=(LEARN/'operation.lock').open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('another learning operation holds the lease')
    return f

def cmd_status(a):
    from .rtc_release import selected
    path,rel=selected();rows=online_episodes(scene=True);last=latest_round();valid=validated_uuids()
    used=set(last['uuids']) if last else set(rel.get('consumed_uuids',[]))
    new=[r for r in rows if r['uuid'] in valid and r['uuid'] not in used]
    print(f"deployed: {rel['name']} (actor_updates {rel['actor_updates']})   session: {session_phase()}")
    print(f"last round: {last['tag'] if last else 'none'}   validated episodes not yet trained on: {len(new)} (suggest a round at >= {a.round_size})")
    versions=[];[versions.append(r['actor_version']) for r in reversed(rows) if r['actor_version'] not in versions]
    print('\nactor   eps  auto  auto-success  95% CI        hil  wrist socket (x,y)   [9/20 good scene ~ (420,96)]')
    for v in versions[:a.last]:
        s=arm_stats(rows,v);w='(%.0f,%.0f)'%s['wrist'] if s['wrist'] else '-'
        print(f"{str(v):>6} {s['n']:4d} {s['n_auto']:5d}   {s['auto_success']:3d} ({s['rate']:4.0%})   [{s['ci'][0]:.0%},{s['ci'][1]:.0%}]  {s['hil']:4d}  {w}")

def train_run(tag,parent_path,parent,data,critic_target,steps,burnin,log,extra=()):
    out=LEARN/('round-'+tag);evals=','.join(str(x) for x in sorted({burnin,burnin+(steps-burnin)//2,steps-(steps-burnin)//4,steps}) if x>0)
    args=['--data',data,'--output',out,'--resume-release',parent_path,'--steps',steps,'--q-burnin-steps',burnin,'--critic-target',critic_target,
          '--delta-weight',parent['rl_config'].get('delta_weight',300.),'--actor-objective','awbc','--eval-steps',evals,*extra]
    print(f'training: {out.name} target={critic_target} steps={steps} burnin={burnin}',flush=True);run('awbc_train',args,log)
    evaluations=[json.loads(p.read_text()) for p in out.glob('evaluation_*.json')]
    return out,evaluations

def cmd_critic(a):
    """Critic-only comparison on the same dataset: actor frozen for the whole run (burnin == steps)."""
    from .rtc_release import selected
    require_stopped();lock=lease();path,parent=selected();tag=time.strftime('%Y%m%dT%H%M%S');ROUNDS.mkdir(exist_ok=True);log=ROUNDS/(tag+'-critic.log')
    data=build_dataset(tag,log);report={}
    for target in a.targets:
        out,ev=train_run(tag+'-critic-'+target,path,parent,data,target,a.steps,a.steps,log,extra=('--mc-gamma',a.mc_gamma))
        report[target]={str(e['step']):value_report(e) for e in sorted(ev,key=lambda e:e['step'])}
    (ROUNDS/(tag+'-critic.json')).write_text(json.dumps(report,indent=2))
    print('\ncritic target   step   held-out autonomous AUC (95% CI)                          n s/f   Q(actor)-Q(data)')
    for target,by in report.items():
        for step,r in by.items():
            f=lambda k:'%.2f [%.2f,%.2f]'%(r[k]['auc'],*r[k]['ci95']) if k in r else '   -   '
            print(f"{target:14s} {step:>6}  q_data {f('q_data')}  early {f('q_data_early')}  {r['n_success']}/{r['n_failure']}   {r.get('q_actor_minus_q_data',float('nan')):+.3f}")

def cmd_train(a):
    from .rtc_release import selected,digest
    require_stopped();lock=lease();path,parent=selected();tag=time.strftime('%Y%m%dT%H%M%S');ROUNDS.mkdir(exist_ok=True);log=ROUNDS/(tag+'.log')
    data=build_dataset(tag,log)
    out,evaluations=train_run(tag,path,parent,data,a.critic_target,a.steps,a.burnin,log,extra=('--awbc-ref-weight',a.ref_weight,'--mc-gamma',a.mc_gamma,'--advantage',a.advantage,'--awbc-beta',a.awbc_beta,*(['--value-effort'] if a.value_effort else []),*(['--value-online-only'] if a.value_online_only else [])))
    chosen,gate_log=choose_checkpoint(evaluations,parent['actor_updates'],a.tol)
    manifest=json.loads((data/'manifest.json').read_text());uuids=sorted({m['uuid'] for m in manifest if not m['expert']})
    run_config=json.loads((out/'config.json').read_text())
    summary=dict(tag=tag,parent=str(path),dataset=str(data),training=str(out),critic_target=a.critic_target,advantage=a.advantage,awbc_beta=a.awbc_beta,value_effort=a.value_effort,value_online_only=a.value_online_only,value_model=run_config.get('value_model'),uuids=uuids,gate=gate_log,candidate=None,
                 value={str(e['step']):value_report(e) for e in sorted(evaluations,key=lambda e:e['step'])})
    if chosen:
        step=int(chosen['step']);new=dict(parent)
        new.update(name='rtc-round-'+tag,global_step=step,actor_updates=int(chosen['actor_version']),checkpoint=str(out/f'checkpoints/step_{step}.pkl'),
            norm_stats=str(out/'norm_stats_delta.json'),rl_config=json.loads((out/'config.json').read_text())['rl_config'],dataset=str(data),consumed_uuids=uuids,
            parent_release=str(path),critic_target=a.critic_target,critic_burnin_updates=a.burnin,critic_calibrated=True,offline_audit=str(out/f'evaluation_{step}.json'),round_summary=str(ROUNDS/(tag+'.json')),
            onsite_validated=False,online_improvement_validated=False,actor_objective='awbc_candidate_v1',advantage=a.advantage,awbc_beta=a.awbc_beta,candidate_training=str(out),promotion='pending_onsite_ab',
            code_sha256={k:digest(ROOT/k) for k in parent['code_sha256']})
        for k in ('rollback_of','rollback_reason','candidate_note'):new.pop(k,None)
        for k in ('checkpoint','norm_stats'):new[k+'_sha256']=digest(new[k])
        if new['norm_stats_sha256']!=parent['norm_stats_sha256']:raise RuntimeError('normalization changed')
        target=RELEASES/(new['name']+'.json')
        if target.exists():raise FileExistsError(target)
        tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(new,indent=2));os.replace(tmp,target)
        probe=ROUNDS/(tag+'-pointer.json');probe.write_text(json.dumps(dict(release=str(target),release_sha256=digest(target))));selected(probe)  # full validation
        summary['candidate']=dict(release=str(target),step=step,actor_updates=new['actor_updates'])
    for p in (ROUNDS/(tag+'.json'),ROUNDS/'latest.json'):p.write_text(json.dumps(summary,indent=2))
    print('\ncheckpoint gate:');[print(f"  step {g['step']} actor {g['actor_version']}: {'PASS' if not g['reasons'] else ', '.join(g['reasons'])}") for g in gate_log]
    v=summary['value'].get(str(chosen['step'])) if chosen else None
    vm=summary['value_model']
    if vm:print('value model V(s): held-out autonomous AUC',vm['heldout_autonomous'],'| early stop epoch',vm['best_epoch'],'| advantage on successful policy rows',vm['advantage_policy_success'])
    if v:print('held-out autonomous critic AUC:',{k:'%.2f [%.2f,%.2f]'%(v[k]['auc'],*v[k]['ci95']) for k in ('q_data','q_data_early','q_actor') if k in v},'Q(actor)-Q(data) %+.3f'%v.get('q_actor_minus_q_data',float('nan')))
    if chosen:print(f"\ncandidate: {summary['candidate']['release']}\n  A/B: run the deployed actor and this candidate in alternating blocks of 5 episodes on the same scene, >=15 each,\n  switch with: rtc_round switch {Path(summary['candidate']['release']).name}  (then restart the Session)")
    else:print('\nno checkpoint passed the deployment gate; data is kept for the next round, the deployment is unchanged')

def cmd_compare(a):
    rows=online_episodes(scene=True)
    by_release=not (a.a.isdigit() and a.b.isdigit())
    if by_release:
        assign_releases(rows,read_switch_log())
        norm=lambda n:n if n.endswith('.json') else n+'.json'
        A=arm_stats(rows,norm(a.a),'release');B=arm_stats(rows,norm(a.b),'release')
    else:
        A=arm_stats(rows,int(a.a));B=arm_stats(rows,int(a.b))
    for s in (A,B):print(f"actor {s['version']}: autonomous {s['auto_success']}/{s['n_auto']} = {s['rate']:.0%}  95% CI [{s['ci'][0]:.0%},{s['ci'][1]:.0%}]  (+{s['hil']} HIL)  wrist socket {'(%.0f,%.0f)'%s['wrist'] if s['wrist'] else '-'}")
    p=fisher_two_sided(B['auto_success'],B['n_auto']-B['auto_success'],A['auto_success'],A['n_auto']-A['auto_success']);print(f'Fisher exact two-sided p = {p:.3f}')
    if A['wrist'] and B['wrist'] and (abs(A['wrist'][0]-B['wrist'][0])>8 or abs(A['wrist'][1]-B['wrist'][1])>8):print('warning: the two arms ran on different socket placements; the comparison is confounded')
    if min(A['n_auto'],B['n_auto'])<a.min_n:print(f'verdict: keep collecting (need >= {a.min_n} autonomous episodes per arm)')
    elif B['rate']>A['rate'] and p<.05:print(f'verdict: promote {a.b} (significantly better)')
    elif B['rate']>=A['rate']:print(f'verdict: {a.b} is not worse; promotion optional, more episodes would make it decisive')
    else:print(f'verdict: keep {a.a}' + (' (candidate significantly worse)' if p<.05 else ''))

def cmd_switch(a):
    from .rtc_release import selected,digest,CURRENT
    phase=session_phase()
    if phase not in IDLE_FOR_SWITCH:raise SystemExit('Session is '+phase+'; switch between episodes')
    target=(RELEASES/a.release).resolve(strict=True);target.relative_to(RELEASES.resolve())
    probe=ROUNDS/('switch-'+time.strftime('%Y%m%dT%H%M%S')+'.json');ROUNDS.mkdir(exist_ok=True)
    probe.write_text(json.dumps(dict(release=str(target),release_sha256=digest(target))));_,m=selected(probe)
    shutil.copyfile(CURRENT,CURRENT.with_name('current.json.bak-'+time.strftime('%Y%m%dT%H%M%S')))
    tmp=CURRENT.with_suffix('.json.tmp');shutil.copyfile(probe,tmp);os.replace(tmp,CURRENT)
    with SWITCH_LOG.open('a') as f:f.write(json.dumps(dict(time=time.time(),release=target.name,actor_updates=m['actor_updates']))+'\n')
    print(f"deployed pointer -> {m['name']} (actor_updates {m['actor_updates']}); restart the Session to load it:\n  ./scripts/rlt_up.sh --frozen-actor --restart")

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);s=p.add_subparsers(dest='cmd',required=True)
    x=s.add_parser('status');x.add_argument('--round-size',type=int,default=25);x.add_argument('--last',type=int,default=8)
    x=s.add_parser('critic');x.add_argument('--targets',nargs='+',default=['mc_success','mc_discounted'],choices=('td','mc_success','mc_discounted'));x.add_argument('--steps',type=int,default=3000);x.add_argument('--mc-gamma',type=float,default=.995)
    x=s.add_parser('train');x.add_argument('--critic-target',default='mc_success',choices=('td','mc_success','mc_discounted'));x.add_argument('--steps',type=int,default=4000);x.add_argument('--burnin',type=int,default=2000)
    x.add_argument('--ref-weight',type=float,default=.5);x.add_argument('--mc-gamma',type=float,default=.995);x.add_argument('--tol',type=float,default=.05)
    x.add_argument('--advantage',choices=('q','v'),default='q',help='q: critic Q(s,a_data)-Q(s,a_ref); v: state-value change V(s\')-V(s)')
    x.add_argument('--value-effort',action='store_true',help='V also sees causal gripper effort (fit on episodes with recorded effort)');x.add_argument('--value-online-only',action='store_true',help='V fit without demonstrations (matched ablation)')
    x.add_argument('--awbc-beta',type=float,default=.1,help='advantage temperature; very large (1e6) = plain imitation of successful and corrected data')
    x=s.add_parser('compare',help='A B: actor versions, or release names (episodes assigned by the switch log)');x.add_argument('a');x.add_argument('b');x.add_argument('--min-n',type=int,default=15)
    x=s.add_parser('switch');x.add_argument('release')
    a=p.parse_args();{'status':cmd_status,'critic':cmd_critic,'train':cmd_train,'compare':cmd_compare,'switch':cmd_switch}[a.cmd](a)
if __name__=='__main__':main()
