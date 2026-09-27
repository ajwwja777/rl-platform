"""AWBC candidate trainer for plug_v2 RTC; offline candidate only, never automatic release.
Reuses the pinned rtc_upstream_train pipeline (dataset, replay, evaluation, gate payloads) and swaps
only the actor objective and the per-transition success label. Pinned modules are not modified.
"""
import os,json,time,argparse,dataclasses,hashlib,types,shutil,functools
from pathlib import Path
import torch,numpy as np
import jax,yaml
from .upstream_baseline import ROOT,RUN,load
from . import rtc_upstream_train as base
from . import rtc_upstream_core
from .rtc_upstream_train import PreciseSource,evaluate,resolve_online_weights,resolved_delta_weight,burnin_stop_step
from .awbc_core import train_step_awbc,AWBC_METRIC_KEYS
from rlt_online_rl import trainer as upstream
from rlt_online_rl.config import system_config_from_mapping,LearnerServiceConfig
from rlt_online_rl.replay import RLTTransition
from .balanced_replay import BalancedReplayBuffer
from .training_flow import rtc_training_idle
from .critic_targets import training_target
from .online_value_gate import aggregate_metric_window

HUMAN_SOURCES=(2,3)

def transition_success(frame,source_chunk,*,expert,episode_success):
    """Per-transition success label.
    In a non-expert episode that needed a human, policy rows before the first human frame are the
    segment the operator had to rescue: they are labelled failure for the critic target and for imitation.
    Human rows and policy rows after the handover keep the episode outcome."""
    frame=np.asarray(frame);human=np.isin(np.asarray(source_chunk)[:,0],HUMAN_SOURCES)
    out=np.full(len(frame),bool(episode_success))
    if expert or not human.any() or not episode_success:return out
    out[(~human)&(frame<frame[human].min())]=False
    return out

def remaining_frames(frame,duration):
    """Frames from each decision to the end of the recorded episode (terminal row ends at frame+duration)."""
    frame=np.asarray(frame,np.float64);end=float(np.max(frame+np.asarray(duration,np.float64)))
    return np.maximum(end-frame,0.)

def discounted_success(labels,frame,duration,gamma):
    """Monte-Carlo return of a terminal 0/1 reward: label*gamma**frames_to_end. Failure rows stay 0,
    success rows rise toward 1 as the episode approaches the insertion, which gives the critic a progress signal."""
    if not 0<gamma<=1:raise ValueError('invalid mc gamma')
    return np.asarray(labels,np.float32)*np.power(gamma,remaining_frames(frame,duration)).astype(np.float32)

def _state_features(d,i,prefix=''):
    return np.concatenate([np.asarray(d[prefix+'z_rl'][i],np.float32).reshape(-1),np.asarray(d[prefix+'proprio'][i],np.float32).reshape(-1)])

EFFORT_FEATURES=('grip_effort_now','grip_effort_max_2s')
EFFORT_STARTUP_SAMPLES=30  # ~1 s: recordings from 9/21 on start with a gripper transient (~2.0) before any command

def effort_features(meta,d,window=60):
    """Causal right-gripper effort features at each row's decision frame and next frame.
    The arm joints publish zero effort on this robot; only the gripper channel (index 13) is real.
    The start-of-recording transient is flattened: it marks which session an episode came from, not
    how the insertion went, and would let V identify sessions instead of states. Running maxima are
    not used for the same reason. Aligned as rtc_upstream_data aligns images: trace frame -> nearest sample."""
    from .upstream_replay import nearest
    with np.load(meta['trace'],allow_pickle=True) as t:frames=json.loads(str(t['frames_json']))
    with np.load(Path(meta['root'])/'source_facts.npz') as z:
        stamps=np.maximum(z['rollout/topic_timestamp/front_left'],z['rollout/topic_timestamp/front_right']);eff=np.abs(z['observations/effort'][:,13]).astype(np.float32)
    start=min(EFFORT_STARTUP_SAMPLES,len(eff)-1);eff[:start]=eff[start]
    vi=nearest(stamps,np.array([f['ros_timestamp'] for f in frames]))
    def at(frame):
        k=np.asarray(vi[np.asarray(frame,int)]);recent=np.array([eff[max(0,j-window):j+1].max() for j in k],np.float32)
        return np.stack([eff[k],recent],1)
    return at(d['frame']),at(d['next_frame'])

def _episode_auc(scores,labels):
    s=np.asarray(scores,float);y=np.asarray(labels,bool)
    if y.all() or not y.any():return None
    diff=s[y][:,None]-s[~y][None,:];return float(np.mean((diff>0)+.5*(diff==0)))

def attach_value_advantages(rows,value_rows,ordered,episodes,*,seed=0,effort=False,online_only=False,effort_fn=None):
    """Fit V(s) on the training rows, store A=V(s')-V(s) (terminal: label-V(s)) as rows[k]['awbc_adv'].
    effort: append causal gripper-effort features (needs recorded source facts, so demonstrations are
    excluded from fitting; their rows are human rows whose AWBC weight ignores the advantage and get A=0).
    online_only: the same exclusion without effort, as the matched ablation.
    Returns a report; the validation split is only scored, never used for fitting or early stopping."""
    from .state_value import fit_value,predict,advantages
    effort_fn=effort_fn or effort_features;cache={}
    def feats(m,d,idx,nxt=False):
        base=np.stack([_state_features(d,i,'next_' if nxt else '') for i in idx])
        if not effort:return base
        if m['uuid'] not in cache:cache[m['uuid']]=effort_fn(m,d)
        return np.hstack([base,cache[m['uuid']][1 if nxt else 0][np.asarray(idx,int)]])
    use=np.array([not ((effort or online_only) and ordered[e][0]['expert']) for _,e,_,_,_ in value_rows])
    if not use.any():raise ValueError('no rows eligible for the value model')
    S=[];N=[]
    for k,(i,e,_,_,_) in enumerate(value_rows):
        if use[k]:m,d=ordered[e];S.append(feats(m,d,[i])[0]);N.append(feats(m,d,[i],True)[0])
    S=np.stack(S);N=np.stack(N)
    labels=np.array([l for *_,l,_ in value_rows],np.float32);done=np.array([dn for _,_,dn,_,_ in value_rows]);groups=np.array([g for *_,g in value_rows])
    model,report=fit_value(S,labels[use],groups[use],seed=seed)
    adv=np.zeros(len(value_rows),np.float32);adv[use]=advantages(predict(model,S),predict(model,N),done[use],labels[use])
    if len(adv)!=len(rows):raise RuntimeError('value rows out of sync with replay rows')
    for record,a in zip(rows,adv):record['awbc_adv']=np.float32(a)
    human=np.array([bool(ordered[e][1]['source_chunk'][i,0]==2) for i,e,_,_,_ in value_rows]);policy_success=(~human)&(labels>.5)&use
    report['advantage_policy_success']={k:float(f(adv[policy_success])) for k,f in (('mean',np.mean),('std',np.std),('p10',lambda a:np.quantile(a,.1)),('p90',lambda a:np.quantile(a,.9)))} if policy_success.any() else None
    report.update(inputs='z_rl+proprio'+('+gripper_effort' if effort else ''),effort_features=list(EFFORT_FEATURES) if effort else None,
                  fit_excludes_demonstrations=bool(effort or online_only),fit_rows_total=int(use.sum()))
    held=[];early=[];y=[]
    for m,d in episodes:
        if m['split']!='val' or m['expert'] or m['phase']!='online' or np.any(d['source_chunk'][:,0]==2):continue
        order=np.argsort(d['frame']);v=predict(model,feats(m,d,order))
        held.append(float(v.mean()));early.append(float(v[:3].mean()));y.append(bool(m['success']))
    report['heldout_autonomous']={'episodes':len(y),'success':int(sum(y)),'auc_mean':_episode_auc(held,y),'auc_early3':_episode_auc(early,y)}
    return report

def make_learner(step_fn):
    fn=upstream.LearnerService.train_once
    bound=types.FunctionType(fn.__code__,dict(fn.__globals__,train_step=step_fn,_resolve_actor_loss_weights=base._runtime_actor_loss_weights),fn.__name__,fn.__defaults__,fn.__closure__)
    bound.__kwdefaults__=fn.__kwdefaults__
    class Learner(base.Learner):train_once=bound
    return Learner

def aggregate(window):
    metric=aggregate_metric_window(window)
    actor=[row for row in window if int(row.get('did_actor_update',0))==1]
    for key in AWBC_METRIC_KEYS:
        if not key.startswith('awbc_'):continue
        values=[float(row[key]) for row in actor if key in row and np.isfinite(float(row[key]))]
        if values:metric[key]=float(np.mean(values))
        else:metric.pop(key,None)
    return metric

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--delta-weight',type=float,default=10.);p.add_argument('--q-burnin-steps',type=int,default=1000);p.add_argument('--critic-target',choices=('td','mc_success','mc_discounted'),default='td');p.add_argument('--mc-gamma',type=float,default=.995);p.add_argument('--online-bc-weight',type=float);p.add_argument('--online-q-weight',type=float);p.add_argument('--actor-lr',type=float);p.add_argument('--resume-release',type=Path)
    p.add_argument('--actor-objective',choices=('awbc','q'),default='awbc');p.add_argument('--awbc-beta',type=float,default=.1);p.add_argument('--awbc-max-weight',type=float,default=5.);p.add_argument('--awbc-ref-weight',type=float,default=.1);p.add_argument('--no-hil-prefix-relabel',action='store_true');p.add_argument('--eval-steps',type=str,default='500,1000,2000,5000')
    p.add_argument('--advantage',choices=('q','v'),default='q',help='q: Q(s,a_data)-Q(s,a_ref) from the critic; v: V(s\')-V(s) from a state-only value model');p.add_argument('--value-seed',type=int,default=0)
    p.add_argument('--value-effort',action='store_true',help='add causal gripper-effort features to V (fits on episodes with recorded effort only)');p.add_argument('--value-online-only',action='store_true',help='fit V without demonstrations (matched ablation for --value-effort)')
    args=p.parse_args()
    if args.advantage=='v' and args.actor_objective!='awbc':raise ValueError('value advantages need --actor-objective awbc')
    if (args.value_effort or args.value_online_only) and args.advantage!='v':raise ValueError('--value-effort/--value-online-only need --advantage v')
    eval_steps={int(x) for x in args.eval_steps.split(',') if x.strip()}|{args.steps}
    if args.q_burnin_steps<0 or args.q_burnin_steps>args.steps:raise ValueError('invalid q burnin')
    if args.actor_objective=='awbc' and (not np.isfinite(args.awbc_beta) or args.awbc_beta<=0 or args.awbc_max_weight<1 or args.awbc_ref_weight<0):raise ValueError('invalid awbc parameters')
    base._Q_BURNIN_STEPS=args.q_burnin_steps
    out=args.output.resolve();out.relative_to(RUN.resolve());out.mkdir(exist_ok=False)
    if json.loads((args.data/'progress.json').read_text())['phase']!='completed':raise ValueError('dataset incomplete')
    if not rtc_training_idle():raise ValueError('end Session before offline training')
    torch.set_num_threads(2);episodes=load(args.data);train=[(m,d) for m,d in episodes if m['split']=='train']
    deltas=[]
    for m,d in train:
        delta=d['action_chunk'].copy();delta[:,:,:6]-=d['proprio'][:,None,:6];deltas.append(delta.reshape(-1,7))
    x=np.concatenate(deltas);q01,q99=np.quantile(x,[.01,.99],axis=0).astype(np.float32);stats=out/'norm_stats_delta.json';stats.write_text(json.dumps({'norm_stats':{'actions':{'q01':q01.tolist(),'q99':q99.tolist()}}}))
    cfg=yaml.safe_load((ROOT/'code/openpi-rlt/rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml').read_text())['experiment']['rl'];cfg.update(proprio_dim=99,action_norm_stats_path=str(stats),delta_weight=args.delta_weight,warmup_post_collect_updates=args.steps,freeze_after_warmup=True)
    if args.resume_release:
        release=json.loads(args.resume_release.read_text())
        if hashlib.sha256(Path(release['checkpoint']).read_bytes()).hexdigest()!=release['checkpoint_sha256']:raise ValueError('resume checkpoint hash changed')
        if hashlib.sha256(Path(release['norm_stats']).read_bytes()).hexdigest()!=release['norm_stats_sha256']:raise ValueError('resume stats hash changed')
        shutil.copyfile(release['norm_stats'],stats)
        cfg=dict(release['rl_config']);cfg.update(action_norm_stats_path=str(stats),warmup_post_collect_updates=0,freeze_after_warmup=False,delta_weight=resolved_delta_weight(release['rl_config'],args.delta_weight))
        (out/'checkpoints').mkdir();shutil.copyfile(release['checkpoint'],out/'checkpoints/latest.pkl')
    # Under AWBC the critic is a scorer, not an objective: no Q term unless explicitly requested.
    q_weight=args.online_q_weight if args.online_q_weight is not None else (0. if args.actor_objective=='awbc' else None)
    online_bc,online_q=resolve_online_weights(cfg,args.online_bc_weight,q_weight);cfg.update(online_bc_weight=online_bc,online_q_weight=online_q)
    if args.actor_lr is not None:
        if not np.isfinite(args.actor_lr) or args.actor_lr<=0:raise ValueError('invalid actor lr')
        cfg['actor_lr']=float(args.actor_lr)
    rl=system_config_from_mapping({'rl':cfg}).rl;buffer=BalancedReplayBuffer(capacity=sum(len(d['z_rl']) for m,d in train),seed=42,sample_strategy='stratified',human_intervention_ratio=.2);rows=[]
    ordered=sorted(train,key=lambda x:({'demonstrations':0,'warmup':1,'online':2}[x[0]['phase']],x[0]['uuid']))
    relabel={'episodes':0,'transitions':0};value_rows=[]  # (state, next state, original done, label, episode) per replay row
    for eid,(m,d) in enumerate(ordered):
        labels=transition_success(d['frame'],d['source_chunk'],expert=bool(m['expert']),episode_success=bool(m['success'])) if not args.no_hil_prefix_relabel else np.full(len(d['z_rl']),bool(m['success']))
        flipped=int(np.sum(labels!=bool(m['success'])));relabel['transitions']+=flipped;relabel['episodes']+=int(flipped>0)
        returns=discounted_success(labels,d['frame'],d['duration'],args.mc_gamma) if args.critic_target=='mc_discounted' else None
        for i in range(len(d['z_rl'])):
            if not d['td_valid'][i]:continue  # No fabricated terminal/Q target from a censored edge.
            values={k:d[k][i] for k in ('z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk')};human=bool(d['source_chunk'][i,0]==2)
            record=training_target(dict(values,duration=d['duration'][i],td_valid=d['td_valid'][i]),success=bool(labels[i]),mode='mc_success' if returns is not None else args.critic_target)
            if returns is not None:record['rewards'][0]=returns[i]
            record.update(original_done=bool(d['done'][i]),episode_position=float(d['position'][i]));idx=len(rows);rows.append(record);transition={k:record[k] for k in values}
            if args.advantage=='v':value_rows.append((i,eid,bool(d['done'][i]),float(labels[i]),m['uuid']))
            buffer.add_with_metadata(RLTTransition(**transition,source=int(values['source_chunk'][0]),collection_phase='online' if m['phase']=='online' else 'warmup',success=int(labels[i]),intervention_flag=human and not m['expert'],episode_id=eid,step_id=idx),original_done=record['original_done'],episode_position=record['episode_position'])
    value_report=None
    if args.advantage=='v':
        value_report=attach_value_advantages(rows,value_rows,ordered,episodes,seed=args.value_seed,effort=args.value_effort,online_only=args.value_online_only)
        print('AWBC_VALUE',json.dumps(value_report),flush=True)
    awbc=dict(beta=args.awbc_beta,max_weight=args.awbc_max_weight,ref_weight=args.awbc_ref_weight) if args.actor_objective=='awbc' else None
    step_fn=functools.partial(train_step_awbc,**awbc) if awbc else rtc_upstream_core.train_step
    config={'rl_config':dataclasses.asdict(rl),'learning_algorithm':'awbc_candidate_v1' if awbc else 'pinned_upstream_rlt_rtc_smdp_v1','actor_objective':args.actor_objective,'awbc':awbc,'hil_prefix_relabel':not args.no_hil_prefix_relabel,'relabelled':relabel,'mc_gamma':args.mc_gamma if args.critic_target=='mc_discounted' else None,'advantage':args.advantage,'value_model':value_report,
            'upstream_commit':'c1e40ac360185778c98cf20da2820e22d2d415e7','dataset_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),'q_burnin_steps':args.q_burnin_steps,'q_burnin_stop_step':base._Q_BURNIN_STEPS,'critic_target':args.critic_target,
            'adaptation':'rtc_upstream_train pipeline; actor objective='+args.actor_objective+('; advantage-weighted BC, no Q gradient into actor' if awbc else '')+('' if args.no_hil_prefix_relabel else '; HIL prefix relabelled as failure'),'production_publish':False}
    (out/'config.json').write_text(json.dumps(config,indent=2));service=LearnerServiceConfig(sample_batch_size=128,checkpoint_dir=str(out/'checkpoints'),actor_snapshot_path=str(out/'actor_snapshot.pkl'),push_actor_interval_steps=500,checkpoint_interval_steps=1000)
    learner=make_learner(step_fn)(rl,service,PreciseSource(buffer,len(ordered)-1,rows),rng=jax.random.PRNGKey(42));base._Q_BURNIN_STEPS=burnin_stop_step(int(learner.state.global_step),args.q_burnin_steps);start=time.monotonic()
    config['q_burnin_stop_step']=base._Q_BURNIN_STEPS
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print('AWBC_CONFIG',json.dumps({k:config[k] for k in ('actor_objective','awbc','advantage','relabelled','critic_target','q_burnin_stop_step')}),flush=True)
    if args.resume_release:
        learner.explicit_stop_step=int(learner.state.global_step)+args.steps
        evaluate(learner,episodes,out,int(learner.state.global_step))
    metric_window=[]
    for step in range(1,args.steps+1):
        if step%100==1 and not rtc_training_idle():raise RuntimeError('Session changed')
        metric=learner.train_once()
        if metric is None:raise RuntimeError('no training budget')
        metric_window.append(metric)
        if step%100==0:
            metric=aggregate(metric_window);metric_window=[]
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            progress={'phase':'training','pid':os.getpid(),'step':step,'steps':args.steps,'seconds':time.monotonic()-start,'production_publish':False};(out/'operation.json').write_text(json.dumps(progress))
            print('AWBC_TRAIN',step,round(progress['seconds'],2),flush=True)
        if step in eval_steps:learner.save_checkpoint();evaluate(learner,episodes,out,int(learner.state.global_step))
    (out/'operation.json').write_text(json.dumps({'phase':'completed','steps':args.steps,'global_step':int(learner.state.global_step),'seconds':time.monotonic()-start,'production_publish':False}));print('AWBC_TRAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
