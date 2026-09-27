"""Pinned-upstream RTC learner runner; candidate only, never automatic release."""
import os,sys,json,time,argparse,pickle,dataclasses,hashlib,types,shutil
from pathlib import Path
import torch,numpy as np
from .upstream_baseline import ROOT,RUN,OfflineLearner,Source,load
from .rtc_upstream_core import train_step,original_context
import jax,jax.numpy as jnp,yaml
from rlt_online_rl import trainer as upstream
from rlt_online_rl.config import system_config_from_mapping,LearnerServiceConfig
from rlt_online_rl.replay import RLTTransition
from .balanced_replay import BalancedReplayBuffer
from .training_flow import session_phase,rtc_training_idle
from .conditioning_v2 import conditioned
from .chunk_projection import project_correction
from .critic_targets import training_target
from .online_value_gate import aggregate_metric_window, build_value_episode, episode_q_payload, summarize_value_episodes, validate_episode_splits

_Q_BURNIN_STEPS=0
def burnin_stop_step(start_global_step,requested_steps):
    return int(start_global_step)+int(requested_steps)

def resolve_online_weights(config,bc_weight,q_weight):
    bc=float(config['online_bc_weight'] if bc_weight is None else bc_weight)
    q=float(config['online_q_weight'] if q_weight is None else q_weight)
    if not np.isfinite(bc) or bc<=0 or not np.isfinite(q) or q<0:raise ValueError('invalid online loss weights')
    return bc,q

def resolved_delta_weight(release_config,requested):
    value=float(requested)
    if not np.isfinite(value) or value<=0:raise ValueError('invalid delta weight')
    return value
def staged_actor_loss_weights(rl_config,progress,q_burnin_steps):
    warmup=int(progress['warmup_required_updates']);step=int(progress['global_step'])
    if step<int(q_burnin_steps):return 0.,0.
    if warmup>0 and step<warmup:return float(rl_config.warmup_bc_weight),float(rl_config.warmup_q_weight)
    return float(rl_config.online_bc_weight),float(rl_config.online_q_weight)

def _runtime_actor_loss_weights(rl_config,progress):
    return staged_actor_loss_weights(rl_config,progress,_Q_BURNIN_STEPS)

_fn=upstream.LearnerService.train_once
_bound=types.FunctionType(_fn.__code__,dict(_fn.__globals__,train_step=train_step,_resolve_actor_loss_weights=_runtime_actor_loss_weights),_fn.__name__,_fn.__defaults__,_fn.__closure__)
_bound.__kwdefaults__=_fn.__kwdefaults__
class Learner(OfflineLearner):
    train_once=_bound
    def _desired_total_updates(self,replay_size,adds_total):
        if hasattr(self,'explicit_stop_step'):return self.explicit_stop_step
        return super()._desired_total_updates(replay_size,adds_total)
class PreciseSource(Source):
    def __init__(self,buffer,max_episode,rows):super().__init__(buffer,max_episode);self.rows=rows
    def sample_batch(self,n):
        b=super().sample_batch(n);at=b['step_id'].astype(int)
        # Keep original sampler selections; retain float32 factual commands
        # rather than quantizing absolute radian commands to fp16 in replay.
        for k in self.rows[0]:b[k]=np.asarray([self.rows[i][k] for i in at])
        return b

def evaluate(learner,episodes,out,step):
    a,q,s,adapter=learner._actor,learner._critic,learner.state,learner._action_adapter
    @jax.jit
    def fn(z,p,r,action):
        pred=a.actor_mean(s.actor_params,z,p,r)
        return pred,q.q_values(s.critic_params,z,p,action)[0],q.q_values(s.critic_params,z,p,r)[0],q.q_values(s.critic_params,z,p,pred)[0]
    rows=[];value_episodes=[]
    validate_episode_splits(episodes)
    for m,d in episodes:
        if m['split']!='val':continue
        pred=[];qs=[]
        for start in range(0,len(d['z_rl']),128):
            at=np.minimum(np.arange(start,start+128),len(d['z_rl'])-1);n=min(128,len(d['z_rl'])-start)
            b=adapter.prepare_training_batch({k:v[at] for k,v in d.items()})
            v=fn(*[jnp.asarray(b[k],jnp.float32) for k in ('z_rl','proprio','ref_chunk','action_chunk')]);pred.append(adapter.denormalize_to_abs_chunk(np.asarray(v[0])[:n],d['proprio'][at][:n]));qs.append(np.stack([np.asarray(x)[:n] for x in v[1:]],axis=-1))
        pred=np.concatenate(pred);qv=np.concatenate(qs);c=original_context(d['proprio']);state=c[:,:14]
        value_episode=build_value_episode(m,qv,d['source_chunk'],d['done'])
        if value_episode is not None:value_episodes.append(value_episode)
        ref=np.tile(state[:,None,:],(1,10,1));ref[:,:,7:14]=d['ref_chunk'];full=ref.copy();full[:,:,7:14]=pred;target=d['action_chunk'][:,:,:6]
        targetfull=ref.copy();targetfull[:,:,7:14]=d['action_chunk']
        with torch.no_grad():
            cmd=conditioned(torch.tensor(full),torch.tensor(c)).numpy()[:,:,7:13]
            refcmd=conditioned(torch.tensor(ref),torch.tensor(c)).numpy()[:,:,7:13]
            targetcmd=conditioned(torch.tensor(targetfull),torch.tensor(c)).numpy()[:,:,7:13]
            deployed=full.copy();deployed[:,:,7:14]=project_correction(pred,d['ref_chunk'],1)
            deploycmd=conditioned(torch.tensor(deployed),torch.tensor(c)).numpy()[:,:,7:13]
        for delay in (0,6):
            h=(d['source_chunk'][:,0]==2)&(d['delay']==delay)
            if not h.any():continue
            delta=np.diff(cmd[h],axis=1);refdelta=np.diff(refcmd[h],axis=1)
            rows.append(dict(uuid=m['uuid'],expert=m['expert'],success=m['success'],delay=delay,windows=int(h.sum()),
                deployment_matched_mse=float(np.mean((deploycmd[h]-targetcmd[h])**2)),deployment_accel_rms=float(np.sqrt(np.mean(np.diff(deploycmd[h],n=2,axis=1)**2))),
                raw_human_mse=float(np.mean((pred[h,:,:6]-target[h])**2)),ref_raw_human_mse=float(np.mean((d['ref_chunk'][h,:,:6]-target[h])**2)),
                matched_conditioner_mse=float(np.mean((cmd[h]-targetcmd[h])**2)),ref_matched_conditioner_mse=float(np.mean((refcmd[h]-targetcmd[h])**2)),
                executed_human_mse=float(np.mean((cmd[h]-target[h])**2)),ref_executed_human_mse=float(np.mean((refcmd[h]-target[h])**2)),
                command_accel_rms=float(np.sqrt(np.mean(np.diff(delta,axis=1)**2))),ref_command_accel_rms=float(np.sqrt(np.mean(np.diff(refdelta,axis=1)**2))),
                raw_step_p95=float(np.quantile(abs(np.diff(pred[h,:,:6],axis=1)),.95)),ref_raw_step_p95=float(np.quantile(abs(np.diff(d['ref_chunk'][h,:,:6],axis=1)),.95))))
        np.savez_compressed(out/f'eval_{step}_{m["uuid"]}_{m["index"]}.npz',pred=pred,q=qv,delay=d['delay'],frame=d['frame'],source_chunk=d['source_chunk'],done=d['done'],td_valid=d['td_valid'])
    summary={}
    for delay in (0,6):
        chosen=[r for r in rows if r['delay']==delay]
        summary[str(delay)]={k:float(np.mean([r[k] for r in chosen])) for k in ('deployment_matched_mse','deployment_accel_rms','raw_human_mse','ref_raw_human_mse','matched_conditioner_mse','ref_matched_conditioner_mse','executed_human_mse','ref_executed_human_mse','command_accel_rms','ref_command_accel_rms','raw_step_p95','ref_raw_step_p95')}
    value_summary=summarize_value_episodes(value_episodes)
    q_payload=episode_q_payload(value_episodes)
    report={'step':step,'actor_version':int(s.actor_version),'summary':summary,'value_summary':value_summary,'value_episodes':value_episodes,
            'rows':rows,'episode_q':q_payload,'production_publish':False,'heldout':'episode UUID validation split; autonomous/HIL reported separately'}
    (out/f'evaluation_{step}.json').write_text(json.dumps(report,indent=2));(out/f'episode_q_{step}.json').write_text(json.dumps(q_payload,indent=2));print('RTC_EVAL',step,json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--delta-weight',type=float,default=10.);p.add_argument('--q-burnin-steps',type=int,default=1000);p.add_argument('--critic-target',choices=('td','mc_success'),default='td');p.add_argument('--online-bc-weight',type=float);p.add_argument('--online-q-weight',type=float);p.add_argument('--actor-lr',type=float);p.add_argument('--resume-release',type=Path);args=p.parse_args()
    global _Q_BURNIN_STEPS
    _Q_BURNIN_STEPS=args.q_burnin_steps
    if _Q_BURNIN_STEPS<0 or _Q_BURNIN_STEPS>args.steps:raise ValueError('invalid q burnin')
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
    online_bc,online_q=resolve_online_weights(cfg,args.online_bc_weight,args.online_q_weight);cfg.update(online_bc_weight=online_bc,online_q_weight=online_q)
    if args.actor_lr is not None:
        if not np.isfinite(args.actor_lr) or args.actor_lr<=0:raise ValueError('invalid actor lr')
        cfg['actor_lr']=float(args.actor_lr)
    rl=system_config_from_mapping({'rl':cfg}).rl;buffer=BalancedReplayBuffer(capacity=sum(len(d['z_rl']) for m,d in train),seed=42,sample_strategy='stratified',human_intervention_ratio=.2);rows=[]
    ordered=sorted(train,key=lambda x:({'demonstrations':0,'warmup':1,'online':2}[x[0]['phase']],x[0]['uuid']))
    for eid,(m,d) in enumerate(ordered):
        for i in range(len(d['z_rl'])):
            if not d['td_valid'][i]:continue  # No fabricated terminal/Q target from a censored edge.
            values={k:d[k][i] for k in ('z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk')};human=bool(d['source_chunk'][i,0]==2)
            record=training_target(dict(values,duration=d['duration'][i],td_valid=d['td_valid'][i]),success=m['success'],mode=args.critic_target)
            record.update(original_done=bool(d['done'][i]),episode_position=float(d['position'][i]));idx=len(rows);rows.append(record);transition={k:record[k] for k in values}
            buffer.add_with_metadata(RLTTransition(**transition,source=int(values['source_chunk'][0]),collection_phase='online' if m['phase']=='online' else 'warmup',success=int(m['success']),intervention_flag=human and not m['expert'],episode_id=eid,step_id=idx),original_done=record['original_done'],episode_position=record['episode_position'])
    config={'rl_config':dataclasses.asdict(rl),'upstream_commit':'c1e40ac360185778c98cf20da2820e22d2d415e7','dataset_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),'q_burnin_steps':args.q_burnin_steps,'q_burnin_stop_step':_Q_BURNIN_STEPS,'critic_target':args.critic_target,'adaptation':'right-first99 RTC state; actual duration TD; censored TD mask; replay float32; original network/loss; exclusive 20% HIL and outcome-balanced policy replay; actor frozen during critic burnin','production_publish':False}
    (out/'config.json').write_text(json.dumps(config,indent=2));service=LearnerServiceConfig(sample_batch_size=128,checkpoint_dir=str(out/'checkpoints'),actor_snapshot_path=str(out/'actor_snapshot.pkl'),push_actor_interval_steps=500,checkpoint_interval_steps=1000)
    learner=Learner(rl,service,PreciseSource(buffer,len(ordered)-1,rows),rng=jax.random.PRNGKey(42));_Q_BURNIN_STEPS=burnin_stop_step(int(learner.state.global_step),args.q_burnin_steps);start=time.monotonic()
    config['q_burnin_stop_step']=_Q_BURNIN_STEPS
    (out/'config.json').write_text(json.dumps(config,indent=2))
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
            metric=aggregate_metric_window(metric_window);metric_window=[]
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
            progress={'phase':'training','pid':os.getpid(),'step':step,'steps':args.steps,'seconds':time.monotonic()-start,'production_publish':False};(out/'operation.json').write_text(json.dumps(progress))
            telemetry_path=os.environ.get('RLT_TELEMETRY_PROGRESS_PATH')
            if telemetry_path:
                target=Path(telemetry_path);temporary=target.with_suffix(target.suffix+'.tmp')
                temporary.write_text(json.dumps({'phase':'training','completed':step,'total':args.steps,'message':'critic / actor training'}));os.replace(temporary,target)
            print('RTC_TRAIN',step,round(progress['seconds'],2),flush=True)
        if step in (500,1000,2000,5000,args.steps):learner.save_checkpoint();evaluate(learner,episodes,out,int(learner.state.global_step))
    (out/'operation.json').write_text(json.dumps({'phase':'completed','steps':args.steps,'global_step':int(learner.state.global_step),'seconds':time.monotonic()-start,'production_publish':False}));print('RTC_TRAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
