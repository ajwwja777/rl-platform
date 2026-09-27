"""Isolated pinned-upstream LearnerService runner; no production publication."""
import os,sys,json,time,hashlib,dataclasses,argparse,pickle
from pathlib import Path
import torch # Required import order for this installed environment.
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
sys.path.insert(0,str(ROOT/'code/openpi-rlt/rlt_online_rl/src'))
import jax,jax.numpy as jnp,yaml
from rlt_online_rl.config import RLTOnlineRLConfig,LearnerServiceConfig,system_config_from_mapping
from rlt_online_rl.trainer import LearnerService
from rlt_online_rl.replay import ReplayBuffer,RLTTransition,TransitionSource
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from .training_flow import session_phase

class OfflineLearner(LearnerService):
 # Only reduce durable status I/O; inherited train_once/train_step unchanged.
 def _write_status(self,progress):
  if int(progress['global_step'])%100==0:super()._write_status(progress)

class Source:
 def __init__(self,buffer,max_episode):self.buffer=buffer;self.max_episode=max_episode
 def stats(self):return {**self.buffer.stats(),'adds_total':len(self.buffer),'max_episode_id':self.max_episode}
 def sample_batch(self,n):return self.buffer.sample(n)

def load(data):
 episodes=[]
 for p in sorted((data/'episodes').glob('*.npz')):
  with np.load(p) as f:m=json.loads(str(f['metadata']));d={k:f[k].copy() for k in f.files if k!='metadata'}
  if m['rows']:episodes.append((m,d))
 train={m['uuid'] for m,d in episodes if m['split']=='train'};val={m['uuid'] for m,d in episodes if m['split']=='val'}
 if train&val:raise ValueError('UUID leakage')
 return episodes

def evaluate(learner,episodes,out,step):
 actor,critic,state,adapter=learner._actor,learner._critic,learner.state,learner._action_adapter
 @jax.jit
 def forward(z,p,r,a,hold):
  pred=actor.actor_mean(state.actor_params,z,p,r)
  return pred,critic.q_values(state.critic_params,z,p,a)[0],critic.q_values(state.critic_params,z,p,r)[0],critic.q_values(state.critic_params,z,p,pred)[0],critic.q_values(state.critic_params,z,p,hold)[0]
 curves=[]
 for m,d in episodes:
  if m['split']!='val':continue
  # Pad evaluation batches to 128 to avoid one XLA compile per episode length.
  values={k:[] for k in ('pred','q_data','q_ref','q_actor','q_hold')}
  for start in range(0,len(d['z_rl']),128):
   at=np.minimum(np.arange(start,start+128),len(d['z_rl'])-1);n=min(128,len(d['z_rl'])-start)
   b={k:v[at] for k,v in d.items()};b=adapter.prepare_training_batch(b)
   hold=adapter.normalize_chunk(np.repeat(d['proprio'][at,None,:],10,axis=1),d['proprio'][at])
   result=forward(*[jnp.asarray(b[k],dtype=jnp.float32) for k in ('z_rl','proprio','ref_chunk','action_chunk')],jnp.asarray(hold))
   for k,v in zip(values,result):values[k].append(np.asarray(v)[:n])
  values={k:np.concatenate(v) for k,v in values.items()};pred=adapter.denormalize_to_abs_chunk(values.pop('pred'),d['proprio']);human=d['source_chunk']==int(TransitionSource.HUMAN)
  pred_error=np.mean((pred[:,:,:6]-d['action_chunk'][:,:,:6])**2,axis=-1);ref_error=np.mean((d['ref_chunk'][:,:,:6]-d['action_chunk'][:,:,:6])**2,axis=-1)
  delta=np.diff(pred[:,:,:6],axis=1);ref_delta=np.diff(d['ref_chunk'][:,:,:6],axis=1)
  metrics={'human_bc_rad2':float(pred_error[human].mean()) if human.any() else None,'reference_human_bc_rad2':float(ref_error[human].mean()) if human.any() else None,'actor_max_adjacent_rad':float(abs(delta).max()),'reference_max_adjacent_rad':float(abs(ref_delta).max()),'actor_delta_rms':float(np.sqrt(np.mean(delta**2))),'reference_delta_rms':float(np.sqrt(np.mean(ref_delta**2)))}
  curves.append({'uuid':m['uuid'],'expert':m['expert'],'phase':m['phase'],'success':m['success'],'position':d['position'].tolist(),'human_ratio':human.mean(1).tolist(),'reward':d['rewards'].sum(1).tolist(),'done':d['done'].tolist(),**{k:v.tolist() for k,v in values.items()},'metrics':metrics})
 report={'step':step,'actor_version':int(state.actor_version),'heldout_kind':'fixed UUID regression set, already used in earlier experiments','episodes':curves,'production_publish':False,'live_RTC_compatible':False}
 human=[x['metrics'] for x in curves if x['metrics']['human_bc_rad2'] is not None]
 report['summary']={'episode_mean_human_bc_rad2':float(np.mean([m['human_bc_rad2'] for m in human])),'episode_mean_reference_human_bc_rad2':float(np.mean([m['reference_human_bc_rad2'] for m in human])),'max_actor_adjacent_rad':max(x['metrics']['actor_max_adjacent_rad'] for x in curves)}
 # Time-aligned early-policy-only Q separation, excluding expert episodes.
 scores={True:[],False:[]}
 for c in curves:
  mask=(np.asarray(c['position'])<.5)&(np.asarray(c['human_ratio'])==0)
  if not c['expert'] and mask.any():scores[c['success']].append(float(np.asarray(c['q_data'])[mask].mean()))
 if scores[True] and scores[False]:
  diff=np.asarray(scores[True])[:,None]-np.asarray(scores[False])[None,:]
  report['summary'].update(early_policy_episode_auc=float((diff>0).mean()+.5*(diff==0).mean()),early_policy_success_episodes=len(scores[True]),early_policy_failure_episodes=len(scores[False]))
 (out/f'evaluation_{step}.json').write_text(json.dumps(report,indent=2));print('UPSTREAM_EVAL',step,json.dumps(report['summary']),flush=True)
 return report['summary']

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=20000);p.add_argument('--delta-weight',type=float);args=p.parse_args();out=args.output.resolve();out.relative_to(RUN.resolve());out.mkdir(exist_ok=False)
 if json.loads((args.data/'progress.json').read_text())['phase']!='completed':raise RuntimeError('dataset incomplete')
 if session_phase()!='stopped':raise RuntimeError('Session must remain ended')
 (out/'operation.json').write_text(json.dumps({'phase':'initializing','pid':os.getpid(),'production_publish':False}))
 episodes=load(args.data);train=[(m,d) for m,d in episodes if m['split']=='train'];delta=[]
 for m,d in train:
  a=d['action_chunk'].copy();a[:,:,:6]-=d['proprio'][:,None,:6];delta.append(a.reshape(-1,7))
 x=np.concatenate(delta);q01,q99=np.quantile(x,[.01,.99],axis=0).astype(np.float32)
 stats=out/'norm_stats_delta.json';stats.write_text(json.dumps({'norm_stats':{'actions':{'q01':q01.tolist(),'q99':q99.tolist()}}},indent=2))
 cfg=yaml.safe_load((ROOT/'code/openpi-rlt/rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml').read_text())['experiment']['rl'];cfg.update(action_norm_stats_path=str(stats),warmup_post_collect_updates=args.steps,freeze_after_warmup=True)
 if args.delta_weight is not None:
  if not np.isfinite(args.delta_weight) or args.delta_weight<=0:raise ValueError('invalid delta weight')
  cfg['delta_weight']=args.delta_weight
 rl=system_config_from_mapping({"rl":cfg}).rl
 # Pinned upstream stratified sampler, not a rewritten hand-balanced sampler.
 buffer=ReplayBuffer(capacity=sum(len(d['z_rl']) for m,d in train),seed=42,sample_strategy='stratified')
 journal=[]
 # Monotonic phase ordering: expert/warmup first, then online. UUID split unchanged.
 ordered=sorted(train,key=lambda x:({'demonstrations':0,'warmup':1,'online':2}[x[0]['phase']],x[0]['uuid']))
 for eid,(m,d) in enumerate(ordered):
  for i in range(len(d['z_rl'])):
   sources=d['source_chunk'][i];human=sources==int(TransitionSource.HUMAN);source=int(TransitionSource.HUMAN if human.all() else TransitionSource.MIXED if human.any() else TransitionSource.RL)
   r=RLTTransition(**{k:d[k][i] for k in ('z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk')},source=source,collection_phase='online' if m['phase']=='online' else 'warmup',success=int(m['success']),intervention_flag=bool(human.any() and not m['expert']),episode_id=eid,step_id=int(d['frame'][i]))
   buffer.add(r);journal.append(r.to_journal_record())
 with (out/'replay_journal.pkl').open('wb') as f:pickle.dump(journal,f)
 service=LearnerServiceConfig(sample_batch_size=128,checkpoint_dir=str(out/'checkpoints'),actor_snapshot_path=str(out/'actor_snapshot.pkl'),push_actor_interval_steps=500,checkpoint_interval_steps=1000)
 provenance={'rl_config':dataclasses.asdict(rl),'dataset_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),'upstream_commit':'c1e40ac360185778c98cf20da2820e22d2d415e7','network_and_loss':'unmodified upstream LearnerService/train_step','sampling':'unmodified upstream stratified 0.4 recent / 0.3 warmup / 0.2 human / 0.1 uniform (overlapping pools)','device':[str(d) for d in jax.devices()],'live_RTC_compatible':False,'production_publish':False}
 (out/'config.json').write_text(json.dumps(provenance,indent=2));learner=OfflineLearner(rl,service,Source(buffer,len(ordered)-1),rng=jax.random.PRNGKey(42),metrics_path=None);start=time.monotonic()
 for step in range(1,args.steps+1):
  if step%100==1 and session_phase()!='stopped':raise RuntimeError('Session changed; stopped offline learner')
  metric=learner.train_once()
  if metric is None:raise RuntimeError('upstream training budget unavailable')
  if step%100==0:
   with (out/'metrics.jsonl').open('a') as mf:mf.write(json.dumps(metric)+'\n')
   progress={'phase':'training','pid':os.getpid(),'step':step,'steps':args.steps,'seconds':time.monotonic()-start,'production_publish':False};(out/'operation.json').write_text(json.dumps(progress));print('UPSTREAM_TRAIN',step,round(progress['seconds'],2),metric['critic_loss'],metric['bc_human_penalty'],flush=True)
  if step in (500,2000,5000,10000,args.steps):learner.save_checkpoint();evaluate(learner,episodes,out,step)
 learner.save_checkpoint();(out/'operation.json').write_text(json.dumps({'phase':'completed','pid':os.getpid(),'steps':args.steps,'seconds':time.monotonic()-start,'production_publish':False,'live_RTC_compatible':False}));print('UPSTREAM_COMPLETE',flush=True)
if __name__=='__main__':main()
