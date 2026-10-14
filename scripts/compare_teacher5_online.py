#!/usr/bin/env python3
"""Read-only CPU comparison of a Teacher5 runtime and its initial package.

Only writes the requested output directory. Reuses fixed old/new development
Episode manifests; no optimization, publication, service or robot operation.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,sys,pickle,json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
import numpy as np,jax,jax.numpy as jnp
from rlt_online_rl import trainer,networks
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from methods.openpi_rlt.experiments.supported_hil_runtime import load_profile,build_components
from methods.openpi_rlt.experiments.supported_sampler import FIELDS
from methods.openpi_rlt.experiments.journal_import import remap_collection
from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns

def read(path):
 out=[]
 with Path(path).open('rb') as f:
  while True:
   try:out.append(pickle.load(f))
   except EOFError:return out

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(r):return int(r['collection_phase_id']),int(r['episode_id'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert all(d.platform=='cpu'for d in jax.devices())
 a.output.mkdir(exist_ok=False,parents=True)
 oldroot=a.assets/'models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010'
 collection=a.assets/'outputs/matched-learning-20261012';mf=collection/'comparison-manifest.json'
 paths=[a.runtime/'replay/replay_journal.pkl',a.runtime/'checkpoints/latest.pkl',a.baseline/'checkpoints/latest.pkl',a.runtime/'profile.json',a.runtime/'teacher.pkl',a.runtime/'action_norm_stats.json',mf,collection/'exploration-29ep-1068.pkl',oldroot/'replay/replay_journal.pkl',a.assets/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260924_v1/holdout/replay/replay_journal.pkl']
 protected={str(f):sha(f)for f in paths}
 train=read(paths[0]);old=read(oldroot/'replay/replay_journal.pkl');rawnew=read(collection/'exploration-29ep-1068.pkl');_,newdev=remap_collection(old,rawnew,json.load(open(mf))['new_data']);olddev=read(paths[-1]);dev=newdev+olddev
 assert not set(map(identity,train))&set(map(identity,dev))
 records=train+olddev+newdev
 split=['TRAIN_new4'if identity(r)[1]>=10040 else 'TRAIN_old' for r in train]+['DEV_old_reused']*len(olddev)+['DEV_new_reused']*len(newdev)
 groups=collections.defaultdict(list)
 for i,r in enumerate(records):groups[(split[i],*identity(r))].append(i)
 keys=[(*identity(r),int(r['step_id']))for r in records];assert len(keys)==len(set(keys))
 raw={k:np.stack([r[k]for r in records])for k in FIELDS};profile=load_profile(a.runtime/'profile.json');latest=pickle.loads(paths[1].read_bytes());cfg=RLTOnlineRLConfig(**dict(latest['rl_config'],action_norm_stats_path=str(a.runtime/'action_norm_stats.json')))
 actor,critic,_=build_components(cfg,profile);adapter=ActionRepresentationAdapter.from_config(cfg);data=adapter.prepare_training_batch(raw);observed,integrity=reconstruct_observed_returns(raw,cfg.gamma);assert np.isfinite(observed).all()
 actfun=jax.jit(actor.actor_mean);qfun=jax.jit(critic.q_values)
 base=None;allmodels=[]
 for label,path in [('initial8127',a.baseline/'checkpoints/latest.pkl'),('online8234',a.runtime/'checkpoints/latest.pkl')]:
  payload=pickle.loads(path.read_bytes());state=payload['state'];aparam=trainer._tree_to_jax(state['actor_params']);cparam=trainer._tree_to_jax(state['critic_params']);tapa=trainer._tree_to_jax(state['target_actor_params']);tcpa=trainer._tree_to_jax(state['target_critic_params']);pred=[];qs={name:[]for name in ['actor','reference','record']};tds=[]
  for start in range(0,len(records),128):
   sl=slice(start,start+128);bb={k:jnp.asarray(v[sl])for k,v in data.items()};action=actfun(aparam,bb['z_rl'],bb['proprio'],bb['ref_chunk']);pred.append(np.asarray(action))
   for name,actions in [('actor',action),('reference',bb['ref_chunk']),('record',bb['action_chunk'])]:
    q1,q2=qfun(cparam,bb['z_rl'],bb['proprio'],actions);qs[name].append(np.stack([q1,q2,np.minimum(q1,q2)],-1))
   td=networks.build_td_target(actor,tapa,critic,tcpa,bb['next_z_rl'],bb['next_proprio'],bb['next_ref_chunk'],bb['rewards'],bb['done'],cfg.gamma,jax.random.PRNGKey(start));tds.append(np.asarray(td))
  physical=np.asarray(adapter.denormalize_to_abs_chunk(np.concatenate(pred),raw['proprio']));q={k:np.concatenate(v)for k,v in qs.items()};td=np.concatenate(tds);mixed=np.clip((1-profile['mc_weight'])*td+profile['mc_weight']*observed,0,1)
  assert all(np.isfinite(v).all()for v in [physical,td,mixed,*q.values()])
  if base is None:base=physical.copy()
  np.savez_compressed(a.output/(label+'.npz'),actor=physical,qactor=q['actor'],qref=q['reference'],qrecord=q['record'],native_td=td,mixed_target=mixed)
  episodes=[]
  for (part,phase,ep),idx in groups.items():
   ix=np.asarray(idx);human=np.isin(raw['source_chunk'][ix],[2,3]);success=bool(raw['success'][ix].any());outcome=('assisted_success'if human.any()else 'autonomous_success')if success else 'failure'
   error=np.abs(physical[ix,:,:6]-raw['action_chunk'][ix,:,:6])*1000;referr=np.abs(physical[ix,:,:6]-raw['ref_chunk'][ix,:,:6])*1000;change=np.abs(physical[ix,:,:6]-base[ix,:,:6])*1000
   e={'split':part,'phase':phase,'episode':ep,'outcome':outcome,'windows':len(ix)}
   for name,values in [('human',error[human]),('reference',referr[~human]),('record',error.reshape(-1,6)),('actor_change',change.reshape(-1,6))]:
    e[name]={'points':len(values),'mean_mrad':float(values.mean()),'joint_mean_mrad':values.mean(0).tolist(),'p95_mrad':float(np.quantile(values,.95)),'joint_p95_mrad':np.quantile(values,.95,axis=0).tolist(),'p99_mrad':float(np.quantile(values,.99)),'max_mrad':float(values.max())}if len(values)else None
   for name in q:e['q_'+name]=q[name][ix].mean(0).tolist()
   e['qrecord_mse_observed']=((q['record'][ix]-observed[ix,None])**2).mean(0).tolist();e['record_minus_actor_q']=(q['record'][ix]-q['actor'][ix]).mean(0).tolist();e['pure_hil_record_minus_actor_q']=(q['record'][ix][human.all(1)]-q['actor'][ix][human.all(1)]).mean(0).tolist()if human.all(1).any()else None
   e['observed_return_mean']=float(observed[ix].mean());e['native_td_mean']=float(td[ix].mean());e['mixed_target_mean']=float(mixed[ix].mean());episodes.append(e)
  allmodels.append({'name':label,'step':int(state['global_step']),'actor_version':int(state['actor_version']),'sha256':sha(path),'episodes':episodes})
 report={'protected_sha':protected,'models':allmodels,'data':{'TRAIN':len(train),'DEV_old_reused':len(olddev),'DEV_new_reused':len(newdev)},'episode_counts':dict(collections.Counter(k[0]for k in groups)),'no_train_dev_overlap':True,'observed_integrity':integrity,'CPU':[str(d)for d in jax.devices()],'limits':['Repeated DEV, not independent TEST','Episode-weighted, overlapping windows not independent','Q1 actor objective vs minQ TD; paired Q is not observed counterfactual outcome','Fixed gripper excluded; six active joints only','Raw Actor proposals compared; does not establish filtered 50Hz physical trajectories']}
 assert protected=={str(f):sha(f)for f in paths}
 (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 print(json.dumps({k:v for k,v in report.items()if k not in ['models','protected_sha','observed_integrity']}))
if __name__=='__main__':main()
