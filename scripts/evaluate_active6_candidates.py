from pathlib import Path
import os,json,pickle,hashlib,sys
import numpy as np
import jax,jax.numpy as jnp
import argparse
p=argparse.ArgumentParser();p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--studies',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
M=args.assets_root;O=args.output;O.mkdir(exist_ok=False)
sys.path[:0]=[str(M),str(M/'third_party/openpi-rlt/rlt_online_rl/src'),str(M/'third_party/openpi-rlt/packages/openpi-client/src')]
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns

def rows(path):
 result=[]
 with path.open('rb')as f:
  while True:
   try:result.append(pickle.load(f))
   except EOFError:break
 return result

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
journal=M/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260924_v1/holdout/replay/replay_journal.pkl'
checkpoint=M/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
trainjournal=checkpoint.parent.parent/'replay/replay_journal.pkl'
dev=rows(journal);trained=rows(trainjournal)
def identity(r):return(int(r['collection_phase_id']),int(r['episode_id']))
assert not {identity(r)for r in dev}&{identity(r)for r in trained},'External DEV overlaps initial5k training journal'
fields=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk','source','success','intervention_flag','episode_id','step_id','collection_phase_id']
data={k:np.stack([r[k]for r in dev])for k in fields}
payload=pickle.loads(checkpoint.read_bytes());cfg=RLTOnlineRLConfig(**dict(payload['rl_config'],action_norm_stats_path=str(M/'outputs/model-repair-20261005/full_pool/action_norm_stats.json')))
adapter=ActionRepresentationAdapter.from_config(cfg);actor,critic=trainer._make_networks(cfg)
b={k:jnp.asarray(v)for k,v in adapter.prepare_training_batch(data).items()}
held={k:v.copy()for k,v in data.items()}
held['action_chunk']=held['action_chunk'].astype(np.float32);held['ref_chunk']=held['ref_chunk'].astype(np.float32)
held['action_chunk'][...,6]=held['proprio'][:,None,6];held['ref_chunk'][...,6]=held['proprio'][:,None,6]
bheld={k:jnp.asarray(v)for k,v in adapter.prepare_training_batch(held).items()}
groups={}
for i,row in enumerate(dev):groups.setdefault(identity(row),[]).append(i)
human=np.isin(data['source_chunk'],[2,3])
observed,integrity=reconstruct_observed_returns(data,cfg.gamma)
assert np.isfinite(observed).all()
@jax.jit
def predict(params):return actor.sample_action(params,jax.random.PRNGKey(0),b['z_rl'],b['proprio'],b['ref_chunk'],deterministic=True)
@jax.jit
def q(cp,a):
 a=a.at[...,6].set(bheld['action_chunk'][...,6])
 q1,q2=critic.q_values(cp,b['z_rl'],b['proprio'],a);return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)

def evaluate(state,name):
 pred=predict(trainer._tree_to_jax(state['actor_params']));physical=np.asarray(adapter.denormalize_to_abs_chunk(np.asarray(pred),data['proprio']))
 cp=trainer._tree_to_jax(state['critic_params'])
 pa=pred.at[...,6].set(bheld['action_chunk'][...,6])
 endpoint=jnp.where(jnp.asarray(human)[...,None],bheld['action_chunk'],pa)
 qa=np.asarray(q(cp,pa));qr=np.asarray(q(cp,bheld['ref_chunk']));qh=np.asarray(q(cp,endpoint));qrecord=np.asarray(q(cp,bheld['action_chunk']))
 es=[]
 for key,ids in sorted(groups.items()):
  ids=np.asarray(ids);mask=human[ids];h=mask.any(1)
  outcome='assisted_success'if mask.any()and data['success'][ids].any()else'autonomous_success'if data['success'][ids].any()else'failure'
  item=dict(phase_id=key[0],episode_id=key[1],outcome=outcome,windows=len(ids),seen_by_initial5k=False,observed_return_mean=float(observed[ids].mean()),qactor=qa[ids].mean(0).tolist(),qref=qr[ids].mean(0).tolist(),qrecord=qrecord[ids].mean(0).tolist(),recorded_q_mse=((qrecord[ids]-observed[ids,None])**2).mean(0).tolist(),qrecord_min=float(qrecord[ids].min()),qrecord_max=float(qrecord[ids].max()),
   human_mae_mrad=np.mean(np.abs(physical[ids]-data['action_chunk'][ids])[mask][:,:6],axis=0).tolist()if mask.any()else None,
   reference_mae_mrad=np.mean(np.abs(physical[ids]-data['ref_chunk'][ids])[~mask][:,:6],axis=0).tolist()if(~mask).any()else None,
   joint_only_hil_minus_actor_q=(qh[ids][h]-qa[ids][h]).mean(0).tolist()if h.any()else None,
   joint_only_hil_minus_ref_q=(qh[ids][h]-qr[ids][h]).mean(0).tolist()if h.any()else None,
   pure_hil_record_minus_actor_q=(qrecord[ids][mask.all(1)]-qa[ids][mask.all(1)]).mean(0).tolist()if mask.all(1).any()else None)
  for field in ['human_mae_mrad','reference_mae_mrad']:
   if item[field]is not None:item[field]=(np.asarray(item[field])*1000).tolist()
  es.append(item)
 np.savez_compressed(O/('external20_'+name+'.npz'),actor=physical,qactor=qa,qref=qr,qendpoint=qh,qrecord=qrecord)
 return es
baseline=evaluate(payload['state'],'initial5k');runs=[]
for study_path in args.studies:
 study_name=study_path.parent.name
 r=json.loads(study_path.read_text())
 for run in r['runs']:
  cp=Path(run['checkpoint']);p=pickle.loads(cp.read_bytes());name='%s_%s_seed%d'%(study_name,run['variant'],run['seed'])
  runs.append(dict(study=study_name,variant=run['variant'],seed=run['seed'],checkpoint=str(cp),checkpoint_sha256=sha(cp),evaluation=evaluate(p['state'],name)))
report=dict(code_head=os.popen('git -C '+str(M)+' rev-parse HEAD').read().strip(),script_sha256=sha(Path(__file__)),journal=str(journal),journal_sha256=sha(journal),episodes=len(groups),windows=len(dev),initial_training_overlap=False,baseline=baseline,runs=runs,devices=[str(d)for d in jax.devices()],limits=['External20 Episodes are excluded from initial5k journal but repeatedly used for selection: DEV, not independent TEST.','Evaluation Actor proposals are reconstructed at current weights, not historical pre-HIL proposals.','All Q alternatives share normalized measured gripper position; only active six joint actions differ. This diagnoses Critic action preference under the user control contract, not physically observed alternative outcomes.','Stored feedback targets are fitting proxies; actual command identity/optimality and moved-socket autonomous capability are not established.'])
(O/'external20_active6.json').write_text(json.dumps(report,indent=2,allow_nan=False));assert sha(journal)==report['journal_sha256'];print(json.dumps(dict(episodes=len(groups),windows=len(dev),runs=len(runs),outcomes={k:sum(e['outcome']==k for e in baseline)for k in ['autonomous_success','assisted_success','failure']})))
