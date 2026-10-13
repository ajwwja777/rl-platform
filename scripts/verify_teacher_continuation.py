#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,pickle,dataclasses
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
import numpy as np,jax,jax.numpy as jnp,optax
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from methods.openpi_rlt.experiments.guarded_retained_actor import make_retained_train_step
from methods.openpi_rlt.experiments.separate_actor_batch import with_separate_actor_batch
from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic
from methods.openpi_rlt.experiments.supported_hil_runtime import SupportedIndex
from methods.openpi_rlt.experiments.supported_sampler import FIELDS
from run_matched_learning import read,arrays
p=Path(sys.argv[1]);o=Path(sys.argv[2]);base=p/'models/rlt/plug_v3_yyshadow';profile=json.loads((base/'history/candidates/supported_hil8_clip_20261010/profile.json').read_text());rows=read(o/'training.pkl');data=arrays(rows)
initial=pickle.loads((base/'history/candidates/supported_hil8_clip_20261010/teacher.pkl').read_bytes());results=[]
arms=list(json.loads((o/'study.json').read_text()).get('arms',{'teacher50':{},'teacher5':{}}))
for arm in arms:
 payload=pickle.loads((o/(arm+'_seed42_u846.pkl')).read_bytes());cfg=RLTOnlineRLConfig(**payload['rl_config']);adapter=ActionRepresentationAdapter.from_config(cfg);actor,critic=trainer._make_networks(cfg)
 critic=HeldGripperCritic(critic,float(adapter.stats.q01[6]),float(adapter.stats.q99[6]))
 index=SupportedIndex(o/'training.pkl',.99,profile['retention_episodes']);raw=index.attach(data)
 b={k:jnp.asarray(v) for k,v in adapter.prepare_training_batch(raw).items()}
 def state(d):return trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in d.items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
 start=state(payload['state']);teacher=trainer._tree_to_jax(initial['actor_params'])
 from methods.openpi_rlt.experiments.actor_q_reduction import with_actor_min_q
 options=payload.get('experimental_options',dict(teacher=50. if arm=='teacher50' else 5.,clip=None,minq=False))
 from methods.openpi_rlt.experiments.critic_stability import with_critic_gradient_diagnostics
 core=make_retained_train_step(.3,teacher,options['teacher'],target_policy='clip')
 if 'experimental_options' in payload:core=with_critic_gradient_diagnostics(core,options['clip'])
 run=with_separate_actor_batch(with_actor_min_q(core,options['minq']))
 sched=np.load(o/(arm+'_seed42_batches.npz'));ci,ai=sched['critic'][-8:],sched['actor'][-8:]
 def advance(s,lo,hi):
  for k in range(lo,hi):
   batch={name:v[ci[k]] for name,v in b.items()}
   batch['actor_batch']={name:v[ai[k]] for name,v in b.items()}
   s,_=run(s,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=.1,delta_weight=10.,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
  return s
 direct=advance(start,0,8);half=advance(start,0,4)
 restored=state(pickle.loads(pickle.dumps({k:trainer._tree_to_numpy(getattr(half,k)) for k in payload['state']})))
 resumed=advance(restored,4,8);leaves=0
 for k in payload['state']:
  for x,y in zip(jax.tree_util.tree_leaves(getattr(direct,k)),jax.tree_util.tree_leaves(getattr(resumed,k))):np.testing.assert_array_equal(x,y);leaves+=1
 # Native actor wire payload and reconstructed action must preserve exact parameters.
 wire=pickle.loads(pickle.dumps(dict(rl_config=payload['rl_config'],actor_params=payload['state']['actor_params'],version=int(start.actor_version),global_step=int(start.global_step))))
 args=(jax.random.PRNGKey(0),b['z_rl'][:32],b['proprio'][:32],b['ref_chunk'][:32])
 x=np.asarray(actor.sample_action(start.actor_params,*args,deterministic=True));y=np.asarray(actor.sample_action(trainer._tree_to_jax(wire['actor_params']),*args,deterministic=True));np.testing.assert_array_equal(x,y)
 results.append(dict(arm=arm,seed=42,resume_8_vs_4_plus_4_exact=True,compared_leaves=leaves,actor_export_max_abs=float(np.max(np.abs(x-y))),verification_updates_only=True))
(o/'resume-export-check.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
