#!/usr/bin/env python3
"""Native in-memory CPU load/resume/budget/Actor-wire checks, no service or field writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,pickle,json,hashlib,threading
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
import numpy as np,jax,jax.numpy as jnp,optax
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.inference import ActorService,ActorRequest,RLTPolicyInferenceWrapper
from methods.openpi_rlt.experiments.supported_hil_runtime import load_profile,build_components,SupportedIndex
from methods.openpi_rlt.experiments.supported_sampler import SupportedSampler,unpack_actor_batch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    rows=[]
    with Path(p).open('rb') as stream:
        while True:
            try:rows.append(pickle.load(stream))
            except EOFError:break
    return rows
package,output=Path(sys.argv[1]),Path(sys.argv[2])
assert not output.exists() and all(d.platform=='cpu' for d in jax.devices())
protected={str(f):sha(f) for f in package.rglob('*') if f.is_file()}
profile=load_profile(package/'profile.json');payload=pickle.loads((package/'checkpoints/latest.pkl').read_bytes())
assert payload['candidate_profile_sha256']==sha(package/'profile.json')
cfg=RLTOnlineRLConfig(**dict(payload['rl_config'],action_norm_stats_path=str(package/'action_norm_stats.json')))
actor,critic,run=build_components(cfg,profile);adapter=ActionRepresentationAdapter.from_config(cfg)
def state(d):return trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in d.items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
initial=state(payload['state']);rows=read(package/'replay/replay_journal.pkl')
assert len(rows)==3623
native=object.__new__(trainer.LearnerService);native._checkpoint_dir=str(package/'checkpoints');native._rl_config=cfg
native._state=native._load_latest_checkpoint();native._write_status=lambda progress:None
for a,b in zip(jax.tree.leaves(native.state),jax.tree.leaves(initial)):np.testing.assert_array_equal(a,b)
budgets={}
for add in [0,1,18,39]:
 info=native._refresh_progress(dict(size=len(rows)+add,adds_total=len(rows)+add));budgets[add]=info['pending_update_budget'];assert budgets[add]==add
index=SupportedIndex(package/'replay/replay_journal.pkl',cfg.gamma,profile['retention_episodes'])
def sampler(ss):return SupportedSampler(index,adapter,state=ss)
def advance(s,ss,n):
 samp=sampler(ss);receipt=[]
 for _ in range(n):
  raw=samp.sample_batch(128);batch=unpack_actor_batch(adapter.prepare_training_batch(raw))
  batch=jax.tree.map(jnp.asarray,batch)
  s,m=run(s,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=.1,delta_weight=10.,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
  assert all(np.isfinite(float(v)) for v in m.values());receipt.append(samp.last_receipt)
 return s,samp.state(),receipt
direct,sd,rd=advance(initial,payload['sampling_state'],8)
half,sh,rh=advance(initial,payload['sampling_state'],4)
serialized=pickle.loads(pickle.dumps(dict(state={k:trainer._tree_to_numpy(getattr(half,k)) for k in payload['state']},sampler=sh)))
restored,sr,rr=advance(state(serialized['state']),serialized['sampler'],4)
leaves=0
for k in payload['state']:
 for x,y in zip(jax.tree.leaves(getattr(direct,k)),jax.tree.leaves(getattr(restored,k))):np.testing.assert_array_equal(x,y);leaves+=1
assert rd==rh+rr and sd==sr
obj=object.__new__(ActorService);obj._rl_config=cfg;obj._action_adapter=adapter;obj._wrapper=RLTPolicyInferenceWrapper(cfg);obj._lock=threading.Lock();obj._actor_params=None;obj._actor_version=-1;obj._rng=jax.random.PRNGKey(0);obj._snapshot_path=str(package/'actor_snapshot/actor_snapshot.pkl');obj._logged_missing_params=False;obj._try_reload_snapshot()
assert obj.actor_param_version==4063
maxerr=0.
for i in np.linspace(0,len(rows)-1,32,dtype=int):
 r=rows[i];req=ActorRequest(z_rl=r['z_rl'],proprio=r['proprio'],ref_chunk=r['ref_chunk'],deterministic=True,request_id=str(i),episode_id=int(r['episode_id']),step_id=int(r['step_id']))
 response=obj.infer(req)
 norm=adapter.normalize_ref_chunk(req.ref_chunk,req.proprio);a=actor.sample_action(initial.actor_params,jax.random.PRNGKey(0),jnp.asarray(req.z_rl)[None],jnp.asarray(req.proprio)[None],jnp.asarray(norm)[None],deterministic=True)
 expected=adapter.denormalize_to_abs_chunk(np.asarray(a)[0],req.proprio);maxerr=max(maxerr,float(np.max(np.abs(response.refined_chunk-expected))))
assert maxerr<1e-6
assert protected=={str(f):sha(f) for f in package.rglob('*') if f.is_file()}
result=dict(status='verified_software_only',candidate=str(package),checkpoint_step=8127,actor_version=4063,native_budget=budgets,full_state_leaves_exact=leaves,resume_8_vs_4_plus_4_exact=True,both_sampler_rngs_and_all_8_receipts_exact=True,native_actor32_max_abs_error=maxerr,read_only_candidate_sha256=protected,limits=['CPU native methods, no Actor/Replay/Learner services instantiated','No Stage1 GPU forward or robot motion','No autonomous improvement claim'])
output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='read_only_candidate_sha256'}))
