#!/usr/bin/env python3
"""Local loss gradients and frozen-input perturbations, not attention or causal attribution."""
import argparse,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"third_party/openpi-rlt/rlt_online_rl/src"),str(ROOT/"third_party/openpi-rlt/packages/openpi-client/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np
from integrations.cobot_runtime.replay_audit import metadata,annotate,atomic_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=ROOT/"configs/rlt/plug_v3_yyshadow/online_rl.yaml");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 import yaml,jax,jax.numpy as jnp
 from rlt_online_rl import trainer
 from rlt_online_rl.config import RLTOnlineRLConfig
 from rlt_online_rl.action_representation import ActionRepresentationAdapter,jax_denormalize_to_abs_chunk
 from rlt_online_rl.networks import apply_reference_dropout
 c=yaml.safe_load(a.config.read_text());cfg=RLTOnlineRLConfig(**c["experiment"]["rl"])
 ad=ActionRepresentationAdapter.from_config(cfg);q01=jnp.asarray(ad.stats.q01);q99=jnp.asarray(ad.stats.q99)
 path=Path(c["runtime"]["learner_service"]["checkpoint_dir"])/"latest.pkl"
 d=pickle.load(path.open("rb"));ap=trainer._tree_to_jax(d["state"]["actor_params"]);cp=trainer._tree_to_jax(d["state"]["critic_params"])
 actor,critic=trainer._make_networks(cfg)
 rows=[]
 with open(c["runtime"]["replay"]["journal_path"],"rb") as f:
  while True:
   try:rows.append(pickle.load(f))
   except EOFError:break
 meta=annotate([metadata(r) for r in rows]);rng=np.random.default_rng(42)
 raw={k:np.stack([r[k] for r in rows]) for k in rows[0] if k!="collection_phase"}
 data={k:jnp.asarray(v) for k,v in ad.prepare_training_batch(raw).items()}
 @jax.jit
 def terms(params,b):
  dk,sk=jax.random.split(jax.random.PRNGKey(42))
  ref=apply_reference_dropout(dk,b["ref_chunk"],cfg.reference_dropout_prob)
  act=actor.sample_action(params,sk,b["z_rl"],b["proprio"],ref,deterministic=False)
  q,_=critic.q_values(cp,b["z_rl"],b["proprio"],act)
  h=jnp.isin(b["source_chunk"],jnp.array([2,3]));target=jnp.where(h[...,None],b["action_chunk"],b["ref_chunk"])
  native=jax_denormalize_to_abs_chunk(act,b["proprio"],q01,q99,action_representation=cfg.action_representation)
  native_target=jax_denormalize_to_abs_chunk(target,b["proprio"],q01,q99,action_representation=cfg.action_representation)
  return jnp.array([cfg.online_bc_weight*jnp.mean((act-target)**2),-cfg.online_q_weight*jnp.mean(q),
   cfg.delta_weight*jnp.mean((jnp.diff(native[...,:6],axis=1)-jnp.diff(native_target[...,:6],axis=1))**2)])
 jac=jax.jit(jax.jacrev(terms))
 groups={"all":list(range(len(rows)))}
 for dim in ["outcome","phase","portion","hil"]:
  for val in sorted(set(str(r[dim]) for r in meta)):
   groups[dim+"="+val]=[i for i,r in enumerate(meta) if str(r[dim])==val]
 report={"schema":1,"generated_at":time.time(),"checkpoint":str(path),"learner_step":int(d["state"]["global_step"]),
  "semantics":"Training-seen snapshot, 128 sampled rows per group, same dropout/noise seed; local gradients, not causal per-example attribution.",
  "gradients":[],"ablations":[],"action_probes":[]}
 for name,indices in groups.items():
  ids=rng.choice(indices,min(128,len(indices)),replace=False);b={k:v[ids] for k,v in data.items()}
  g=jac(ap,b);flat=np.concatenate([np.asarray(x).reshape(3,-1) for x in jax.tree_util.tree_leaves(g)],axis=1)
  norms=np.linalg.norm(flat,axis=1)
  report["gradients"].append(dict(group=name,count=len(ids),bc_norm=float(norms[0]),q_norm=float(norms[1]),delta_norm=float(norms[2]),
   bc_q_cosine=float(np.dot(flat[0],flat[1])/max(norms[0]*norms[1],1e-12)),weighted_terms=np.asarray(terms(ap,b)).tolist()))
 ids=rng.choice(len(rows),128,replace=False);b={k:v[ids] for k,v in data.items()}
 @jax.jit
 def pred(z,proprio,ref):return actor.sample_action(ap,jax.random.PRNGKey(0),z,proprio,ref,deterministic=True)
 base=np.asarray(pred(b["z_rl"],b["proprio"],b["ref_chunk"]));base_native=ad.denormalize_to_abs_chunk(base,raw["proprio"][ids])
 for name,key,replacement in [("RL feature -> zeros","z_rl",jnp.zeros_like(b["z_rl"])),("Proprio -> dataset median","proprio",jnp.broadcast_to(jnp.median(data["proprio"],axis=0),b["proprio"].shape)),("Reference -> zeros","ref_chunk",jnp.zeros_like(b["ref_chunk"]))]:
  x=dict(b);x[key]=replacement
  prediction=np.asarray(pred(x["z_rl"],x["proprio"],x["ref_chunk"]))
  native=ad.denormalize_to_abs_chunk(prediction,raw["proprio"][ids])
  report["ablations"].append(dict(name=name,action_shift_rms_rad=float(np.sqrt(np.mean((native[...,:6]-base_native[...,:6])**2)))))
 # Local Q response toward recorded human actions: same state, identical mean actor, signed perturbation.
 human=np.array([i for i,r in enumerate(meta) if r["hil"]]);ids=rng.choice(human,min(128,len(human)),replace=False)
 b={k:v[ids] for k,v in data.items()};baseline=pred(b["z_rl"],b["proprio"],b["ref_chunk"]);h=jnp.isin(b["source_chunk"],jnp.array([2,3]))[...,None]
 @jax.jit
 def qvalue(actions):return critic.q_values(cp,b["z_rl"],b["proprio"],actions)[0]
 for alpha in [-1.,0.,.25,.5,1.]:
  actions=baseline+alpha*jnp.where(h,b["action_chunk"]-baseline,0)
  report["action_probes"].append(dict(human_direction=alpha,q_mean=float(jnp.mean(qvalue(actions)))))
 report["limitations"]=["No camera images or visual attention were reconstructed from cached z_rl.","Zero/median input replacement can be out of distribution.","A gradient describes a local parameter update, not the downstream causal effect of one recording.","Q-to-human directional preference is only a diagnostic, not a robot success metric."]
 atomic_json(a.output,report);print(a.output,flush=True)
if __name__=="__main__":main()
