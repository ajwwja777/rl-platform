"""CPU objective decomposition at the retained Warmup checkpoint."""
import os,sys,json,pickle,hashlib
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
base=Path('/data/LFT-W02_data/jiaan/jiaan')
project=base/'projects/rl-platform';out=project/'outputs/model-repair-20261005'
upstream=project/'third_party/openpi-rlt'
sys.path[:0]=[str(upstream/'rlt_online_rl/src'),str(upstream/'packages/openpi-client/src')]
import numpy as np,jax,jax.numpy as jnp
from rlt_online_rl import trainer
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter,jax_denormalize_to_abs_chunk
from rlt_online_rl.networks import apply_reference_dropout
checkpoint=project/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
payload=pickle.loads(checkpoint.read_bytes());config=payload['rl_config'].copy();config['action_norm_stats_path']=str(out/'action_norm_stats.json')
cfg=RLTOnlineRLConfig(**config);adapter=ActionRepresentationAdapter.from_config(cfg);actor,critic=trainer._make_networks(cfg)
state={k:trainer._tree_to_jax(v) for k,v in payload['state'].items()}
data=dict(np.load(out/'precision_inputs.npz'));batch=adapter.prepare_training_batch(data)
indices=np.flatnonzero(~data['raw_verified'])
b={k:jnp.asarray(v[indices]) for k,v in batch.items()}
actor_rng=jax.random.split(jax.random.split(state['rng'])[1])[0]
dropout_rng,sample_rng=jax.random.split(actor_rng)
dropped=apply_reference_dropout(dropout_rng,b['ref_chunk'],cfg.reference_dropout_prob)
human=jnp.isin(b['source_chunk'],jnp.array([2,3]));target=jnp.where(human[...,None],b['action_chunk'],b['ref_chunk'])
q01,q99=jnp.asarray(adapter.stats.q01),jnp.asarray(adapter.stats.q99)

def terms(ap):
    pred=actor.sample_action(ap,sample_rng,b['z_rl'],b['proprio'],dropped,deterministic=False)
    q1=critic.q_values(state['critic_params'],b['z_rl'],b['proprio'],pred)[0]
    pa=jax_denormalize_to_abs_chunk(pred,b['proprio'],q01,q99,action_representation=cfg.action_representation)
    ta=jax_denormalize_to_abs_chunk(target,b['proprio'],q01,q99,action_representation=cfg.action_representation)
    delta=(pa[:,1:,:6]-pa[:,:-1,:6])-(ta[:,1:,:6]-ta[:,:-1,:6])
    return jnp.stack([cfg.online_bc_weight*jnp.mean((pred-target)**2),-cfg.online_q_weight*q1.mean(),cfg.delta_weight*jnp.mean(delta**2)])

grads=jax.jit(jax.jacrev(terms))(state['actor_params'])
flat=np.concatenate([np.asarray(a).reshape(3,-1) for a in jax.tree_util.tree_leaves(grads)],axis=1)
norm=np.linalg.norm(flat,axis=1)
cos=lambda i,j:float(flat[i].dot(flat[j])/(norm[i]*norm[j]))
report={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'devices':[str(d) for d in jax.devices()],
 'data':'Six repeatedly used Online development Episodes; stored FP16 HIL targets, 177 windows, not independent test.',
 'semantics':'Actual Actor loss formula, dropout=.5 and sampled std=.002; frozen checkpoint Critic, no preceding Critic update. Single RNG draw at retained state, not historical batch reconstruction.',
 'terms':['weighted_bc','negative_weighted_q','weighted_native_joint_delta'],
 'values':np.asarray(terms(state['actor_params'])).tolist(),'parameter_gradient_norms':norm.tolist(),
 'q_vs_bc_gradient_cosine':cos(0,1),'delta_vs_bc_gradient_cosine':cos(0,2),
 'q_gradient_norm_over_bc':float(norm[1]/norm[0]),'delta_gradient_norm_over_bc':float(norm[2]/norm[0]),
 'limitations':'Negative cosine does not prove net harmful update or human optimality. Native-radian delta penalty has different units from normalized BC, so numeric coefficient magnitudes alone are misleading.'}
(out/'actor_gradient_diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
