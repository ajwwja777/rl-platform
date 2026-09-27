"""Offline modality sensitivity on existing RTC state; no robot I/O."""
import json,pickle,argparse
from pathlib import Path
import torch,numpy as np
from .upstream_baseline import load,ROOT,RUN
from .rtc_upstream_core import original_context
import jax,jax.numpy as jnp
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import TwinCritic,ChunkActor,_layer_norm
from rlt_online_rl.action_representation import ActionRepresentationAdapter

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,default=5000);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 cfg=RLTOnlineRLConfig(**json.loads((args.run/'config.json').read_text())['rl_config']);adapter=ActionRepresentationAdapter.from_config(cfg);state=pickle.load((args.run/f'checkpoints/step_{args.step}.pkl').open('rb'))['state'];episodes=load(args.data)
 actor=ChunkActor(cfg.z_dim,cfg.proprio_dim,10,7,cfg.actor_hidden_dim,cfg.actor_num_layers,cfg.fixed_std);critic=TwinCritic(cfg.z_dim,cfg.proprio_dim,10,7,cfg.critic_hidden_dim,cfg.critic_num_layers)
 q=jax.jit(lambda z,p,a:critic.q_values(state['critic_params'],z,p,a)[0]);mu=jax.jit(lambda z,p,r:actor.actor_mean(state['actor_params'],z,p,r))
 grad=jax.jit(jax.grad(lambda z,p,a:jnp.sum(q(z,p,a)),argnums=(0,1,2)))
 batches=[];meta=[]
 for m,d in episodes:
  if m['split']!='val':continue
  b=adapter.prepare_training_batch(d);batches.append(b);meta.extend([(m['uuid'],int(x),int(y[0]),m['success']) for x,y in zip(d['delay'],d['source_chunk'])])
 keys=['z_rl','proprio','action_chunk','ref_chunk'];values={k:np.concatenate([b[k] for b in batches]) for k in keys};rng=np.random.default_rng(42);at=np.sort(rng.choice(len(meta),min(512,len(meta)),replace=False));values={k:v[at] for k,v in values.items()};meta=[meta[i] for i in at]
 z,pr,a,ref=[jnp.asarray(values[k]) for k in keys];baseq=np.asarray(q(z,pr,a));basemu=np.asarray(mu(z,pr,ref));groups={}
 for i,(_,delay,source,_) in enumerate(meta):groups.setdefault((delay,source),[]).append(i)
 perm=np.arange(len(meta))
 for ids in groups.values():perm[ids]=rng.permutation(ids)
 out={'scope':'within delay/source permutation stress test, not a causal feature-importance proof','samples':len(meta),'q_std':float(baseq.std()),'groups':{str(k):len(v) for k,v in groups.items()},'critic':{},'actor':{},'branch_rms':{}}
 for name,pos in [('rl_token',0),('proprio_rtc',1),('action_chunk',2)]:
  x=[z,pr,a];x[pos]=x[pos][perm];diff=np.asarray(q(*x))-baseq;out['critic'][name]={'permutation_q_rms':float(np.sqrt(np.mean(diff**2))),'relative_to_q_std':float(np.sqrt(np.mean(diff**2))/(baseq.std()+1e-8))}
 for name,pos in [('rl_token',0),('proprio_rtc',1),('reference_chunk',2)]:
  x=[z,pr,ref];x[pos]=x[pos][perm];diff=np.asarray(mu(*x))-basemu;out['actor'][name]={'permutation_normalized_action_rms':float(np.sqrt(np.mean(diff**2)))}
 gs=grad(z,pr,a)
 for name,g,x in zip(['rl_token','proprio_rtc','action_chunk'],gs,[z,pr,a]):
  standardized=np.asarray(g)*np.asarray(x).std(axis=0,keepdims=True);out['critic'][name]['standardized_gradient_rms']=float(np.sqrt(np.mean(standardized**2)))
 for role,params,chunk,key in [('actor',state['actor_params'],ref,'ref_proj'),('critic_q1',state['critic_params']['q1'],a,'action_proj')]:
  out['branch_rms'][role]={}
  for label,value,project,tanh in [('rl_token',z,'z_proj',False),('proprio_rtc',pr,'proprio_proj',True),('chunk',chunk.reshape(len(at),-1),key,True)]:
   encoded=_layer_norm(value@params[project]['w']+params[project]['b']);encoded=jnp.tanh(encoded) if tanh else encoded
   out['branch_rms'][role][label]={'width':encoded.shape[-1],'rms':float(jnp.sqrt(jnp.mean(encoded**2))),'mean_l2':float(jnp.mean(jnp.linalg.norm(encoded,axis=-1)))}
 # Same observed state: does critic prefer actual human correction to reference?
 human=np.array([x[2]==2 for x in meta]);qr=np.asarray(q(z,pr,ref));out['human_vs_reference']={'windows':int(human.sum()),'q_human_minus_ref_mean':float((baseq-qr)[human].mean()),'fraction_q_human_higher':float(((baseq-qr)[human]>0).mean())}
 args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
