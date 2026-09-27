"""EXPERIMENT: critic action is executed command; actor Q includes deployment transform.
Original upstream architecture, BC/delta/optimizer/target cadence retained.
Never imported by production runtime or auto-published.
"""
import functools,types
import torch,numpy as np
from .rtc_upstream_core import original_context,rtc_proprio
from .rtc_queue import PIPER_LOWER,PIPER_UPPER
import jax,jax.numpy as jnp,optax
from rlt_online_rl import trainer as upstream
from rlt_online_rl.action_representation import ActionRepresentationAdapter,jax_denormalize_to_abs_chunk
def executed_normalized(pred,p,ref,cfg):
 adapter=ActionRepresentationAdapter.from_config(cfg)
 # Exact runtime linear projection followed by 30Hz command conditioning.
 raw=jax_denormalize_to_abs_chunk(pred,p,jnp.asarray(adapter.stats.q01),jnp.asarray(adapter.stats.q99),action_representation=cfg.action_representation)
 refabs=jax_denormalize_to_abs_chunk(ref,p,jnp.asarray(adapter.stats.q01),jnp.asarray(adapter.stats.q99),action_representation=cfg.action_representation)
 t=jnp.arange(10,dtype=jnp.float32)-4.5;delta=raw[:,:,:6]-refabs[:,:,:6]
 fit=delta.mean(1,keepdims=True)+(delta*t[None,:,None]).sum(1,keepdims=True)/jnp.sum(t*t)*t[None,:,None]
 targets=refabs[:,:,:6]+fit
 c=jnp.concatenate((p[:,7:14],p[:,:7],p[:,14:]),axis=1);prefix=c[:,14:98].reshape(-1,6,14)+c[:,None,:14]
 anchor=jnp.where(c[:,-1,None]>0,prefix[:,-1,7:13],c[:,7:13]);prev=jnp.where(c[:,-1,None]>0,prefix[:,-1,7:13]-prefix[:,-2,7:13],jnp.zeros_like(anchor))
 alpha=1-np.exp(-1/(30*(-.05/np.log(.65))));out=[]
 for k in range(10):
  target=jnp.clip(targets[:,k],jnp.asarray(PIPER_LOWER),jnp.asarray(PIPER_UPPER));dq=jnp.clip(alpha*(target-anchor),-.1/30,.1/30);dq=jnp.clip(dq,prev-.9/900,prev+.9/900);nxt=jnp.clip(anchor+dq,jnp.asarray(PIPER_LOWER),jnp.asarray(PIPER_UPPER));prev=nxt-anchor;anchor=nxt;out.append(nxt)
 cmd=jnp.concatenate((jnp.stack(out,1),jnp.broadcast_to(p[:,None,6:7],(p.shape[0],10,1))),axis=-1)
 delta=cmd.at[:,:,:6].add(-p[:,None,:6]) if cfg.action_representation!='abs' else cmd
 return (delta-jnp.asarray(adapter.stats.q01))/(jnp.asarray(adapter.stats.q99-adapter.stats.q01)+1e-6)*2-1

def update_critic(state,batch,actor,critic,rl_config):
    critic_rng,next_rng=jax.random.split(state.rng)
    def loss_fn(params):
        next_action=actor.sample_action(state.target_actor_params,critic_rng,
            batch['next_z_rl'],batch['next_proprio'],batch['next_ref_chunk'],deterministic=False)
        next_action=executed_normalized(next_action,batch['next_proprio'],batch['next_ref_chunk'],rl_config)
        nq1,nq2=critic.q_values(state.target_critic_params,batch['next_z_rl'],batch['next_proprio'],next_action)
        target=jnp.sum(batch['rewards']*rl_config.gamma**jnp.arange(batch['rewards'].shape[-1]),axis=-1)
        target=target+(1.-batch['done'].astype(target.dtype))*rl_config.gamma**batch['duration']*jnp.minimum(nq1,nq2)
        target=jax.lax.stop_gradient(target)
        q1,q2=critic.q_values(params,batch['z_rl'],batch['proprio'],batch['action_chunk'])
        valid=batch['td_valid'].astype(q1.dtype);den=jnp.maximum(valid.sum(),1.)
        loss=jnp.sum((q1-target)**2*valid)/den+jnp.sum((q2-target)**2*valid)/den
        return loss,dict(critic_loss=loss,q1_mean=jnp.mean(q1),q2_mean=jnp.mean(q2),target_q_mean=jnp.mean(target))
    (loss,metrics),grads=jax.value_and_grad(loss_fn,has_aux=True)(state.critic_params)
    updates,opt_state=state.critic_tx.update(grads,state.critic_opt_state,state.critic_params)
    return state.replace(critic_params=optax.apply_updates(state.critic_params,updates),critic_opt_state=opt_state,rng=next_rng),metrics


class ExecutedCritic:
 def __init__(self,critic,ref,cfg):self.critic=critic;self.ref=ref;self.cfg=cfg
 def q_values(self,params,z,p,raw):return self.critic.q_values(params,z,p,executed_normalized(raw,p,self.ref,self.cfg))
def update_actor(state,batch,actor,critic,rl_config,**kwargs):
 return upstream.update_actor(state,batch,actor,ExecutedCritic(critic,batch['ref_chunk'],rl_config),rl_config,**kwargs)
_original=upstream.train_step.__wrapped__
_globals=dict(_original.__globals__,update_critic=update_critic,update_actor=update_actor)
_step=types.FunctionType(_original.__code__,_globals,_original.__name__,_original.__defaults__,_original.__closure__)
_step.__kwdefaults__=_original.__kwdefaults__
train_step=functools.partial(jax.jit,static_argnames=('actor','critic','rl_config','use_action_adapter'))(_step)
