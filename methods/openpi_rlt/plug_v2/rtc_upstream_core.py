"""Pinned RLT core with explicit SMDP clock and censored-transition mask.
Actor update/network/target-update order are the upstream functions, unchanged.
No imported upstream module is mutated. No ROS imports.
"""
import functools,types,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/openpi-rlt/rlt_online_rl/src'))
import jax,jax.numpy as jnp,optax
from rlt_online_rl import trainer as upstream

def rtc_proprio(context):
    """Right action anchor must occupy first seven entries for upstream delta adapter."""
    c=np.asarray(context,np.float32)
    if c.shape[-1]!=99 or not np.isfinite(c).all():raise ValueError('invalid RTC context')
    return np.concatenate((c[...,7:14],c[...,:7],c[...,14:]),axis=-1)

def original_context(proprio):
    p=np.asarray(proprio,np.float32)
    if p.shape[-1]!=99:raise ValueError('invalid RTC proprio')
    return np.concatenate((p[...,7:14],p[...,:7],p[...,14:]),axis=-1)

def update_critic(state,batch,actor,critic,rl_config):
    critic_rng,next_rng=jax.random.split(state.rng)
    def loss_fn(params):
        next_action=actor.sample_action(state.target_actor_params,critic_rng,
            batch['next_z_rl'],batch['next_proprio'],batch['next_ref_chunk'],deterministic=False)
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

def update_actor(state,batch,actor,critic,rl_config,*,bc_weight,q_weight,delta_weight=0.,
                 use_action_adapter=False,action_q01=None,action_q99=None):
    actor_rng,next_rng=jax.random.split(state.rng)
    def loss_fn(actor_params):
        dropout_rng,sample_rng=jax.random.split(actor_rng)
        dropped_ref=upstream.apply_reference_dropout(dropout_rng,batch['ref_chunk'],rl_config.reference_dropout_prob)
        action_chunk=actor.sample_action(actor_params,sample_rng,batch['z_rl'],batch['proprio'],dropped_ref,deterministic=False)
        q1,_=critic.q_values(state.critic_params,batch['z_rl'],batch['proprio'],action_chunk)
        source_chunk=batch['source_chunk']
        human_mask=jnp.logical_or(source_chunk==2,source_chunk==3)
        human_mask_f=human_mask.astype(jnp.float32);policy_mask_f=1.-human_mask_f
        bc_target=jnp.where(human_mask[...,None],batch['action_chunk'],batch['ref_chunk'])
        bc_error=jnp.mean(jnp.square(action_chunk-bc_target),axis=-1)
        ref_error=jnp.mean(jnp.square(action_chunk-batch['ref_chunk']),axis=-1)
        human_error=jnp.mean(jnp.square(action_chunk-batch['action_chunk']),axis=-1)
        bc_penalty=jnp.mean(bc_error)
        bc_ref_penalty=jnp.sum(ref_error*policy_mask_f)/jnp.maximum(jnp.sum(policy_mask_f),1.)
        bc_human_penalty=jnp.sum(human_error*human_mask_f)/jnp.maximum(jnp.sum(human_mask_f),1.)
        human_mask_ratio=jnp.mean(human_mask_f)
        original_done=batch.get('original_done',jnp.ones_like(batch['done'],dtype=jnp.bool_))
        q_mask=jnp.logical_or(original_done,jnp.any(human_mask,axis=1)).astype(jnp.float32)
        actor_q=jnp.sum(q1*q_mask)/jnp.maximum(jnp.sum(q_mask),1.)
        if not use_action_adapter:
            pred_abs_chunk=action_chunk;target_abs_chunk=bc_target
        else:
            pred_abs_chunk=upstream.jax_denormalize_to_abs_chunk(
                action_chunk,batch['proprio'],action_q01,action_q99,
                action_representation=rl_config.action_representation)
            target_abs_chunk=upstream.jax_denormalize_to_abs_chunk(
                bc_target,batch['proprio'],action_q01,action_q99,
                action_representation=rl_config.action_representation)
        pred_step_delta=pred_abs_chunk[:,1:,:6]-pred_abs_chunk[:,:-1,:6]
        target_step_delta=target_abs_chunk[:,1:,:6]-target_abs_chunk[:,:-1,:6]
        delta_penalty=jnp.mean(jnp.square(pred_step_delta-target_step_delta))
        weighted_bc=jnp.asarray(bc_weight,jnp.float32)*bc_penalty
        weighted_q=jnp.asarray(q_weight,jnp.float32)*actor_q
        weighted_delta=jnp.asarray(delta_weight,jnp.float32)*delta_penalty
        actor_loss=weighted_bc-weighted_q+weighted_delta
        return actor_loss,dict(actor_loss=actor_loss,actor_q=actor_q,bc_penalty=bc_penalty,
            bc_ref_penalty=bc_ref_penalty,bc_human_penalty=bc_human_penalty,
            human_mask_ratio=human_mask_ratio,policy_mask_ratio=1.-human_mask_ratio,
            actor_q_mask_ratio=jnp.mean(q_mask),delta_penalty=delta_penalty,
            weighted_bc=weighted_bc,weighted_delta=weighted_delta,weighted_q=weighted_q)
    (loss,metrics),grads=jax.value_and_grad(loss_fn,has_aux=True)(state.actor_params)
    updates,opt_state=state.actor_tx.update(grads,state.actor_opt_state,state.actor_params)
    return state.replace(actor_params=optax.apply_updates(state.actor_params,updates),
                         actor_opt_state=opt_state,rng=next_rng),metrics

@functools.partial(jax.jit,static_argnames=('actor','critic','rl_config','use_action_adapter'))
def train_step(state,batch,*,actor,critic,rl_config,bc_weight=1.,q_weight=1.,delta_weight=0.,
               use_action_adapter=False,action_q01=None,action_q99=None):
    """Upstream update order with a true actor-frozen critic burn-in."""
    state,critic_metrics=update_critic(state,batch,actor,critic,rl_config)
    cadence=((state.global_step+1)%rl_config.actor_update_period)==0
    actor_enabled=(jnp.abs(jnp.asarray(bc_weight))+jnp.abs(jnp.asarray(q_weight)))>0
    should_update_actor=jnp.logical_and(cadence,actor_enabled)
    zero={key:jnp.array(0.,dtype=jnp.float32) for key in (
        'actor_loss','actor_q','bc_penalty','bc_ref_penalty','bc_human_penalty',
        'human_mask_ratio','policy_mask_ratio','actor_q_mask_ratio','delta_penalty','weighted_bc',
        'weighted_delta','weighted_q')}

    def actor_update(train_state):
        updated,metrics=update_actor(
            train_state,batch,actor,critic,rl_config,bc_weight=bc_weight,
            q_weight=q_weight,delta_weight=delta_weight,
            use_action_adapter=use_action_adapter,action_q01=action_q01,action_q99=action_q99)
        updated=updated.replace(
            target_actor_params=upstream.soft_update_targets(
                updated.target_actor_params,updated.actor_params,rl_config.target_tau),
            actor_version=updated.actor_version+1)
        return updated,metrics

    state,actor_metrics=jax.lax.cond(
        should_update_actor,actor_update,lambda train_state:(train_state,zero),state)

    def critic_target_update(train_state):
        return train_state.replace(target_critic_params=upstream.soft_update_targets(
            train_state.target_critic_params,train_state.critic_params,rl_config.target_tau))
    state=jax.lax.cond(cadence,critic_target_update,lambda train_state:train_state,state)
    state=state.replace(global_step=state.global_step+1)
    metrics={
        **critic_metrics,**actor_metrics,
        'did_actor_update':should_update_actor.astype(jnp.float32),
        'global_step':state.global_step.astype(jnp.float32),
        'actor_version':state.actor_version.astype(jnp.float32),
        'bc_weight':jnp.asarray(bc_weight,dtype=jnp.float32),
        'q_weight':jnp.asarray(q_weight,dtype=jnp.float32),
        'delta_weight':jnp.asarray(delta_weight,dtype=jnp.float32),
    }
    return state,metrics

