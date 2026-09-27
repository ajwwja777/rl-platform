"""Advantage-weighted BC actor update for plug_v2 RTC candidates.
Critic update, target updates and step order reuse rtc_upstream_core unchanged.
No Q gradient reaches the actor: Q only decides which data actions to imitate and how hard.
Pinned modules are imported, never mutated.
"""
import functools
import jax,jax.numpy as jnp,optax
from .rtc_upstream_core import update_critic,upstream

AWBC_METRIC_KEYS=('actor_loss','actor_q','bc_penalty','bc_ref_penalty','bc_human_penalty',
    'human_mask_ratio','policy_mask_ratio','actor_q_mask_ratio','delta_penalty','weighted_bc',
    'weighted_delta','weighted_q','awbc_weight_mean','awbc_weight_min','awbc_weight_max',
    'awbc_adv_mean','awbc_data_target_ratio','awbc_success_ratio','awbc_ref_penalty')

def awbc_weights(q_data,q_ref,success,human,*,beta,max_weight):
    """Per-row imitation weight and target selector.
    human rows: imitate the human command, weight 1.
    policy rows of a successful (sub)episode: imitate the executed action, weight clip(exp(adv/beta),0,max_weight),
      adv=Q(s,a_data)-Q(s,a_ref). An uninformative critic (adv~0) degrades to filtered BC on successes.
    policy rows of a failed (sub)episode: retreat to the Stage1 reference, weight 1.
    """
    return awbc_weights_from_advantage(q_data-q_ref,success,human,beta=beta,max_weight=max_weight)

def awbc_weights_from_advantage(adv,success,human,*,beta,max_weight):
    """Same rule as awbc_weights for any advantage estimate (Q-based, or value change V(s')-V(s))."""
    adv=jax.lax.stop_gradient(jnp.asarray(adv,jnp.float32))
    w_adv=jnp.clip(jnp.exp(adv/jnp.asarray(beta,jnp.float32)),0.,jnp.asarray(max_weight,jnp.float32))
    success=jnp.asarray(success).astype(bool);human=jnp.asarray(human).astype(bool)
    use_data=jnp.logical_or(human,success)
    weight=jnp.where(human,1.,jnp.where(success,w_adv,1.)).astype(jnp.float32)
    return use_data,weight,adv

def update_actor_awbc(state,batch,actor,critic,rl_config,*,bc_weight,q_weight,delta_weight=0.,
                      use_action_adapter=False,action_q01=None,action_q99=None,
                      beta=0.1,max_weight=5.,ref_weight=0.1):
    actor_rng,next_rng=jax.random.split(state.rng)
    def loss_fn(actor_params):
        dropout_rng,sample_rng=jax.random.split(actor_rng)
        dropped_ref=upstream.apply_reference_dropout(dropout_rng,batch['ref_chunk'],rl_config.reference_dropout_prob)
        action_chunk=actor.sample_action(actor_params,sample_rng,batch['z_rl'],batch['proprio'],dropped_ref,deterministic=False)
        source_chunk=batch['source_chunk']
        human_step=jnp.logical_or(source_chunk==2,source_chunk==3)          # (B,T)
        human_row=jnp.any(human_step,axis=1)                                 # (B,)
        success_row=jnp.asarray(batch['success']).reshape(-1)>0              # (B,) per-transition label (HIL prefix relabelled upstream)
        if 'awbc_adv' in batch:
            # Precomputed value-change advantage (state_value.advantages); the critic is not consulted.
            use_data,weight,adv=awbc_weights_from_advantage(jnp.asarray(batch['awbc_adv']).reshape(-1),success_row,human_row,beta=beta,max_weight=max_weight)
        else:
            # Critic is a fixed scorer here: no gradient path from Q to actor params.
            qd1,qd2=critic.q_values(state.critic_params,batch['z_rl'],batch['proprio'],batch['action_chunk'])
            qr1,qr2=critic.q_values(state.critic_params,batch['z_rl'],batch['proprio'],batch['ref_chunk'])
            q_data=jax.lax.stop_gradient(jnp.minimum(qd1,qd2));q_ref=jax.lax.stop_gradient(jnp.minimum(qr1,qr2))
            use_data,weight,adv=awbc_weights(q_data,q_ref,success_row,human_row,beta=beta,max_weight=max_weight)
        use_data_step=jnp.logical_or(human_step,use_data[:,None])            # human steps always take the human command
        bc_target=jnp.where(use_data_step[...,None],batch['action_chunk'],batch['ref_chunk'])
        bc_error=jnp.mean(jnp.square(action_chunk-bc_target),axis=-1)        # (B,T)
        ref_error=jnp.mean(jnp.square(action_chunk-batch['ref_chunk']),axis=-1)
        human_error=jnp.mean(jnp.square(action_chunk-batch['action_chunk']),axis=-1)
        row_error=jnp.mean(bc_error,axis=1)
        bc_penalty=jnp.sum(weight*row_error)/jnp.maximum(jnp.sum(weight),1e-6)
        # Small anchor to Stage1 on rows that imitate data, so repeated self-imitation cannot drift unboundedly.
        data_row=use_data.astype(jnp.float32)
        awbc_ref_penalty=jnp.sum(jnp.mean(ref_error,axis=1)*data_row)/jnp.maximum(jnp.sum(data_row),1.)
        human_mask_f=human_step.astype(jnp.float32);policy_mask_f=1.-human_mask_f
        bc_ref_penalty=jnp.sum(ref_error*policy_mask_f)/jnp.maximum(jnp.sum(policy_mask_f),1.)
        bc_human_penalty=jnp.sum(human_error*human_mask_f)/jnp.maximum(jnp.sum(human_mask_f),1.)
        human_mask_ratio=jnp.mean(human_mask_f)
        # Diagnostic only unless q_weight>0: identical masking to the upstream objective.
        q1,_=critic.q_values(state.critic_params,batch['z_rl'],batch['proprio'],action_chunk)
        original_done=batch.get('original_done',jnp.ones_like(batch['done'],dtype=jnp.bool_))
        q_mask=jnp.logical_or(original_done,human_row).astype(jnp.float32)
        actor_q=jnp.sum(q1*q_mask)/jnp.maximum(jnp.sum(q_mask),1.)
        if not use_action_adapter:
            pred_abs_chunk=action_chunk;target_abs_chunk=bc_target
        else:
            pred_abs_chunk=upstream.jax_denormalize_to_abs_chunk(action_chunk,batch['proprio'],action_q01,action_q99,action_representation=rl_config.action_representation)
            target_abs_chunk=upstream.jax_denormalize_to_abs_chunk(bc_target,batch['proprio'],action_q01,action_q99,action_representation=rl_config.action_representation)
        pred_step_delta=pred_abs_chunk[:,1:,:6]-pred_abs_chunk[:,:-1,:6]
        target_step_delta=target_abs_chunk[:,1:,:6]-target_abs_chunk[:,:-1,:6]
        delta_penalty=jnp.mean(jnp.square(pred_step_delta-target_step_delta))
        bc_w=jnp.asarray(bc_weight,jnp.float32)
        weighted_bc=bc_w*(bc_penalty+jnp.asarray(ref_weight,jnp.float32)*awbc_ref_penalty)
        weighted_q=jnp.asarray(q_weight,jnp.float32)*actor_q
        weighted_delta=jnp.asarray(delta_weight,jnp.float32)*delta_penalty
        actor_loss=weighted_bc-weighted_q+weighted_delta
        metrics=dict(actor_loss=actor_loss,actor_q=actor_q,bc_penalty=bc_penalty,bc_ref_penalty=bc_ref_penalty,
            bc_human_penalty=bc_human_penalty,human_mask_ratio=human_mask_ratio,policy_mask_ratio=1.-human_mask_ratio,
            actor_q_mask_ratio=jnp.mean(q_mask),delta_penalty=delta_penalty,weighted_bc=weighted_bc,
            weighted_delta=weighted_delta,weighted_q=weighted_q,awbc_weight_mean=jnp.mean(weight),
            awbc_weight_min=jnp.min(weight),awbc_weight_max=jnp.max(weight),awbc_adv_mean=jnp.mean(adv),
            awbc_data_target_ratio=jnp.mean(data_row),awbc_success_ratio=jnp.mean(success_row.astype(jnp.float32)),
            awbc_ref_penalty=awbc_ref_penalty)
        return actor_loss,{k:jnp.asarray(metrics[k],jnp.float32) for k in AWBC_METRIC_KEYS}
    (loss,metrics),grads=jax.value_and_grad(loss_fn,has_aux=True)(state.actor_params)
    updates,opt_state=state.actor_tx.update(grads,state.actor_opt_state,state.actor_params)
    return state.replace(actor_params=optax.apply_updates(state.actor_params,updates),actor_opt_state=opt_state,rng=next_rng),metrics

@functools.partial(jax.jit,static_argnames=('actor','critic','rl_config','use_action_adapter'))
def train_step_awbc(state,batch,*,actor,critic,rl_config,bc_weight=1.,q_weight=0.,delta_weight=0.,
                    use_action_adapter=False,action_q01=None,action_q99=None,
                    beta=0.1,max_weight=5.,ref_weight=0.1):
    """rtc_upstream_core.train_step order (critic, actor on cadence, target updates) with the AWBC actor objective."""
    state,critic_metrics=update_critic(state,batch,actor,critic,rl_config)
    cadence=((state.global_step+1)%rl_config.actor_update_period)==0
    actor_enabled=(jnp.abs(jnp.asarray(bc_weight))+jnp.abs(jnp.asarray(q_weight)))>0
    should_update_actor=jnp.logical_and(cadence,actor_enabled)
    zero={key:jnp.array(0.,dtype=jnp.float32) for key in AWBC_METRIC_KEYS}
    def actor_update(train_state):
        updated,metrics=update_actor_awbc(train_state,batch,actor,critic,rl_config,bc_weight=bc_weight,q_weight=q_weight,
            delta_weight=delta_weight,use_action_adapter=use_action_adapter,action_q01=action_q01,action_q99=action_q99,
            beta=beta,max_weight=max_weight,ref_weight=ref_weight)
        updated=updated.replace(target_actor_params=upstream.soft_update_targets(updated.target_actor_params,updated.actor_params,rl_config.target_tau),
                                actor_version=updated.actor_version+1)
        return updated,metrics
    state,actor_metrics=jax.lax.cond(should_update_actor,actor_update,lambda train_state:(train_state,zero),state)
    def critic_target_update(train_state):
        return train_state.replace(target_critic_params=upstream.soft_update_targets(train_state.target_critic_params,train_state.critic_params,rl_config.target_tau))
    state=jax.lax.cond(cadence,critic_target_update,lambda train_state:train_state,state)
    state=state.replace(global_step=state.global_step+1)
    metrics={**critic_metrics,**actor_metrics,
        'did_actor_update':should_update_actor.astype(jnp.float32),
        'global_step':state.global_step.astype(jnp.float32),
        'actor_version':state.actor_version.astype(jnp.float32),
        'bc_weight':jnp.asarray(bc_weight,dtype=jnp.float32),
        'q_weight':jnp.asarray(q_weight,dtype=jnp.float32),
        'delta_weight':jnp.asarray(delta_weight,dtype=jnp.float32)}
    return state,metrics
