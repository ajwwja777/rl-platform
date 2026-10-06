"""Native networks with opt-in round TD guard and outcome BC (no Q gradient).

All functions are process-local and do not monkeypatch the fixed upstream learner.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from rlt_online_rl import trainer
from rlt_online_rl.networks import apply_reference_dropout, build_td_target
from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk


def guarded_target(actor, critic, state, batch, cfg, key, *, clip=True):
    target = build_td_target(actor, state.target_actor_params, critic, state.target_critic_params,
        batch["next_z_rl"], batch["next_proprio"], batch["next_ref_chunk"],
        batch["rewards"], batch["done"], cfg.gamma, key)
    # Applicable only after roundwise.annotate validates one unit success reward
    # on the terminal logical step, and no other rewards in the complete Episode.
    return jnp.clip(target, 0., 1.) if clip else target


def actor_targets(batch, *, successful_executed=True, failure_anchor=.1):
    human = jnp.isin(batch["source_chunk"], jnp.array([2, 3]))
    success = batch["autonomous_success"][:, None]
    executed = human | (success & successful_executed)
    target = jnp.where(executed[..., None], batch["action_chunk"], batch["ref_chunk"])
    weights = jnp.where(human | success, 1., failure_anchor)
    return target, weights


def make_critic_step(actor, critic, cfg, *, clip=True, mc=False):
    @jax.jit
    def step(state, batch):
        key, rng = jax.random.split(state.rng)
        target = jax.lax.stop_gradient(batch["mc_return"] if mc else
            guarded_target(actor, critic, state, batch, cfg, key, clip=clip))
        def loss(params):
            q1, q2 = critic.q_values(params, batch["z_rl"], batch["proprio"], batch["action_chunk"])
            value = jnp.mean((q1-target)**2 + (q2-target)**2)
            return value, dict(critic_loss=value, q1_mean=q1.mean(), q2_mean=q2.mean(),
                target_mean=target.mean(), raw_q_above_bound=jnp.mean((jnp.stack([q1,q2])>1.).astype(jnp.float32)))
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(state.critic_params)
        updates, opt = state.critic_tx.update(grads, state.critic_opt_state, state.critic_params)
        params = optax.apply_updates(state.critic_params, updates)
        state = state.replace(critic_params=params, critic_opt_state=opt, rng=rng,
            target_critic_params=trainer.soft_update_targets(state.target_critic_params, params, cfg.target_tau),
            global_step=state.global_step+1)
        return state, metrics
    return step


def make_actor_step(actor, cfg, q01, q99, *, successful_executed=True,
                    failure_anchor=.1, delta_weight=1.):
    if not 0 <= failure_anchor <= 1 or delta_weight < 0:
        raise ValueError("invalid Actor weights")
    @jax.jit
    def step(state, batch):
        key, rng = jax.random.split(state.rng)
        dropout, sample = jax.random.split(key)
        reference = apply_reference_dropout(dropout, batch["ref_chunk"], cfg.reference_dropout_prob)
        target, weights = actor_targets(batch, successful_executed=successful_executed,
                                         failure_anchor=failure_anchor)
        def loss(params):
            action = actor.sample_action(params, sample, batch["z_rl"], batch["proprio"], reference, deterministic=False)
            errors = jnp.square(action-target).mean(-1)
            bc = (errors*weights).sum() / jnp.maximum(weights.sum(), 1.)
            absolute = jax_denormalize_to_abs_chunk(action, batch["proprio"], q01, q99,
                action_representation=cfg.action_representation)
            target_absolute = jax_denormalize_to_abs_chunk(target, batch["proprio"], q01, q99,
                action_representation=cfg.action_representation)
            delta_errors = jnp.square(jnp.diff(absolute[...,:6],axis=1)-jnp.diff(target_absolute[...,:6],axis=1)).mean(-1)
            delta_weights = jnp.minimum(weights[:,1:],weights[:,:-1])
            delta = (delta_errors*delta_weights).sum() / jnp.maximum(delta_weights.sum(),1.)
            value = cfg.online_bc_weight*bc + delta_weight*delta
            return value, dict(actor_loss=value, bc_penalty=bc, delta_penalty=delta,
                grip_bc=jnp.mean(jnp.square(action[...,-1]-target[...,-1])),
                autonomous_success_ratio=batch["autonomous_success"].mean(), q_gradient_enabled=jnp.array(0.))
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(state.actor_params)
        updates, opt = state.actor_tx.update(grads, state.actor_opt_state, state.actor_params)
        return state.replace(actor_params=optax.apply_updates(state.actor_params,updates),
            actor_opt_state=opt, rng=rng, actor_version=state.actor_version+1), metrics
    return step
