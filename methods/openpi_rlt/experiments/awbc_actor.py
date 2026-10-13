"""Opt-in offline AWBC Actor objective (online-review-20261011). Not used by any runtime entry.

Replaces only the Actor BC target and row weights inside a copy of the guarded
credit train step; the Critic update, RNG use, Q term, optimizer and metric keys
are unchanged. Per control step:
  human / mixed step            -> imitate executed action, weight 1
  policy step, autonomous success episode -> imitate executed action, weight clip(exp(A/beta), 0, max_weight)
  other policy step             -> imitate Stage-1 reference, weight failure_anchor
A = min Q(s, executed) - min Q(s, reference), stop-gradient, current Critic.
Uses batch["actor_batch"] (separate Actor batch) and batch["actor_batch"]["awbc_success"].
"""
from __future__ import annotations
import types


def make_awbc_train_step(mc_weight, *, beta, max_weight, failure_anchor, target_policy="clip"):
    import jax
    import jax.numpy as jnp
    import optax
    from rlt_online_rl.networks import apply_reference_dropout
    from rlt_online_rl.replay import TransitionSource
    from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk
    from .guarded_credit import make_train_step

    if beta <= 0 or max_weight <= 0 or failure_anchor < 0:
        raise ValueError("beta>0, max_weight>0 and failure_anchor>=0 required")
    run = make_train_step(mc_weight, target_policy=target_policy)

    def update_actor(state, batch, actor, critic, rl_config, *, bc_weight, q_weight, delta_weight=0.0,
                     use_action_adapter=False, action_q01=None, action_q99=None):
        b = batch["actor_batch"]
        actor_rng, next_rng = jax.random.split(state.rng)
        human = (b["source_chunk"] == int(TransitionSource.HUMAN)) | (b["source_chunk"] == int(TransitionSource.MIXED))
        human_f = human.astype(jnp.float32)
        succ = b["awbc_success"].astype(jnp.float32)[:, None]
        q_exec = jnp.minimum(*critic.q_values(state.critic_params, b["z_rl"], b["proprio"], b["action_chunk"]))
        q_ref = jnp.minimum(*critic.q_values(state.critic_params, b["z_rl"], b["proprio"], b["ref_chunk"]))
        adv = jax.lax.stop_gradient(q_exec - q_ref)
        w_succ = jnp.clip(jnp.exp(adv / beta), 0.0, max_weight)[:, None]
        imitate = jnp.maximum(human_f, succ)
        weight = human_f + (1.0 - human_f) * (succ * w_succ + (1.0 - succ) * failure_anchor)
        target = jnp.where(imitate[..., None] > 0.5, b["action_chunk"], b["ref_chunk"])

        def loss_fn(actor_params):
            dropout_rng, sample_rng = jax.random.split(actor_rng)
            dropped_ref = apply_reference_dropout(dropout_rng, b["ref_chunk"], rl_config.reference_dropout_prob)
            action = actor.sample_action(actor_params, sample_rng, b["z_rl"], b["proprio"], dropped_ref, deterministic=False)
            q1, _ = critic.q_values(state.critic_params, b["z_rl"], b["proprio"], action)
            err = jnp.mean(jnp.square(action - target), axis=-1)
            bc_penalty = jnp.mean(weight * err)
            policy_f = 1.0 - human_f
            ref_err = jnp.mean(jnp.square(action - b["ref_chunk"]), axis=-1)
            hum_err = jnp.mean(jnp.square(action - b["action_chunk"]), axis=-1)
            bc_ref_penalty = jnp.sum(ref_err * policy_f) / jnp.maximum(jnp.sum(policy_f), 1.0)
            bc_human_penalty = jnp.sum(hum_err * human_f) / jnp.maximum(jnp.sum(human_f), 1.0)
            if use_action_adapter:
                pa = jax_denormalize_to_abs_chunk(action, b["proprio"], action_q01, action_q99, action_representation=rl_config.action_representation)
                ta = jax_denormalize_to_abs_chunk(target, b["proprio"], action_q01, action_q99, action_representation=rl_config.action_representation)
            else:
                pa, ta = action, target
            delta_penalty = jnp.mean(jnp.square((pa[:, 1:, :6] - pa[:, :-1, :6]) - (ta[:, 1:, :6] - ta[:, :-1, :6])))
            actor_q = jnp.mean(q1)
            wb = jnp.asarray(bc_weight, jnp.float32) * bc_penalty
            wq = jnp.asarray(q_weight, jnp.float32) * actor_q
            wd = jnp.asarray(delta_weight, jnp.float32) * delta_penalty
            loss = wb - wq + wd
            return loss, {"actor_loss": loss, "actor_q": actor_q, "bc_penalty": bc_penalty, "bc_ref_penalty": bc_ref_penalty,
                          "bc_human_penalty": bc_human_penalty, "human_mask_ratio": jnp.mean(human_f),
                          "policy_mask_ratio": 1.0 - jnp.mean(human_f), "delta_penalty": delta_penalty,
                          "weighted_bc": wb, "weighted_delta": wd, "weighted_q": wq}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.actor_params)
        updates, opt = state.actor_tx.update(grads, state.actor_opt_state, state.actor_params)
        new = state.replace(actor_params=optax.apply_updates(state.actor_params, updates), actor_opt_state=opt, rng=next_rng)
        return new, {**metrics, "actor_loss": loss}

    original = run.__wrapped__
    nested = [c for c in original.__code__.co_consts if isinstance(c, types.CodeType)]
    if "update_actor" not in original.__globals__ or not any("update_actor" in c.co_names for c in nested):
        raise RuntimeError("Upstream trainer changed; review the AWBC adapter")
    private = types.FunctionType(original.__code__, dict(original.__globals__, update_actor=update_actor),
                                 original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    return jax.jit(private, static_argnames=("actor", "critic", "rl_config", "use_action_adapter"))
