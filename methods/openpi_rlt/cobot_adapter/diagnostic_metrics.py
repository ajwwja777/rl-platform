"""Optional critic diagnostics; preserve the native loss, gradients and RNG.

Rewarded chunks are rows containing a positive recorded reward, not successful
Episodes or autonomous successes. Q1 is the Actor objective; min-Q is TD-only.
"""
from __future__ import annotations


def install_diagnostic_metrics_patch(trainer_module):
    original = trainer_module.compute_critic_loss
    if getattr(original, "_cobot_diagnostic_metrics", False):
        return
    import jax.numpy as jnp
    from rlt_online_rl.networks import build_td_target, compute_critic_loss
    if original is not compute_critic_loss:
        raise ValueError("Diagnostic target metrics require the native TD loss")

    def diagnostic_loss(critic, critic_params, actor, target_actor_params,
                        target_critic_params, z_rl, proprio, action_chunk,
                        rewards, done, next_z_rl, next_proprio, next_ref_chunk,
                        gamma, rng):
        loss, metrics = original(
            critic, critic_params, actor, target_actor_params,
            target_critic_params, z_rl, proprio, action_chunk, rewards, done,
            next_z_rl, next_proprio, next_ref_chunk, gamma, rng)
        q1, q2 = critic.q_values(critic_params, z_rl, proprio, action_chunk)
        target = build_td_target(
            actor, target_actor_params, critic, target_critic_params,
            next_z_rl, next_proprio, next_ref_chunk, rewards, done, gamma, rng)
        rewarded = jnp.any(rewards > 0, axis=-1)

        def masked_mean(values, mask):
            count = jnp.sum(mask)
            return jnp.where(count > 0, jnp.sum(jnp.where(mask, values, 0)) /
                             jnp.maximum(count, 1), jnp.nan)

        return loss, {**metrics,
            "diagnostic_metrics_schema": jnp.asarray(1),
            "q1_max": jnp.max(q1), "q2_max": jnp.max(q2),
            "q1_rewarded_chunk_mean": masked_mean(q1, rewarded),
            "q1_unrewarded_chunk_mean": masked_mean(q1, ~rewarded),
            "rewarded_chunk_count": jnp.sum(rewarded),
            "unrewarded_chunk_count": jnp.sum(~rewarded),
            "done_chunk_count": jnp.sum(done.astype(bool)),
            "td_target_max": jnp.max(target), "td_target_min": jnp.min(target)}

    diagnostic_loss._cobot_diagnostic_metrics = True
    trainer_module.compute_critic_loss = diagnostic_loss
