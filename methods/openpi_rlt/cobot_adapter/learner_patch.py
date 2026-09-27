"""Project-local replacement for upstream's left-arm-only actor smoothness."""

from __future__ import annotations

import os

from methods.openpi_rlt.cobot_adapter.smoothness_loss import jax_smoothness_components


def install_symmetric_smoothness_patch(trainer_module) -> None:
    """Replace only ``update_actor``; fixed upstream files remain unchanged."""
    if getattr(trainer_module.update_actor, "_cobot_symmetric_smoothness", False):
        return
    jax = trainer_module.jax
    jnp = trainer_module.jnp
    optax = trainer_module.optax

    def update_actor(
        state,
        batch,
        actor,
        critic,
        rl_config,
        *,
        bc_weight,
        q_weight,
        delta_weight=0.0,
        use_action_adapter=False,
        action_q01=None,
        action_q99=None,
    ):
        actor_rng, next_rng = jax.random.split(state.rng)

        def loss_fn(actor_params):
            dropout_rng, sample_rng = jax.random.split(actor_rng)
            dropped_ref = trainer_module.apply_reference_dropout(
                dropout_rng, batch["ref_chunk"], rl_config.reference_dropout_prob
            )
            action_chunk = actor.sample_action(
                actor_params,
                sample_rng,
                batch["z_rl"],
                batch["proprio"],
                dropped_ref,
                deterministic=False,
            )
            q1, _ = critic.q_values(
                state.critic_params, batch["z_rl"], batch["proprio"], action_chunk
            )
            source_chunk = batch["source_chunk"]
            human_mask = jnp.logical_or(
                source_chunk == int(trainer_module.TransitionSource.HUMAN),
                source_chunk == int(trainer_module.TransitionSource.MIXED),
            )
            human_mask_f = human_mask.astype(jnp.float32)
            policy_mask_f = 1.0 - human_mask_f
            bc_target = jnp.where(human_mask[..., None], batch["action_chunk"], batch["ref_chunk"])
            bc_error = jnp.mean(jnp.square(action_chunk - bc_target), axis=-1)
            ref_error = jnp.mean(jnp.square(action_chunk - batch["ref_chunk"]), axis=-1)
            human_error = jnp.mean(jnp.square(action_chunk - batch["action_chunk"]), axis=-1)
            bc_penalty = jnp.mean(bc_error)
            bc_ref_penalty = jnp.sum(ref_error * policy_mask_f) / jnp.maximum(
                jnp.sum(policy_mask_f), 1.0
            )
            bc_human_penalty = jnp.sum(human_error * human_mask_f) / jnp.maximum(
                jnp.sum(human_mask_f), 1.0
            )
            human_mask_ratio = jnp.mean(human_mask_f)
            if use_action_adapter:
                pred_abs_chunk = trainer_module.jax_denormalize_to_abs_chunk(
                    action_chunk,
                    batch["proprio"],
                    action_q01,
                    action_q99,
                    action_representation=rl_config.action_representation,
                )
                target_abs_chunk = trainer_module.jax_denormalize_to_abs_chunk(
                    bc_target,
                    batch["proprio"],
                    action_q01,
                    action_q99,
                    action_representation=rl_config.action_representation,
                )
            else:
                pred_abs_chunk = action_chunk
                target_abs_chunk = bc_target
            smooth = jax_smoothness_components(
                pred_abs_chunk, target_abs_chunk, batch["proprio"]
            )
            delta_penalty = (
                0.5 * (smooth["left_velocity"] + smooth["right_velocity"])
                + float(os.environ.get("COBOT_RLT_FIRST_ACTION_WEIGHT", "0.5"))
                * smooth["state_first"]
                + float(os.environ.get("COBOT_RLT_ACCEL_WEIGHT", "0.25"))
                * smooth["acceleration"]
                + float(os.environ.get("COBOT_RLT_GRIPPER_SMOOTH_WEIGHT", "0.25"))
                * smooth["gripper_velocity"]
            )
            actor_q = jnp.mean(q1)
            weighted_bc = jnp.asarray(bc_weight, dtype=jnp.float32) * bc_penalty
            weighted_q = jnp.asarray(q_weight, dtype=jnp.float32) * actor_q
            weighted_delta = jnp.asarray(delta_weight, dtype=jnp.float32) * delta_penalty
            actor_loss = weighted_bc - weighted_q + weighted_delta
            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_q": actor_q,
                "bc_penalty": bc_penalty,
                "bc_ref_penalty": bc_ref_penalty,
                "bc_human_penalty": bc_human_penalty,
                "human_mask_ratio": human_mask_ratio,
                "policy_mask_ratio": 1.0 - human_mask_ratio,
                "delta_penalty": delta_penalty,
                "weighted_bc": weighted_bc,
                "weighted_delta": weighted_delta,
                "weighted_q": weighted_q,
            }

        (actor_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.actor_params)
        updates, actor_opt_state = state.actor_tx.update(
            grads, state.actor_opt_state, state.actor_params
        )
        actor_params = optax.apply_updates(state.actor_params, updates)
        new_state = state.replace(
            actor_params=actor_params,
            actor_opt_state=actor_opt_state,
            rng=next_rng,
        )
        return new_state, {**metrics, "actor_loss": actor_loss}

    update_actor._cobot_symmetric_smoothness = True
    trainer_module.update_actor = update_actor

