"""Symmetric Cobot smoothness components used by R2 actor training."""

from __future__ import annotations

import numpy as np

LEFT = np.arange(0, 6)
RIGHT = np.arange(7, 13)
GRIPPERS = np.asarray([6, 13])


def numpy_smoothness_components(pred_abs_chunk, target_abs_chunk, state0) -> dict[str, float]:
    pred = np.asarray(pred_abs_chunk, dtype=np.float64)
    target = np.asarray(target_abs_chunk, dtype=np.float64)
    state = np.asarray(state0, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 14:
        raise ValueError("predicted and target chunks must share shape [B,T,14]")
    if state.shape != (pred.shape[0], 14):
        raise ValueError("state0 must have shape [B,14]")

    pred_velocity = np.diff(pred, axis=1)
    target_velocity = np.diff(target, axis=1)
    velocity_error = pred_velocity - target_velocity
    pred_accel = np.diff(pred_velocity, axis=1)
    target_accel = np.diff(target_velocity, axis=1)
    first_delta = pred[:, 0] - state

    def mse(value) -> float:
        return float(np.mean(np.square(value))) if np.size(value) else 0.0

    return {
        "left_velocity": mse(velocity_error[..., LEFT]),
        "right_velocity": mse(velocity_error[..., RIGHT]),
        "gripper_velocity": mse(velocity_error[..., GRIPPERS]),
        "state_first": mse(first_delta[..., np.concatenate((LEFT, RIGHT))]),
        "acceleration": mse((pred_accel - target_accel)[..., np.concatenate((LEFT, RIGHT))]),
    }


def jax_smoothness_components(pred_abs_chunk, target_abs_chunk, state0):
    """JAX equivalent; imported lazily so CPU-only manifest tools stay lightweight."""
    import jax.numpy as jnp

    pred = jnp.asarray(pred_abs_chunk)
    target = jnp.asarray(target_abs_chunk)
    state = jnp.asarray(state0)
    joints = jnp.asarray(np.concatenate((LEFT, RIGHT)))
    grippers = jnp.asarray(GRIPPERS)
    velocity_error = (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    accel_error = (pred_velocity[:, 1:] - pred_velocity[:, :-1]) - (
        target_velocity[:, 1:] - target_velocity[:, :-1]
    )
    first_delta = pred[:, 0] - state
    return {
        "left_velocity": jnp.mean(jnp.square(velocity_error[..., :6])),
        "right_velocity": jnp.mean(jnp.square(velocity_error[..., 7:13])),
        "gripper_velocity": jnp.mean(jnp.square(jnp.take(velocity_error, grippers, axis=-1))),
        "state_first": jnp.mean(jnp.square(jnp.take(first_delta, joints, axis=-1))),
        "acceleration": jnp.mean(jnp.square(jnp.take(accel_error, joints, axis=-1))),
    }
