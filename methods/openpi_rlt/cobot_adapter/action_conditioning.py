"""Deterministic evaluation, low-frequency exploration and 14D safety conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

class CommandTrackingConflict(ValueError):
    """No command can satisfy both tracking and command continuity bounds."""


LEFT_JOINTS = np.arange(0, 6)
RIGHT_JOINTS = np.arange(7, 13)
JOINTS = np.concatenate((LEFT_JOINTS, RIGHT_JOINTS))
GRIPPERS = np.asarray([6, 13])


def active_action_mask(active_arm: str, *, explore_gripper: bool = False) -> np.ndarray:
    if active_arm not in {"left", "right", "both"}:
        raise ValueError("active_arm must be left, right, or both")
    mask = np.zeros(14, dtype=np.float32)
    if active_arm in {"left", "both"}:
        mask[LEFT_JOINTS] = 1.0
        if explore_gripper:
            mask[6] = 1.0
    if active_arm in {"right", "both"}:
        mask[RIGHT_JOINTS] = 1.0
        if explore_gripper:
            mask[13] = 1.0
    return mask


def chunk_exploration_noise(
    chunk_len: int,
    *,
    mode: str,
    active_arm: str,
    std: float,
    seed: int,
    explore_gripper: bool = False,
) -> np.ndarray:
    """Generate one smooth, run-reproducible perturbation per action chunk."""
    if chunk_len <= 0 or not np.isfinite(std) or std < 0:
        raise ValueError("chunk_len and std must be valid")
    if mode == "eval":
        return np.zeros((chunk_len, 14), dtype=np.float32)
    if mode != "explore":
        raise ValueError("mode must be eval or explore")
    rng = np.random.default_rng(seed)
    endpoints = rng.normal(0.0, std, size=(2, 14))
    alpha = np.linspace(0.0, 1.0, chunk_len, dtype=np.float64)[:, None]
    noise = endpoints[0][None, :] * (1.0 - alpha) + endpoints[1][None, :] * alpha
    noise *= active_action_mask(active_arm, explore_gripper=explore_gripper)[None, :]
    return np.asarray(noise, dtype=np.float32)


@dataclass(frozen=True)
class ConditioningReport:
    velocity_clipped: int
    acceleration_clipped: int
    passive_dimensions_held: int
    grippers_held: int
    max_requested_delta: float
    max_executed_delta: float


class ActionConditioner:
    """Stateful rate/acceleration limiter with explicit passive-arm semantics."""

    def __init__(
        self,
        *,
        active_arm: str,
        hold_grippers: bool,
        joint_step_limit: float,
        joint_accel_limit: float,
        gripper_step_limit: float,
        command_smoothing_alpha: float = 1.0,
        command_step_limit: float | None = None,
    ) -> None:
        if min(joint_step_limit, joint_accel_limit, gripper_step_limit) <= 0:
            raise ValueError("conditioning limits must be positive")
        if not np.isfinite(command_smoothing_alpha) or not 0 < command_smoothing_alpha <= 1:
            raise ValueError("command smoothing alpha must be in (0, 1]")
        if command_step_limit is not None and (not np.isfinite(command_step_limit) or command_step_limit <= 0):
            raise ValueError("command step limit must be positive")
        self._command_alpha = float(command_smoothing_alpha)
        self._command_step = command_step_limit
        self._previous_command = None
        self._active_mask = active_action_mask(active_arm, explore_gripper=not hold_grippers).astype(bool)
        self._active_joint_mask = active_action_mask(active_arm, explore_gripper=False).astype(bool)
        self._hold_grippers = bool(hold_grippers)
        self._joint_step_limit = float(joint_step_limit)
        self._joint_accel_limit = float(joint_accel_limit)
        self._gripper_step_limit = float(gripper_step_limit)
        self._previous_delta: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_delta = None
        self._previous_command = None

    def condition(self, requested: object, measured: object) -> tuple[np.ndarray, ConditioningReport]:
        target = self._finite_14d("requested", requested)
        state = self._finite_14d("measured", measured)
        raw_delta = target - state
        delta = raw_delta.copy()

        passive = ~self._active_mask
        delta[passive] = 0.0
        if self._hold_grippers:
            delta[GRIPPERS] = 0.0

        before_velocity = delta.copy()
        delta[JOINTS] = np.clip(delta[JOINTS], -self._joint_step_limit, self._joint_step_limit)
        delta[GRIPPERS] = np.clip(
            delta[GRIPPERS], -self._gripper_step_limit, self._gripper_step_limit
        )
        velocity_clipped = int(np.count_nonzero(np.abs(delta - before_velocity) > 1e-9))

        previous = np.zeros(14, dtype=np.float32) if self._previous_delta is None else self._previous_delta
        before_accel = delta.copy()
        lower = previous - self._joint_accel_limit
        upper = previous + self._joint_accel_limit
        delta[self._active_joint_mask] = np.clip(
            delta[self._active_joint_mask],
            lower[self._active_joint_mask],
            upper[self._active_joint_mask],
        )
        acceleration_clipped = int(
            np.count_nonzero(np.abs(delta[self._active_joint_mask] - before_accel[self._active_joint_mask]) > 1e-9)
        )
        output = np.asarray(state + delta, dtype=np.float32)
        if self._command_step is not None:
            anchor = state if self._previous_command is None else self._previous_command
            mask = self._active_joint_mask
            lower = np.maximum(anchor[mask] - self._command_step, state[mask] - self._joint_step_limit)
            upper = np.minimum(anchor[mask] + self._command_step, state[mask] + self._joint_step_limit)
            if np.any(lower > upper):
                self.reset()
                raise CommandTrackingConflict("command/tracking bounds conflict; pause and reanchor required")
            filtered = anchor[mask] + self._command_alpha * (output[mask] - anchor[mask])
            output[mask] = np.clip(filtered, lower, upper)
        self._previous_command = output.copy()
        delta = output - state
        self._previous_delta = delta.copy()
        return output, ConditioningReport(
            velocity_clipped=velocity_clipped,
            acceleration_clipped=acceleration_clipped,
            passive_dimensions_held=int(np.count_nonzero(passive)),
            grippers_held=2 if self._hold_grippers else 0,
            max_requested_delta=float(np.max(np.abs(raw_delta))),
            max_executed_delta=float(np.max(np.abs(delta))),
        )

    @staticmethod
    def _finite_14d(name: str, value: object) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (14,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be one finite 14D vector")
        return array.copy()

