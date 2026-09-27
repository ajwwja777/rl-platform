"""Pure Task2 takeover and action-safety runtime used by the ROS bridge."""

from __future__ import annotations

import numpy as np
from methods.openpi_rlt.cobot_adapter.action_conditioning import ActionConditioner
from methods.openpi_rlt.cobot_adapter.episode_control import (
    EpisodeController,
    EpisodeSnapshot,
)
from methods.openpi_rlt.cobot_adapter.trace import ControlSource, EpisodeOutcome


class Task2PolicyRuntime:
    """Bind Task2's authoritative mode to a fail-closed episode controller."""

    def __init__(
        self,
        *,
        joint_step_limit: float,
        gripper_step_limit: float,
        joint_accel_limit: float | None = None,
        active_arm: str = "both",
        hold_grippers: bool = False,
        command_smoothing_alpha: float = 1.0,
        command_step_limit: float | None = None,
    ) -> None:
        if joint_step_limit <= 0 or gripper_step_limit <= 0:
            raise ValueError("action step limits must be positive")
        self._controller = EpisodeController()
        self._joint_step_limit = float(joint_step_limit)
        self._gripper_step_limit = float(gripper_step_limit)
        optional_conditioning = (joint_accel_limit is not None or active_arm != "both"
                                 or hold_grippers or command_smoothing_alpha != 1.0
                                 or command_step_limit is not None)
        self._conditioner = ActionConditioner(
            active_arm=active_arm,
            hold_grippers=hold_grippers,
            command_smoothing_alpha=command_smoothing_alpha,
            command_step_limit=command_step_limit,
            joint_step_limit=joint_step_limit,
            joint_accel_limit=(joint_step_limit if joint_accel_limit is None else joint_accel_limit),
            gripper_step_limit=gripper_step_limit,
        ) if optional_conditioning else None
        self._last_conditioning_report = None

    def reset_action_conditioning(self) -> None:
        if self._conditioner is not None:
            self._conditioner.reset()

    def snapshot(self) -> EpisodeSnapshot:
        return self._controller.snapshot()

    def arm(self) -> EpisodeSnapshot:
        if self._conditioner is not None:
            self._conditioner.reset()
        return self._controller.arm()

    def install_fresh_plan(self, generation: int) -> EpisodeSnapshot:
        return self._controller.mark_plan_installed(generation)

    def accepts_policy_result(self, generation: int) -> bool:
        return self._controller.accepts_policy_result(generation)

    def finish_episode(self, outcome: EpisodeOutcome) -> EpisodeSnapshot:
        if self._conditioner is not None:
            self._conditioner.reset()
        return self._controller.finish_episode(outcome)

    def mark_robot_reset_complete(self) -> EpisodeSnapshot:
        return self._controller.mark_robot_reset_complete()

    def confirm_object_reset_ready(self) -> EpisodeSnapshot:
        return self._controller.confirm_object_reset_ready()

    def observe_mode(self, mode: str) -> EpisodeSnapshot:
        value = str(mode).strip().lower()
        if value == "policy":
            mask = (False, False)
        elif value.startswith("manual:"):
            sides = value.split(":", 1)[1].split("+")
            if not sides or any(side not in {"left", "right"} for side in sides):
                self._controller.fail(f"unknown Task2 mode: {mode}")
                raise RuntimeError(f"unknown Task2 mode: {mode}")
            mask = ("left" in sides, "right" in sides)
        elif value == "fault":
            return self._controller.fail("Task2 coordinator fault")
        else:
            self._controller.fail(f"unknown Task2 mode: {mode}")
            raise RuntimeError(f"unknown Task2 mode: {mode}")
        snapshot = self._controller.update_takeover(left=mask[0], right=mask[1])
        if any(mask) or snapshot.fresh_plan_required:
            if self._conditioner is not None:
                self._conditioner.reset()
        return snapshot

    def control_source(self, policy_source: ControlSource | int) -> ControlSource:
        source = ControlSource(policy_source)
        takeover_count = sum(self.snapshot().expert_mask)
        if takeover_count == 2:
            return ControlSource.HUMAN
        if takeover_count == 1:
            return ControlSource.MIXED
        if source not in (ControlSource.BASE, ControlSource.RL):
            raise ValueError("policy source must be BASE or RL without takeover")
        return source

    def safe_policy_target(self, requested: object, current: object) -> np.ndarray:
        snapshot = self.snapshot()
        if snapshot.policy_paused:
            raise RuntimeError("policy is paused by Task2")
        if snapshot.fresh_plan_required:
            raise RuntimeError("fresh policy plan has not been installed")
        try:
            target = self._finite_14d("requested", requested)
            measured = self._finite_14d("current", current)
        except ValueError as error:
            self._controller.fail(str(error))
            raise
        if self._conditioner is None:
            # Exact deployed behavior for the faithful default: no EMA/acceleration shaping.
            delta = target - measured
            joints = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
            grippers = np.asarray([6, 13])
            delta[joints] = np.clip(delta[joints], -self._joint_step_limit, self._joint_step_limit)
            delta[grippers] = np.clip(delta[grippers], -self._gripper_step_limit, self._gripper_step_limit)
            return np.asarray(measured + delta, dtype=np.float32)
        output, report = self._conditioner.condition(target, measured)
        self._last_conditioning_report = report
        return output

    def last_conditioning_report(self):
        return self._last_conditioning_report

    @staticmethod
    def _finite_14d(name: str, value: object) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (14,):
            raise ValueError(f"{name} must be one 14-dimensional vector, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array.copy()
