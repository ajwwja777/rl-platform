"""Single-right-arm Cobot adapter for the upstream 7D yyshadow online runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import CobotIOSample, RosTask2IO
from methods.openpi_rlt.cobot_adapter.task2_runtime import Task2PolicyRuntime
from methods.openpi_rlt.cobot_adapter.trace import ControlSource


class RightArmPolicyRuntime(Task2PolicyRuntime):
    """Keep the existing takeover reducer while enforcing one 7D command."""

    def control_source(self, policy_source: ControlSource | int) -> ControlSource:
        source = ControlSource(policy_source)
        # Only the right rear arm is an expert for this single-right-arm task.
        if self.snapshot().expert_mask[1]:
            return ControlSource.HUMAN
        if source not in (ControlSource.BASE, ControlSource.RL):
            raise ValueError("policy source must be BASE or RL without right-arm takeover")
        return source

    def safe_policy_target(self, requested: object, current: object) -> np.ndarray:
        snapshot = self.snapshot()
        if snapshot.policy_paused:
            raise RuntimeError("policy is paused by Task2")
        if snapshot.fresh_plan_required:
            raise RuntimeError("fresh policy plan has not been installed")
        try:
            target = self._finite_7d("requested", requested)
            measured = self._finite_7d("current", current)
        except ValueError as error:
            self._controller.fail(str(error))
            raise
        delta = target - measured
        delta[:6] = np.clip(delta[:6], -self._joint_step_limit, self._joint_step_limit)
        delta[6] = np.clip(delta[6], -self._gripper_step_limit, self._gripper_step_limit)
        return np.asarray(measured + delta, dtype=np.float32)

    @staticmethod
    def _finite_7d(name: str, value: object) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be one finite 7D right-arm vector, got {array.shape}")
        return array.copy()


class RightArmRosTask2IO(RosTask2IO):
    """Slice feedback to the right arm and publish no left-arm policy command."""

    def sample(self) -> CobotIOSample:
        sample = super().sample()
        observation = dict(sample.observation)
        full_state = np.asarray(observation["state"], dtype=np.float32)
        if full_state.shape != (14,):
            raise ValueError(f"Task2 bridge returned invalid state {full_state.shape}")
        observation["state"] = full_state[7:].copy()
        return CobotIOSample(
            observation=observation,
            mode=sample.mode,
            outcome=sample.outcome,
            paused=sample.paused,
            timestamp=sample.timestamp,
        )

    def publish_policy_action(self, action: object) -> bool:
        target = np.asarray(action, dtype=np.float32)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("plug_v3 right-arm policy action must be finite 7D")
        with self._condition:
            if self._paused or self._mode != "policy" or self._shadow_mode:
                return False
            self._publishers["right"].publish(self._joint_message(target))
            return True


class RightArmCobotOnlineEnv(CobotOnlineEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime = RightArmPolicyRuntime(
            joint_step_limit=float(kwargs["joint_step_limit"]),
            gripper_step_limit=float(kwargs["gripper_step_limit"]),
        )

    @staticmethod
    def _state(observation: dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError(f"plug_v3 observation state must be finite right-arm 7D, got {state.shape}")
        return state


def create_right_arm_online_env() -> RightArmCobotOnlineEnv:
    """Reuse the audited Task2/Task5 factory with only its dimensional boundary changed."""
    # ``run_online_rl`` uses multiprocessing with the spawn start method.  A
    # spawned EnvDriver does not inherit the monkey patches installed by the
    # parent ``online_role`` process, so install them again in the child before
    # constructing the environment.  This supplies Cobot's replay gate,
    # Task5-backed raw source, and post-replay Session completion hook.
    from methods.openpi_rlt.cobot_adapter.online_runtime import (
        install_bimanual_runtime_patch,
    )
    from methods.openpi_rlt.cobot_adapter import cobot_ros1

    install_bimanual_runtime_patch()
    original_io = cobot_ros1.RosTask2IO
    original_env = cobot_ros1.CobotOnlineEnv
    cobot_ros1.RosTask2IO = RightArmRosTask2IO
    cobot_ros1.CobotOnlineEnv = RightArmCobotOnlineEnv
    try:
        return cobot_ros1.create_cobot_online_env()
    finally:
        cobot_ros1.RosTask2IO = original_io
        cobot_ros1.CobotOnlineEnv = original_env
