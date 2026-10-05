"""Upstream-compatible Cobot environment with Task2 HIL semantics."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from methods.openpi_rlt.cobot_adapter.episode_control import EpisodePhase
from methods.openpi_rlt.cobot_adapter.task2_runtime import Task2PolicyRuntime
from methods.openpi_rlt.cobot_adapter.trace import ControlSource, EpisodeOutcome


class CobotOnlineEnv:
    """Execute policy chunks only through Task2's policy topics.

    The injected I/O object owns ROS. Keeping the episode reducer here pure
    makes takeover, replay labeling, and reset sequencing testable without a
    publisher or a robot.
    """

    cobot_task2_contract = True

    def __init__(
        self,
        io: Any,
        *,
        chunk_exec_horizon: int,
        control_frequency_hz: float,
        max_episode_steps: int | None,
        joint_step_limit: float,
        gripper_step_limit: float,
        enable_robot_reset: bool = False,
        auto_next_delay_sec: float | None = None,
        collection_phase: str = "warmup",
        phase_controller: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if chunk_exec_horizon <= 0:
            raise ValueError("chunk horizon must be positive")
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max episode steps must be positive when configured")
        if control_frequency_hz <= 0:
            raise ValueError("control frequency must be positive")
        if collection_phase not in {"warmup", "online"}:
            raise ValueError("collection phase must be warmup or online")
        self._io = io
        self._runtime = Task2PolicyRuntime(
            joint_step_limit=joint_step_limit,
            gripper_step_limit=gripper_step_limit,
        )
        self._chunk_exec_horizon = int(chunk_exec_horizon)
        self._period = 1.0 / float(control_frequency_hz)
        self._max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._enable_robot_reset = bool(enable_robot_reset)
        self._auto_next_delay_sec = auto_next_delay_sec
        self._collection_phase = collection_phase
        self._phase_controller = phase_controller
        self._sleep = sleep
        self._episode_steps = 0
        self._last_outcome: EpisodeOutcome | None = None
        self._shadow_mode = bool(getattr(io, "shadow_mode", False))
        self._execution = None
        from .execution_profiles import selected_profile
        selected = selected_profile()
        if selected is not None:
            if float(control_frequency_hz) != 20. or int(chunk_exec_horizon) != 10:
                raise ValueError("Optional RTC executor requires the existing logical20/chunk10 contract")
            from .async_execution import AsyncExecution
            self._execution = AsyncExecution(self, *selected)


    def current_phase_name(self) -> str:
        return self._collection_phase

    def episode_phase_name(self) -> str:
        return self._runtime.snapshot().phase.value

    def replay_commit_allowed(self) -> bool:
        return (
            not self._shadow_mode
            and self._last_outcome in (EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE)
        )

    def persist_upstream_raw_episode(self) -> bool:
        """Task5 HDF5 plus the compact atomic trace are the Cobot raw sources."""
        return False

    def mark_replay_finalized(self) -> None:
        """Advance the operator UI only after EnvDriver has finished replay work."""
        finalize_trace = getattr(self._io, "finalize_raw_episode", None)
        if finalize_trace is not None and self._last_outcome is not None:
            finalize_trace(self._last_outcome)
        self._io.mark_replay_finalized()

    def reset(self) -> dict[str, Any]:
        phase = self._runtime.snapshot().phase
        if phase is EpisodePhase.DISARMED:
            self._io.wait_armed()
            self._runtime.arm()
            if hasattr(self._io, "wait_episode_ready"):
                self._io.wait_episode_ready()
        elif phase is EpisodePhase.RESETTING_ROBOT:
            if self._enable_robot_reset:
                self._io.request_home()
            self._runtime.mark_robot_reset_complete()
            delay = self._auto_next_delay_sec if self._enable_robot_reset else None
            self._io.wait_object_ready(delay)
            self._runtime.confirm_object_reset_ready()
        else:
            raise RuntimeError(f"cannot reset Cobot episode from phase {phase.value}")
        self._episode_steps = 0
        self._last_outcome = None
        self._io.set_chunk_ready(False)
        if self._phase_controller is not None:
            self._collection_phase = self._phase_controller.begin_episode()
        sample = self._io.sample()
        self._runtime.observe_mode(sample.mode)
        return sample.observation

    def execute_chunk(
        self,
        observation: dict[str, Any] | None = None,
        policy_planner=None,
        *,
        control_hz: float | None = None,
    ):
        if self._execution is not None:
            return self._execution.execute_chunk(observation, policy_planner, control_hz)
        """Execute one chunk using the current upstream EnvDriver contract.

        Newer upstream drivers pass ``control_hz`` and ``policy_planner`` as
        keyword arguments and expect the environment to sample its own first
        observation.  Keep the positional observation form for the existing
        Cobot unit tests and older deployment entry points.
        """
        if policy_planner is None:
            raise ValueError("policy_planner is required")
        period = self._period
        if control_hz is not None:
            if float(control_hz) <= 0:
                raise ValueError("control_hz must be positive")
            period = 1.0 / float(control_hz)
        if observation is None:
            observation = self._io.sample().observation
        inference_started = time.perf_counter()
        plan = policy_planner(observation, 0)
        plan_created_monotonic = time.perf_counter()
        inference_latency = time.perf_counter() - inference_started
        if hasattr(self._io, "report_chunk"):
            self._io.report_chunk(inference_latency, int(plan.actor_param_version))
        chunk_start_features = plan.start_features
        generation = self._runtime.snapshot().generation
        if self._runtime.snapshot().phase is EpisodePhase.ROLLOUT:
            self._runtime.install_fresh_plan(generation)
            self._io.set_chunk_ready(True)
        current_observation = observation
        trace: list[dict[str, Any]] = []
        rewards: list[float] = []
        policy_anchor_offsets: list[int] = []
        policy_anchor_features: list[Any] = []
        action_index = 0
        outcome: EpisodeOutcome | None = None

        while len(trace) < self._chunk_exec_horizon and outcome is None:
            sample = self._io.sample()
            sample_received_monotonic = time.perf_counter()
            publish_started_monotonic = None
            publish_finished_monotonic = None
            ref_action_index = action_index
            before = self._runtime.snapshot()
            after = self._runtime.observe_mode(sample.mode)

            if after.phase is EpisodePhase.FAULT:
                raise RuntimeError(after.fault_reason or "Task2 coordinator fault")
            if before.phase is EpisodePhase.ROLLOUT and after.phase is EpisodePhase.HIL:
                self._io.set_chunk_ready(False)

            if sample.outcome is not None:
                # The operator may decide the outcome seconds after pausing.
                # Label the last executed action, never synthesize a transition
                # from the paused observation at the click time.
                outcome = EpisodeOutcome(sample.outcome)
                if trace:
                    terminal = trace[-1]
                    terminal["reward"] = 1.0 if outcome is EpisodeOutcome.SUCCESS else 0.0
                    terminal["done"] = True
                    terminal["outcome"] = outcome.value
                    rewards[-1] = terminal["reward"]
                    current_observation = terminal["next_observation"]
                self._last_outcome = outcome
                self._runtime.finish_episode(outcome)
                if self._phase_controller is not None:
                    self._phase_controller.finish_episode()
                break

            # Terminal selection pauses policy and submits the outcome in one
            # operator action. Never let the pause gate consume that terminal
            # sample before it is written to the trace and returned upstream.
            if (
                sample.outcome is None
                and bool(getattr(sample, "paused", False))
                and after.phase is EpisodePhase.ROLLOUT
            ):
                self._io.set_chunk_ready(False)
                current_observation = sample.observation
                self._sleep(period)
                continue

            if after.phase is EpisodePhase.HIL:
                if bool(getattr(sample, "paused", False)) and before.phase is EpisodePhase.ROLLOUT:
                    current_observation = sample.observation
                executed = self._state(sample.observation)
                source = self._runtime.control_source(plan.source)
                next_observation = sample.observation
            else:
                if after.fresh_plan_required:
                    current_observation = sample.observation
                    inference_started = time.perf_counter()
                    plan = policy_planner(current_observation, len(trace))
                    plan_created_monotonic = time.perf_counter()
                    inference_latency = time.perf_counter() - inference_started
                    if hasattr(self._io, "report_chunk"):
                        self._io.report_chunk(inference_latency, int(plan.actor_param_version))
                    generation = after.generation
                    self._runtime.install_fresh_plan(generation)
                    self._io.set_chunk_ready(True)
                    policy_anchor_offsets.append(len(trace))
                    policy_anchor_features.append(plan.start_features)
                    action_index = 0
                    ref_action_index = 0
                requested = np.asarray(plan.action_chunk[action_index], dtype=np.float32)
                proposal_action_index = action_index
                executed = self._runtime.safe_policy_target(
                    requested,
                    self._state(sample.observation),
                )
                publish_started_monotonic = time.perf_counter()
                published = self._io.publish_policy_action(executed)
                publish_finished_monotonic = time.perf_counter()
                self._sleep(period)
                sample = self._io.sample()
                sample_received_monotonic = time.perf_counter()
                after_publish = self._runtime.observe_mode(sample.mode)
                if published is False:
                    self._io.set_chunk_ready(False)
                    current_observation = sample.observation
                    if self._shadow_mode:
                        source = self._runtime.control_source(plan.source)
                        action_index = min(action_index + 1, len(plan.action_chunk) - 1)
                    elif after_publish.phase is EpisodePhase.ROLLOUT:
                        # The local pause gate changed between sampling and
                        # publishing. No command reached Task2, so do not add a
                        # fictitious policy action to replay.
                        continue
                    else:
                        executed = self._state(sample.observation)
                next_observation = sample.observation
                source = self._runtime.control_source(plan.source)
                if published is not False:
                    action_index = min(action_index + 1, len(plan.action_chunk) - 1)

            self._episode_steps += 1
            if outcome is not None:
                pass
            elif sample.outcome is not None:
                outcome = EpisodeOutcome(sample.outcome)
            elif (
                self._max_episode_steps is not None
                and self._episode_steps >= self._max_episode_steps
            ):
                self._io.set_chunk_ready(False)
                if hasattr(self._io, "mark_terminal_pending"):
                    self._io.mark_terminal_pending("max_episode_steps")
                if not hasattr(self._io, "wait_terminal_outcome"):
                    raise RuntimeError("operator terminal outcome interface is unavailable")
                outcome = EpisodeOutcome(self._io.wait_terminal_outcome())
            done = outcome is not None
            reward = 1.0 if outcome is EpisodeOutcome.SUCCESS else 0.0
            record = {
                "observation": current_observation,
                "action": np.asarray(executed, dtype=np.float32),
                "ref_action": np.asarray(
                    plan.ref_chunk[min(ref_action_index, len(plan.ref_chunk) - 1)],
                    dtype=np.float32,
                ),
                "reward": reward,
                "next_observation": next_observation,
                "human_controlled": source in (ControlSource.HUMAN, ControlSource.MIXED),
                "source": int(source),
                "actor_param_version": int(plan.actor_param_version),
                "done": done,
                "outcome": None if outcome is None else outcome.value,
                "expert_mask": list(self._runtime.snapshot().expert_mask),
                "timestamp": float(sample.timestamp),
                "shadow": self._shadow_mode,
                "replay_episode_id": getattr(self, "_trace_replay_episode_id", None),
                "collection_phase": self.current_phase_name(),
                "action_semantics": (
                    "measured_joint_feedback" if source in (ControlSource.HUMAN, ControlSource.MIXED)
                    else "shadow_target" if self._shadow_mode else "published_policy_target"
                ),
                "sample_received_monotonic": sample_received_monotonic,
                "command_publish_started_monotonic": publish_started_monotonic,
                "command_publish_finished_monotonic": publish_finished_monotonic,
                "plan_created_monotonic": plan_created_monotonic,
                "plan_inference_started_monotonic": inference_started,
                # A proposal at a plan anchor is not a fresh Actor inference at
                # each HIL observation. Preserve that distinction explicitly.
                "planned_action": np.asarray(plan.action_chunk[min(
                    ref_action_index, len(plan.action_chunk) - 1)], dtype=np.float32),
                "proposal_anchor_state": np.asarray(plan.start_features.proprio, dtype=np.float32),
                "proposal_action_index": ref_action_index,
                "proposal_semantics": "plan_anchor_action; not same-state HIL counterfactual",
            }
            trace.append(record)
            rewards.append(reward)
            self._io.record_raw_step(record)
            current_observation = next_observation
            if done:
                self._last_outcome = outcome
                self._runtime.finish_episode(outcome)
                if self._phase_controller is not None:
                    self._phase_controller.finish_episode()
                break

        sources = {int(step["source"]) for step in trace}
        chunk_source = sources.pop() if len(sources) == 1 else int(ControlSource.MIXED)
        return current_observation, rewards, outcome is not None, {
            "step_trace": trace,
            "success": int(outcome is EpisodeOutcome.SUCCESS),
            "source": chunk_source,
            "intervention_flag": any(step["human_controlled"] for step in trace),
            "chunk_start_features": chunk_start_features,
            "policy_anchor_offsets": policy_anchor_offsets,
            "policy_anchor_features": policy_anchor_features,
            "outcome": None if outcome is None else outcome.value,
            "replay_eligible": self.replay_commit_allowed(),
            "drop_transition": self._shadow_mode or outcome is EpisodeOutcome.ABORTED,
        }

    @staticmethod
    def _state(observation: dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.all(np.isfinite(state)):
            raise ValueError(f"Cobot observation state must be finite 14D, got {state.shape}")
        return state
