"""Pure episode and Task2 takeover state for Cobot online RL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading

from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


class EpisodePhase(str, Enum):
    DISARMED = "disarmed"
    ROLLOUT = "rollout"
    HIL = "hil"
    RESETTING_ROBOT = "resetting_robot"
    WAITING_OBJECT_RESET = "waiting_object_reset"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True)
class EpisodeSnapshot:
    phase: EpisodePhase
    episode_id: int
    generation: int
    expert_mask: tuple[bool, bool]
    fresh_plan_required: bool
    outcome: EpisodeOutcome | None
    fault_reason: str | None

    @property
    def policy_paused(self) -> bool:
        return self.phase is not EpisodePhase.ROLLOUT

    @property
    def raw_trace_enabled(self) -> bool:
        return self.phase in (EpisodePhase.ROLLOUT, EpisodePhase.HIL)

    @property
    def replay_commit_ready(self) -> bool:
        return self.phase is EpisodePhase.RESETTING_ROBOT and self.outcome is not None


class EpisodeController:
    """Fail-closed reducer shared by ROS callbacks and offline tests.

    Task5 recording is deliberately absent from this state: starting the RL
    deployment arms its own raw trace. The Task2 rear teach levels are the HIL
    source of truth, while a separate object-ready signal starts the next
    episode after reset without another policy-start click.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = EpisodePhase.DISARMED
        self._episode_id = -1
        self._generation = 0
        self._expert_mask = (False, False)
        self._fresh_plan_required = False
        self._outcome: EpisodeOutcome | None = None
        self._fault_reason: str | None = None

    def snapshot(self) -> EpisodeSnapshot:
        with self._lock:
            return EpisodeSnapshot(
                phase=self._phase,
                episode_id=self._episode_id,
                generation=self._generation,
                expert_mask=self._expert_mask,
                fresh_plan_required=self._fresh_plan_required,
                outcome=self._outcome,
                fault_reason=self._fault_reason,
            )

    def arm(self) -> EpisodeSnapshot:
        with self._lock:
            if self._phase is not EpisodePhase.DISARMED:
                raise RuntimeError("episode controller can only arm from disarmed")
            self._phase = EpisodePhase.ROLLOUT
            self._episode_id = 0
            self._generation += 1
            self._fresh_plan_required = True
            return self.snapshot()

    def update_takeover(self, *, left: bool, right: bool) -> EpisodeSnapshot:
        with self._lock:
            requested_mask = (bool(left), bool(right))
            any_active = any(requested_mask)
            if self._phase is EpisodePhase.ROLLOUT and any_active:
                self._phase = EpisodePhase.HIL
                self._generation += 1
                self._fresh_plan_required = False
            elif self._phase is EpisodePhase.HIL and not any_active:
                self._phase = EpisodePhase.ROLLOUT
                self._generation += 1
                self._fresh_plan_required = True
            elif self._phase not in (EpisodePhase.ROLLOUT, EpisodePhase.HIL):
                requested_mask = (False, False)
            self._expert_mask = requested_mask
            return self.snapshot()

    def mark_plan_installed(self, generation: int) -> EpisodeSnapshot:
        with self._lock:
            if not self.accepts_policy_result(generation):
                raise RuntimeError("cannot install a policy result outside the active rollout generation")
            self._fresh_plan_required = False
            return self.snapshot()

    def accepts_policy_result(self, generation: int) -> bool:
        with self._lock:
            return self._phase is EpisodePhase.ROLLOUT and int(generation) == self._generation

    def finish_episode(self, outcome: EpisodeOutcome) -> EpisodeSnapshot:
        with self._lock:
            if outcome is EpisodeOutcome.PENDING:
                raise ValueError("pending episode cannot be committed to replay")
            if self._phase not in (EpisodePhase.ROLLOUT, EpisodePhase.HIL):
                raise RuntimeError("episode can only finish during rollout or hil")
            self._phase = EpisodePhase.RESETTING_ROBOT
            self._generation += 1
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._outcome = outcome
            return self.snapshot()

    def mark_robot_reset_complete(self) -> EpisodeSnapshot:
        with self._lock:
            if self._phase is not EpisodePhase.RESETTING_ROBOT:
                raise RuntimeError("robot reset completion requires phase resetting_robot")
            self._phase = EpisodePhase.WAITING_OBJECT_RESET
            return self.snapshot()

    def confirm_object_reset_ready(self) -> EpisodeSnapshot:
        with self._lock:
            if self._phase is not EpisodePhase.WAITING_OBJECT_RESET:
                raise RuntimeError("object-ready signal requires phase waiting_object_reset")
            self._phase = EpisodePhase.ROLLOUT
            self._episode_id += 1
            self._generation += 1
            self._fresh_plan_required = True
            self._outcome = None
            return self.snapshot()

    def fail(self, reason: str) -> EpisodeSnapshot:
        with self._lock:
            self._phase = EpisodePhase.FAULT
            self._generation += 1
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._fault_reason = str(reason)
            return self.snapshot()

    def stop(self) -> EpisodeSnapshot:
        with self._lock:
            self._phase = EpisodePhase.STOPPED
            self._generation += 1
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            return self.snapshot()
