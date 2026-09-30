"""Fail-closed operator session state for Cobot RLT rollouts."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


class SessionPhase(str, Enum):
    DISARMED = "disarmed"
    READY = "ready"
    RECORDING_STARTING = "recording_starting"
    ROLLOUT = "rollout"
    HIL = "hil"
    PAUSED = "paused"
    TERMINAL_PENDING = "terminal_pending"
    FINALIZING = "finalizing"
    REPLAY_COMMITTING = "replay_committing"
    WAITING_SCENE = "waiting_scene"
    FAULT = "fault"
    STOPPED = "stopped"


class SessionConflict(RuntimeError):
    """A stale or invalid operator transition was rejected."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class SessionSnapshot:
    phase: SessionPhase
    session_id: str | None
    episode_id: int
    generation: int
    task5_episode_uuid: str | None
    expert_mask: tuple[bool, bool]
    fresh_plan_required: bool
    outcome: EpisodeOutcome | None
    terminal_reason: str | None
    fault_reason: str | None

    @property
    def policy_paused(self) -> bool:
        return self.phase not in (SessionPhase.ROLLOUT, SessionPhase.HIL)

    @property
    def replay_eligible(self) -> bool:
        return (
            self.phase in (SessionPhase.REPLAY_COMMITTING, SessionPhase.WAITING_SCENE)
            and self.outcome in (EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE)
        )


class RltSessionController:
    """Serialize browser, Task2 and recorder events into one episode lifecycle."""

    def __init__(self, *, session_id_factory: Callable[[], str] | None = None) -> None:
        self._lock = threading.RLock()
        self._session_id_factory = session_id_factory or (lambda: str(uuid4()))
        self._phase = SessionPhase.DISARMED
        self._session_id: str | None = None
        self._episode_id = -1
        self._generation = 0
        self._task5_episode_uuid: str | None = None
        self._expert_mask = (False, False)
        self._fresh_plan_required = False
        self._outcome: EpisodeOutcome | None = None
        self._terminal_reason: str | None = None
        self._fault_reason: str | None = None
        self._hil_returns_to_paused = False

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                phase=self._phase,
                session_id=self._session_id,
                episode_id=self._episode_id,
                generation=self._generation,
                task5_episode_uuid=self._task5_episode_uuid,
                expert_mask=self._expert_mask,
                fresh_plan_required=self._fresh_plan_required,
                outcome=self._outcome,
                terminal_reason=self._terminal_reason,
                fault_reason=self._fault_reason,
            )

    def _require_generation(self, expected: int) -> None:
        if int(expected) != self._generation:
            raise SessionConflict(
                "stale_generation",
                f"expected generation {expected}, current {self._generation}",
            )

    def _require_episode(self, expected: int) -> None:
        if int(expected) != self._episode_id:
            raise SessionConflict(
                "stale_episode",
                f"expected episode {expected}, current {self._episode_id}",
            )

    def _require_phase(self, *allowed: SessionPhase) -> None:
        if self._phase not in allowed:
            names = ",".join(phase.value for phase in allowed)
            raise SessionConflict(
                "invalid_phase",
                f"phase {self._phase.value} not in {names}",
            )

    def arm_operator(self) -> SessionSnapshot:
        with self._lock:
            self._require_phase(SessionPhase.DISARMED, SessionPhase.STOPPED)
            self._outcome = None
            self._fault_reason = None
            self._terminal_reason = None
            self._task5_episode_uuid = None
            self._phase = SessionPhase.READY
            self._generation += 1
            return self.snapshot()

    def start_session(self, *, expected_generation: int) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_phase(SessionPhase.READY)
            self._session_id = self._session_id_factory()
            self._episode_id = 0
            self._phase = SessionPhase.RECORDING_STARTING
            self._generation += 1
            return self.snapshot()

    def recording_ready(
        self,
        *,
        expected_generation: int,
        task5_episode_uuid: str | None = None,
    ) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_phase(SessionPhase.RECORDING_STARTING)
            self._task5_episode_uuid = (
                None if task5_episode_uuid is None else str(task5_episode_uuid)
            )
            self._phase = SessionPhase.ROLLOUT
            self._hil_returns_to_paused = False
            self._fresh_plan_required = True
            self._generation += 1
            return self.snapshot()

    def pause_session(self, *, expected_generation: int) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_phase(SessionPhase.ROLLOUT, SessionPhase.HIL)
            self._hil_returns_to_paused = self._phase is SessionPhase.HIL
            self._phase = SessionPhase.PAUSED
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._generation += 1
            return self.snapshot()

    def resume_session(self, *, expected_generation: int) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_phase(SessionPhase.PAUSED)
            self._phase = SessionPhase.ROLLOUT
            self._hil_returns_to_paused = False
            self._fresh_plan_required = True
            self._generation += 1
            return self.snapshot()

    def update_takeover(self, *, left: bool, right: bool) -> SessionSnapshot:
        with self._lock:
            mask = (bool(left), bool(right))
            if self._phase in (SessionPhase.ROLLOUT, SessionPhase.PAUSED) and any(mask):
                self._hil_returns_to_paused = self._phase is SessionPhase.PAUSED
                self._phase = SessionPhase.HIL
                self._expert_mask = mask
                self._fresh_plan_required = False
                self._generation += 1
            elif self._phase is SessionPhase.HIL and not any(mask):
                self._phase = (
                    SessionPhase.PAUSED
                    if self._hil_returns_to_paused
                    else SessionPhase.ROLLOUT
                )
                self._expert_mask = (False, False)
                self._fresh_plan_required = not self._hil_returns_to_paused
                self._hil_returns_to_paused = False
                self._generation += 1
            elif self._phase is SessionPhase.HIL:
                self._expert_mask = mask
            return self.snapshot()

    def mark_terminal_pending(
        self,
        *,
        reason: str,
        expected_episode_id: int,
        expected_generation: int,
    ) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_episode(expected_episode_id)
            self._require_phase(SessionPhase.ROLLOUT, SessionPhase.HIL)
            self._phase = SessionPhase.TERMINAL_PENDING
            self._terminal_reason = str(reason)
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._generation += 1
            return self.snapshot()

    def request_terminal(
        self,
        outcome: EpisodeOutcome,
        *,
        expected_episode_id: int,
        expected_generation: int,
    ) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_episode(expected_episode_id)
            self._require_phase(
                SessionPhase.ROLLOUT,
                SessionPhase.HIL,
                SessionPhase.PAUSED,
                SessionPhase.TERMINAL_PENDING,
            )
            terminal = EpisodeOutcome(outcome)
            if terminal not in (
                EpisodeOutcome.SUCCESS,
                EpisodeOutcome.FAILURE,
                EpisodeOutcome.ABORTED,
                EpisodeOutcome.SAVED,
            ):
                raise ValueError("operator terminal must be success, failure, or aborted")
            self._phase = SessionPhase.FINALIZING
            self._hil_returns_to_paused = False
            self._outcome = terminal
            self._terminal_reason = "operator"
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._generation += 1
            return self.snapshot()

    def episode_finalized(
        self,
        *,
        expected_episode_id: int,
        task5_episode_uuid: str | None = None,
    ) -> SessionSnapshot:
        with self._lock:
            self._require_episode(expected_episode_id)
            self._require_phase(SessionPhase.FINALIZING)
            if task5_episode_uuid is not None:
                self._task5_episode_uuid = str(task5_episode_uuid)
            self._phase = SessionPhase.REPLAY_COMMITTING
            self._generation += 1
            return self.snapshot()

    def replay_finalized(self, *, expected_episode_id: int) -> SessionSnapshot:
        with self._lock:
            self._require_episode(expected_episode_id)
            self._require_phase(SessionPhase.REPLAY_COMMITTING)
            self._phase = SessionPhase.WAITING_SCENE
            self._generation += 1
            return self.snapshot()

    def start_next_episode(
        self,
        *,
        expected_episode_id: int,
        expected_generation: int,
    ) -> SessionSnapshot:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_episode(expected_episode_id)
            self._require_phase(SessionPhase.WAITING_SCENE)
            self._episode_id += 1
            self._task5_episode_uuid = None
            self._outcome = None
            self._terminal_reason = None
            self._phase = SessionPhase.RECORDING_STARTING
            self._hil_returns_to_paused = False
            self._generation += 1
            return self.snapshot()

    def begin_defer(self, *, expected_episode_id: int, expected_generation: int) -> SessionSnapshot:
        with self._lock:
            self._require_episode(expected_episode_id)
            self._require_generation(expected_generation)
            self._require_phase(SessionPhase.ROLLOUT, SessionPhase.HIL, SessionPhase.PAUSED,
                                SessionPhase.TERMINAL_PENDING, SessionPhase.FAULT)
            if self._phase is SessionPhase.FAULT and not str(self._fault_reason or '').startswith('task5_'):
                raise SessionConflict('non_recorder_fault', 'inspect the control/runtime fault first')
            self._phase = SessionPhase.TERMINAL_PENDING
            self._terminal_reason = 'operator_defer_pending'
            self._expert_mask = (False, False)
            self._generation += 1
            return self.snapshot()

    def skip_episode(self, *, expected_episode_id: int, expected_generation: int) -> SessionSnapshot:
        with self._lock:
            self._require_episode(expected_episode_id)
            self._require_generation(expected_generation)
            self._require_phase(SessionPhase.ROLLOUT, SessionPhase.HIL, SessionPhase.PAUSED,
                                SessionPhase.TERMINAL_PENDING, SessionPhase.FAULT)
            if self._phase is SessionPhase.FAULT and not str(self._fault_reason or '').startswith('task5_'):
                raise SessionConflict('non_recorder_fault', 'inspect the control/runtime fault first')
            self._phase = SessionPhase.WAITING_SCENE
            self._outcome = EpisodeOutcome.SAVED
            self._terminal_reason = 'operator_deferred'
            self._fault_reason = None
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._generation += 1
            return self.snapshot()

    def fail(self, reason: str) -> SessionSnapshot:
        with self._lock:
            if self._phase is SessionPhase.STOPPED:
                raise SessionConflict("invalid_phase", "stopped session cannot fault")
            self._phase = SessionPhase.FAULT
            self._fault_reason = str(reason)
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._generation += 1
            return self.snapshot()

    def stop(self) -> SessionSnapshot:
        with self._lock:
            self._phase = SessionPhase.STOPPED
            self._hil_returns_to_paused = False
            self._expert_mask = (False, False)
            self._fresh_plan_required = False
            self._generation += 1
            return self.snapshot()
