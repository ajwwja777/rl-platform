from __future__ import annotations

import pytest


def test_session_runs_recorded_episode_through_hil_terminal_and_next() -> None:
    """Catches HIL or terminal finalization breaking the multi-episode session."""
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionPhase
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(
        expected_generation=starting.generation,
        task5_episode_uuid="uuid-1",
    )

    assert rollout.phase is SessionPhase.ROLLOUT
    assert rollout.episode_id == 0
    assert rollout.policy_paused is False
    hil = controller.update_takeover(left=True, right=False)
    assert hil.phase is SessionPhase.HIL
    assert hil.expert_mask == (True, False)
    resumed = controller.update_takeover(left=False, right=False)
    assert resumed.phase is SessionPhase.ROLLOUT
    assert resumed.fresh_plan_required is True

    finalizing = controller.request_terminal(
        EpisodeOutcome.SUCCESS,
        expected_episode_id=0,
        expected_generation=resumed.generation,
    )
    assert finalizing.phase is SessionPhase.FINALIZING
    assert finalizing.policy_paused is True
    assert finalizing.replay_eligible is False
    committing = controller.episode_finalized(expected_episode_id=0)
    assert committing.phase is SessionPhase.REPLAY_COMMITTING
    assert committing.replay_eligible is True
    waiting = controller.replay_finalized(expected_episode_id=0)
    assert waiting.phase is SessionPhase.WAITING_SCENE
    next_start = controller.start_next_episode(
        expected_episode_id=0,
        expected_generation=waiting.generation,
    )
    assert next_start.phase is SessionPhase.RECORDING_STARTING
    assert next_start.episode_id == 1
    assert next_start.task5_episode_uuid is None


def test_aborted_episode_is_quarantined_and_never_replay_eligible() -> None:
    """Catches operator abort being collapsed into a zero-reward failure."""
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionPhase
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(
        expected_generation=starting.generation,
        task5_episode_uuid="uuid-1",
    )
    aborted = controller.request_terminal(
        EpisodeOutcome.ABORTED,
        expected_episode_id=0,
        expected_generation=rollout.generation,
    )

    assert aborted.phase is SessionPhase.FINALIZING
    assert aborted.outcome is EpisodeOutcome.ABORTED
    assert aborted.replay_eligible is False


def test_horizon_pending_waits_for_operator_outcome() -> None:
    """Catches max steps being silently labeled failure or success."""
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionPhase
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(
        expected_generation=starting.generation,
        task5_episode_uuid="uuid-1",
    )
    pending = controller.mark_terminal_pending(
        reason="max_episode_steps",
        expected_episode_id=0,
        expected_generation=rollout.generation,
    )

    assert pending.phase is SessionPhase.TERMINAL_PENDING
    assert pending.outcome is None
    assert pending.policy_paused is True
    finalizing = controller.request_terminal(
        EpisodeOutcome.FAILURE,
        expected_episode_id=0,
        expected_generation=pending.generation,
    )
    assert finalizing.outcome is EpisodeOutcome.FAILURE


def test_stale_or_duplicate_operator_request_is_rejected() -> None:
    """Catches a delayed browser retry terminating the following episode."""
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionConflict
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(
        expected_generation=starting.generation,
        task5_episode_uuid="uuid-1",
    )

    with pytest.raises(SessionConflict, match="stale_generation"):
        controller.request_terminal(
            EpisodeOutcome.SUCCESS,
            expected_episode_id=0,
            expected_generation=starting.generation,
        )

    controller.request_terminal(
        EpisodeOutcome.SUCCESS,
        expected_episode_id=0,
        expected_generation=rollout.generation,
    )
    with pytest.raises(SessionConflict, match="invalid_phase"):
        controller.request_terminal(
            EpisodeOutcome.SUCCESS,
            expected_episode_id=0,
            expected_generation=controller.snapshot().generation,
        )


def test_fault_is_fail_closed_and_cannot_start_next_episode() -> None:
    """Catches a UI retry resuming policy after a recorder or safety fault."""
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionConflict, SessionPhase

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    controller.start_session(expected_generation=armed.generation)
    fault = controller.fail("task5_unreachable")

    assert fault.phase is SessionPhase.FAULT
    assert fault.policy_paused is True
    assert fault.fault_reason == "task5_unreachable"
    with pytest.raises(SessionConflict, match="invalid_phase"):
        controller.start_next_episode(
            expected_episode_id=fault.episode_id,
            expected_generation=fault.generation,
        )


def test_operator_pause_and_resume_requires_fresh_plan() -> None:
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionPhase

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(expected_generation=starting.generation)
    paused = controller.pause_session(expected_generation=rollout.generation)
    resumed = controller.resume_session(expected_generation=paused.generation)

    assert paused.phase is SessionPhase.PAUSED
    assert paused.policy_paused is True
    assert resumed.phase is SessionPhase.ROLLOUT
    assert resumed.fresh_plan_required is True


def test_hil_from_operator_pause_records_then_returns_to_pause() -> None:
    """A paused rollout may record HIL without resuming policy on release."""
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController, SessionPhase

    controller = RltSessionController(session_id_factory=lambda: "session-1")
    armed = controller.arm_operator()
    starting = controller.start_session(expected_generation=armed.generation)
    rollout = controller.recording_ready(expected_generation=starting.generation)
    paused = controller.pause_session(expected_generation=rollout.generation)

    hil = controller.update_takeover(left=False, right=True)
    released = controller.update_takeover(left=False, right=False)

    assert paused.phase is SessionPhase.PAUSED
    assert hil.phase is SessionPhase.HIL
    assert hil.policy_paused is False
    assert hil.expert_mask == (False, True)
    assert released.phase is SessionPhase.PAUSED
    assert released.policy_paused is True
    assert released.fresh_plan_required is False
