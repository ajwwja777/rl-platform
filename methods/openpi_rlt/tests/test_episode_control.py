from __future__ import annotations

import pytest


def _controller():
    from methods.openpi_rlt.cobot_adapter.episode_control import EpisodeController

    return EpisodeController()


def test_teach_button_takeover_pauses_and_release_requires_fresh_plan_without_ui_start() -> None:
    controller = _controller()
    started = controller.arm()
    controller.mark_plan_installed(started.generation)

    taking_over = controller.update_takeover(left=True, right=False)
    assert taking_over.phase.value == "hil"
    assert taking_over.policy_paused is True
    assert taking_over.expert_mask == (True, False)
    assert taking_over.raw_trace_enabled is True
    assert controller.accepts_policy_result(started.generation) is False

    resumed = controller.update_takeover(left=False, right=False)
    assert resumed.phase.value == "rollout"
    assert resumed.policy_paused is False
    assert resumed.expert_mask == (False, False)
    assert resumed.fresh_plan_required is True
    assert resumed.generation > started.generation


def test_partial_and_bilateral_takeover_masks_are_preserved_while_policy_is_globally_paused() -> None:
    controller = _controller()
    controller.arm()

    left_only = controller.update_takeover(left=True, right=False)
    both = controller.update_takeover(left=True, right=True)
    right_only = controller.update_takeover(left=False, right=True)

    assert left_only.expert_mask == (True, False)
    assert both.expert_mask == (True, True)
    assert right_only.expert_mask == (False, True)
    assert left_only.policy_paused and both.policy_paused and right_only.policy_paused
    assert both.generation == left_only.generation == right_only.generation


def test_terminal_reset_and_object_ready_signal_autostart_the_next_episode() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    controller = _controller()
    first = controller.arm()
    controller.mark_plan_installed(first.generation)

    resetting = controller.finish_episode(EpisodeOutcome.SUCCESS)
    assert resetting.phase.value == "resetting_robot"
    assert resetting.policy_paused is True
    assert resetting.outcome is EpisodeOutcome.SUCCESS
    assert resetting.replay_commit_ready is True

    waiting = controller.mark_robot_reset_complete()
    assert waiting.phase.value == "waiting_object_reset"
    assert waiting.raw_trace_enabled is False

    second = controller.confirm_object_reset_ready()
    assert second.phase.value == "rollout"
    assert second.episode_id == first.episode_id + 1
    assert second.policy_paused is False
    assert second.fresh_plan_required is True
    assert second.outcome is None


def test_pending_episode_cannot_be_committed_or_automatically_restarted() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    controller = _controller()
    controller.arm()

    with pytest.raises(ValueError, match="pending"):
        controller.finish_episode(EpisodeOutcome.PENDING)
    with pytest.raises(RuntimeError, match="resetting_robot"):
        controller.mark_robot_reset_complete()


def test_fault_is_fail_closed_and_object_ready_cannot_resume_it() -> None:
    controller = _controller()
    controller.arm()
    faulted = controller.fail("camera stale")

    assert faulted.phase.value == "fault"
    assert faulted.policy_paused is True
    assert controller.accepts_policy_result(faulted.generation) is False
    with pytest.raises(RuntimeError, match="waiting_object_reset"):
        controller.confirm_object_reset_ready()

