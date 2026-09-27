from __future__ import annotations


def test_phase_switches_only_between_episodes_after_replay_learner_and_actor_ready() -> None:
    from methods.openpi_rlt.cobot_adapter.rollout_phase import CobotRolloutPhaseController

    state = {
        "replay_size": 5,
        "actor_version": 0,
        "learner": {"ready_for_online": False, "global_step": 4},
    }

    def advance(_seconds):
        state["actor_version"] = 1
        state["learner"] = {"ready_for_online": True, "global_step": 20_000}

    controller = CobotRolloutPhaseController(
        replay_stats_getter=lambda: {"size": state["replay_size"]},
        actor_version_getter=lambda: state["actor_version"],
        learner_status_getter=lambda: state["learner"],
        warmup_min_size=10,
        min_online_actor_version=1,
        sleep=advance,
    )

    assert controller.begin_episode() == "warmup"
    state["replay_size"] = 10
    # The current episode remains warmup even after the threshold is crossed.
    assert controller.episode_phase == "warmup"
    controller.finish_episode()
    assert controller.begin_episode() == "online"


def test_phase_wait_aborts_on_runtime_shutdown() -> None:
    import pytest

    from methods.openpi_rlt.cobot_adapter.rollout_phase import CobotRolloutPhaseController

    controller = CobotRolloutPhaseController(
        replay_stats_getter=lambda: {"size": 10},
        actor_version_getter=lambda: 0,
        learner_status_getter=lambda: {"ready_for_online": False},
        warmup_min_size=10,
        min_online_actor_version=1,
        shutdown_requested=lambda: True,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="shut down"):
        controller.begin_episode()
