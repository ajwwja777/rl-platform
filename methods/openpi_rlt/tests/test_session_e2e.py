from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Ref:
    episode_index: int
    episode_uuid: str | None = None


class _Task5:
    def __init__(self) -> None:
        self.starts = 0
        self.outcomes = []
        self.recording = False
        self.capture_enabled = False

    def start_episode(self, _identity):
        ref = _Ref(self.starts)
        self.starts += 1
        self.recording = True
        self.capture_enabled = True
        return ref

    def finish_episode(self, ref, outcome):
        self.outcomes.append(outcome)
        self.recording = False
        return _Ref(ref.episode_index, f"uuid-{ref.episode_index}")

    def status(self):
        return {
            "state": "recording" if self.recording else "stopped",
            "capture_enabled": self.capture_enabled,
        }

    def set_capture_enabled(self, enabled):
        self.capture_enabled = bool(enabled)
        return self.status()


def test_three_episode_session_success_failure_with_hil_then_abort() -> None:
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController
    from methods.openpi_rlt.cobot_adapter.session_http import (
        RltSessionApplication,
        SessionHooks,
    )
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5EpisodeIdentity
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    paused = []
    submitted = []
    ready = []
    homes = []
    task5 = _Task5()
    app = RltSessionApplication(
        RltSessionController(session_id_factory=lambda: "session-e2e"),
        task5,
        identity_factory=lambda _episode_id: Task5EpisodeIdentity(
            "in_the_pot", "openpi_rlt", "step_4999", "online_r1", "/rlt", 600
        ),
        hooks=SessionHooks(
            is_policy_mode=lambda: True,
            set_policy_paused=lambda value: paused.append(bool(value)),
            submit_outcome=submitted.append,
            signal_episode_ready=lambda: ready.append(True),
            request_front_home=lambda: homes.append(True),
        ),
        home_after_terminal=True,
    )

    state = app.arm_operator()
    state = app.start(episode_id=-1, generation=state.generation)
    state = app.terminal(EpisodeOutcome.SUCCESS, episode_id=0, generation=state.generation)
    assert state.replay_eligible is True
    state = app.mark_replay_finalized()
    state = app.next_episode(episode_id=0, generation=state.generation)
    state = app.update_takeover(left=False, right=True)
    state = app.update_takeover(left=False, right=False)
    state = app.terminal(EpisodeOutcome.FAILURE, episode_id=1, generation=state.generation)
    assert state.replay_eligible is True
    state = app.mark_replay_finalized()
    state = app.next_episode(episode_id=1, generation=state.generation)
    state = app.terminal(EpisodeOutcome.ABORTED, episode_id=2, generation=state.generation)
    state = app.mark_replay_finalized()

    assert task5.starts == 3
    assert task5.outcomes == [EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE, EpisodeOutcome.ABORTED]
    assert submitted == task5.outcomes
    assert len(ready) == 3
    assert len(homes) == 3
    assert state.episode_id == 2
    assert state.task5_episode_uuid == "uuid-2"
    assert state.replay_eligible is False
    assert paused[-1] is True
