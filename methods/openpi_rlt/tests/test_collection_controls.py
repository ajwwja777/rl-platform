from __future__ import annotations
from dataclasses import dataclass
import pytest

@dataclass
class _Ref:
    episode_index: int
    episode_uuid: str | None = None

class _Task5:
    def __init__(self) -> None:
        self.started: list[object] = []
        self.finished: list[tuple[object, object]] = []
        self.recording = True
        self.capture_enabled = True
        self.capture_changes: list[bool] = []

    def start_episode(self, identity):
        self.started.append(identity)
        return _Ref(episode_index=len(self.started) - 1)

    def finish_episode(self, ref, outcome):
        self.finished.append((ref, outcome))
        return _Ref(ref.episode_index, f"uuid-{ref.episode_index}")

    def status(self):
        return {
            "state": "recording" if self.recording else "stopped",
            "capture_enabled": self.capture_enabled,
        }

    def set_capture_enabled(self, enabled):
        self.capture_enabled = bool(enabled)
        self.capture_changes.append(bool(enabled))
        return self.status()

    def record_marker(self, kind):
        self.markers = getattr(self, "markers", []) + [kind]

def test_auto_start_saved_episode_and_restart_session():
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
    from methods.openpi_rlt.cobot_adapter.session import SessionConflict
    app, task5, hooks = _app()
    initial = app.snapshot()
    started = app.start(episode_id=initial.episode_id, generation=initial.generation)
    assert started.phase.value == "rollout"
    assert hooks.armed == 1
    with pytest.raises(SessionConflict):
        app.start(episode_id=initial.episode_id, generation=initial.generation)
    app.marker(episode_id=started.episode_id, generation=started.generation)
    paused = app.pause(episode_id=started.episode_id, generation=started.generation)
    resumed = app.resume(episode_id=paused.episode_id, generation=paused.generation)
    assert task5.markers == ["marker", "pause", "resume"]
    saved = app.terminal(EpisodeOutcome.SAVED, episode_id=resumed.episode_id, generation=resumed.generation, home_after_terminal=False)
    assert saved.phase.value == "waiting_scene" and not saved.replay_eligible
    assert task5.finished[-1][1] is EpisodeOutcome.SAVED
    assert hooks.outcomes[-1] is EpisodeOutcome.ABORTED
    second = app.start(episode_id=saved.episode_id, generation=saved.generation)
    assert second.episode_id == 1
    stopped = app.stop(episode_id=second.episode_id, generation=second.generation)
    prepared = app.prepare(episode_id=stopped.episode_id, generation=stopped.generation)
    assert prepared.phase.value == "ready" and prepared.policy_paused
    restarted = app.start(episode_id=prepared.episode_id, generation=prepared.generation)
    assert restarted.phase.value == "rollout" and restarted.outcome is None

class _Hooks:
    def __init__(self) -> None:
        self.pauses: list[bool] = []
        self.outcomes: list[object] = []
        self.ready = 0
        self.armed = 0
        self.homes = 0
        self.policy_mode = True

    def is_policy_mode(self):
        return self.policy_mode

    def set_policy_paused(self, paused):
        self.pauses.append(bool(paused))

    def submit_outcome(self, outcome):
        self.outcomes.append(outcome)

    def signal_episode_ready(self):
        self.ready += 1

    def signal_policy_armed(self):
        self.armed += 1

    def request_front_home(self):
        self.homes += 1

def _app(*, home_hook=None):
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController
    from methods.openpi_rlt.cobot_adapter.session_http import (
        RltSessionApplication,
        SessionHooks,
    )
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5EpisodeIdentity

    task5 = _Task5()
    raw_hooks = _Hooks()
    hooks = SessionHooks(
        is_policy_mode=raw_hooks.is_policy_mode,
        set_policy_paused=raw_hooks.set_policy_paused,
        submit_outcome=raw_hooks.submit_outcome,
        signal_episode_ready=raw_hooks.signal_episode_ready,
        request_front_home=home_hook or raw_hooks.request_front_home,
        signal_policy_armed=raw_hooks.signal_policy_armed,
    )
    identity = Task5EpisodeIdentity("in_the_pot", "openpi_rlt", "step_4999", "online_r1", "/rlt", 600)
    app = RltSessionApplication(
        RltSessionController(session_id_factory=lambda: "session-1"),
        task5,
        identity_factory=lambda _episode_id: identity,
        hooks=hooks,
        home_after_terminal=True,
    )
    return app, task5, raw_hooks
