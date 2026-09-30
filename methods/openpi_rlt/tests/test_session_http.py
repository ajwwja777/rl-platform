from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

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


def test_paused_hil_records_only_during_takeover_and_stays_paused_after_release() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    app, task5, _hooks = _app()
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    paused = app.pause(episode_id=0, generation=rollout.generation)
    assert paused.phase.value == "paused"
    assert task5.capture_enabled is False

    hil = app.update_takeover(left=False, right=True)
    assert hil.phase.value == "hil"
    assert task5.capture_enabled is True

    paused_again = app.update_takeover(left=False, right=False)
    assert paused_again.phase.value == "paused"
    assert task5.capture_enabled is False
    assert task5.capture_changes == [False, True, False]

    app.terminal(
        EpisodeOutcome.SUCCESS,
        episode_id=0,
        generation=paused_again.generation,
    )
    assert task5.capture_enabled is False
    assert task5.capture_changes[-1] is False


def test_resume_opens_capture_before_policy_unpauses() -> None:
    app, task5, hooks = _app()
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    paused = app.pause(episode_id=0, generation=rollout.generation)
    resumed = app.resume(episode_id=0, generation=paused.generation)
    assert resumed.phase.value == "rollout"
    assert task5.capture_changes == [False, True]
    assert hooks.pauses[-1] is False


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


def test_application_terminal_pauses_before_task5_and_only_then_submits_outcome() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    app, task5, hooks = _app()
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    assert rollout.phase.value == "rollout"
    assert hooks.pauses == [True, False]

    committing = app.terminal(EpisodeOutcome.SUCCESS, episode_id=0, generation=rollout.generation)

    assert committing.phase.value == "replay_committing"
    assert committing.task5_episode_uuid == "uuid-0"
    assert hooks.pauses == [True, False, True]
    assert len(task5.finished) == 1
    assert hooks.outcomes == [EpisodeOutcome.SUCCESS]
    assert hooks.homes == 0

    waiting = app.mark_replay_finalized()

    assert waiting.phase.value == "waiting_scene"
    assert hooks.homes == 1


def test_aborted_episode_skips_replay_committing_and_late_finalize_is_idempotent() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    app, task5, hooks = _app()
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)

    waiting = app.terminal(
        EpisodeOutcome.ABORTED,
        episode_id=0,
        generation=rollout.generation,
    )

    assert waiting.phase.value == "waiting_scene"
    assert waiting.replay_eligible is False
    assert waiting.task5_episode_uuid == "uuid-0"
    assert task5.finished[0][1] is EpisodeOutcome.ABORTED
    assert hooks.outcomes == [EpisodeOutcome.ABORTED]
    assert hooks.homes == 1

    # EnvDriver can return after the HTTP request has already completed.
    repeated = app.mark_replay_finalized()
    assert repeated.phase.value == "waiting_scene"
    assert hooks.homes == 1


def test_shadow_replay_finalize_never_requests_front_home() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    app, _task5, hooks = _app()
    app.update_metrics(shadow_mode=True)
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    app.terminal(EpisodeOutcome.SUCCESS, episode_id=0, generation=rollout.generation)

    waiting = app.mark_replay_finalized()

    assert waiting.phase.value == "waiting_scene"
    assert hooks.homes == 0


def test_front_home_failure_faults_before_waiting_scene() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    failing_home = lambda: (_ for _ in ()).throw(RuntimeError("home failed"))
    app, _task5, _hooks = _app(home_hook=failing_home)
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    app.terminal(EpisodeOutcome.FAILURE, episode_id=0, generation=rollout.generation)

    with pytest.raises(RuntimeError, match="home failed"):
        app.mark_replay_finalized()

    assert app.snapshot().phase.value == "fault"
    assert app.snapshot().fault_reason == "front_home_failed"


def test_shadow_status_never_advertises_replay_eligibility() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    app, _task5, _hooks = _app()
    app.update_metrics(shadow_mode=True)
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    app.terminal(EpisodeOutcome.SUCCESS, episode_id=0, generation=rollout.generation)

    assert app.status()["replay_eligible"] is False


def test_stale_terminal_request_never_pauses_or_finalizes() -> None:
    from methods.openpi_rlt.cobot_adapter.session import SessionConflict
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    app, task5, hooks = _app()
    armed = app.arm_operator()
    rollout = app.start(episode_id=-1, generation=armed.generation)
    with pytest.raises(SessionConflict, match="stale_generation"):
        app.terminal(EpisodeOutcome.FAILURE, episode_id=0, generation=armed.generation)
    assert hooks.pauses == [True, False]
    assert task5.finished == []
    assert rollout.phase.value == "rollout"


def test_task5_start_failure_is_fail_closed() -> None:
    app, task5, hooks = _app()
    task5.start_episode = lambda _identity: (_ for _ in ()).throw(RuntimeError("offline"))
    armed = app.arm_operator()
    with pytest.raises(RuntimeError, match="offline"):
        app.start(episode_id=-1, generation=armed.generation)
    assert app.snapshot().phase.value == "fault"
    assert app.snapshot().replay_eligible is False
    assert hooks.pauses == [True, True]


def test_recorder_safety_ceiling_pauses_policy_instead_of_running_unrecorded() -> None:
    app, task5, hooks = _app()
    armed = app.arm_operator()
    app.start(episode_id=-1, generation=armed.generation)
    task5.recording = False

    assert app.ensure_recorder_active() is False
    assert app.snapshot().phase.value == "terminal_pending"
    assert app.snapshot().terminal_reason == "task5_recorder_stopped"
    assert hooks.pauses[-1] is True


def test_http_api_rejects_stale_request_and_serves_status() -> None:
    from methods.openpi_rlt.cobot_adapter.session_http import RltSessionHttpServer

    app, _task5, _hooks = _app()
    server = RltSessionHttpServer(app, host="127.0.0.1", port=0)
    server.start()
    opener = build_opener(ProxyHandler({}))
    try:
        base = server.url
        with opener.open(base + "/api/session", timeout=2) as response:
            status = json.load(response)
        assert status["phase"] == "disarmed"
        app.arm_operator()
        request = Request(
            base + "/api/session/start",
            data=json.dumps({"episode_id": -1, "generation": 0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as caught:
            opener.open(request, timeout=2)
        assert caught.value.code == 409
    finally:
        server.shutdown()


def test_http_api_exposes_guarded_operator_arm() -> None:
    from methods.openpi_rlt.cobot_adapter.session_http import RltSessionHttpServer

    app, _task5, hooks = _app()
    server = RltSessionHttpServer(app, host="127.0.0.1", port=0)
    server.start()
    opener = build_opener(ProxyHandler({}))
    try:
        request = Request(
            server.url + "/api/session/arm",
            data=json.dumps({"episode_id": -1, "generation": 0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=2) as response:
            status = json.load(response)
        assert status["phase"] == "ready"
        assert status["generation"] == 1
        assert hooks.armed == 1
        assert hooks.pauses == [True]
    finally:
        server.shutdown()


def test_routine_status_poll_does_not_flood_operator_console(capsys) -> None:
    from methods.openpi_rlt.cobot_adapter.session_http import RltSessionHttpServer

    app, _task5, _hooks = _app()
    server = RltSessionHttpServer(app, host="127.0.0.1", port=0)
    server.start()
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(server.url + "/api/session", timeout=2) as response:
            assert response.status == 200
    finally:
        server.shutdown()
    assert "GET /api/session" not in capsys.readouterr().out

@pytest.mark.parametrize("network_error", [False, True])
def test_recorder_network_wait_does_not_block_pause_or_fault_new_generation(network_error):
    import threading
    app, task5, hooks = _app()
    initial = app.snapshot()
    rollout = app.start(episode_id=initial.episode_id, generation=initial.generation)
    entered, release = threading.Event(), threading.Event()
    results = []
    def slow_status():
        if threading.current_thread() is threading.main_thread():
            return {"state": "recording", "capture_enabled": task5.capture_enabled}
        entered.set()
        assert release.wait(2)
        if network_error:
            raise RuntimeError("old check HTTP timed out")
        return {"state": "stopped"}
    task5.status = slow_status
    thread = threading.Thread(target=lambda: results.append(app.ensure_recorder_active()))
    thread.start()
    assert entered.wait(1)
    try:
        paused = app.pause(episode_id=rollout.episode_id, generation=rollout.generation)
        assert paused.policy_paused and hooks.pauses[-1]
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert results == [True]
    assert app.snapshot().phase.value == "paused"
