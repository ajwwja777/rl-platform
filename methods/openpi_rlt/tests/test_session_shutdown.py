from __future__ import annotations

import threading
import time


def test_shutdown_aborts_active_episode_before_stopping_session() -> None:
    from methods.openpi_rlt.cobot_adapter.session import RltSessionController
    from methods.openpi_rlt.cobot_adapter.session_http import (
        RltSessionApplication,
        RltSessionHttpServer,
        SessionHooks,
    )
    from methods.openpi_rlt.cobot_adapter.session_shutdown import shutdown_via_http
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5EpisodeIdentity
    from methods.openpi_rlt.tests.test_session_http import _Hooks, _Task5

    task5 = _Task5()
    raw_hooks = _Hooks()
    app = RltSessionApplication(
        RltSessionController(session_id_factory=lambda: "shutdown-session"),
        task5,
        identity_factory=lambda _episode_id: Task5EpisodeIdentity(
            "in_the_pot", "openpi_rlt", "step_4999", "online_r1", "/rlt", 3600
        ),
        hooks=SessionHooks(
            is_policy_mode=raw_hooks.is_policy_mode,
            set_policy_paused=raw_hooks.set_policy_paused,
            submit_outcome=raw_hooks.submit_outcome,
            signal_episode_ready=raw_hooks.signal_episode_ready,
            request_front_home=raw_hooks.request_front_home,
        ),
        home_after_terminal=True,
    )
    armed = app.arm_operator()
    app.start(episode_id=-1, generation=armed.generation)
    server = RltSessionHttpServer(app, host="127.0.0.1", port=0)
    server.start()

    def finish_replay() -> None:
        deadline = time.monotonic() + 2
        while not raw_hooks.outcomes and time.monotonic() < deadline:
            time.sleep(0.01)
        app.mark_replay_finalized()

    thread = threading.Thread(target=finish_replay)
    thread.start()
    try:
        result = shutdown_via_http(server.url, timeout_sec=2)
    finally:
        thread.join(timeout=2)
        server.shutdown()

    assert result["phase"] == "stopped"
    assert len(task5.finished) == 1
    assert task5.finished[0][1].value == "aborted"
    assert raw_hooks.outcomes[0].value == "aborted"
    assert raw_hooks.homes == 0
