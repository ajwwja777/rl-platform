from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.parse import parse_qs, urlparse

import pytest


class _State:
    def __init__(self) -> None:
        self.recording = False
        self.requests: list[tuple[str, str, object | None]] = []
        self.episode_index = 7
        self.episode_uuid = "123e4567-e89b-12d3-a456-426614174000"
        self.start_status = 200
        self.start_payload: object = {"state": "recording", "active": True, "episode_uuid": self.episode_uuid}
        self.label_blocked = False
        self.capture_enabled = False


@contextmanager
def _task5_server():
    state = _State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _body(self) -> object | None:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length)) if length else None

        def _send(self, status: int, payload: object) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            state.requests.append(("GET", parsed.path, parse_qs(parsed.query)))
            if parsed.path == "/healthz":
                self._send(200, {"status": "ok"})
            elif parsed.path == "/api/status":
                self._send(
                    200,
                    {
                        "state": "recording" if state.recording else "stopped",
                        "active": state.recording,
                        "completion_state": None if state.recording else "complete",
                    },
                )
            elif parsed.path == "/api/episodes":
                self._send(
                    200,
                    [
                        {
                            "episode_uuid": state.episode_uuid,
                            "task_id": "in_the_pot",
                            "model_id": "openpi_rlt",
                            "checkpoint_id": "step_4999",
                            "dataset_round": "online_r1",
                            "episode_index": state.episode_index,
                            "completion_state": "complete",
                            "has_labels": not state.label_blocked,
                        }
                    ],
                )
            else:
                self._send(404, {"detail": "missing"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            body = self._body()
            state.requests.append(("POST", parsed.path, body))
            if parsed.path == "/api/storage/prepare":
                self._send(
                    200,
                    {
                        "next_episode_index": state.episode_index,
                        "label_blocked": state.label_blocked,
                        "latest_episode_index": state.episode_index,
                        "latest_episode_uuid": state.episode_uuid,
                    },
                )
            elif parsed.path == "/api/episodes/start":
                state.recording = state.start_status == 200
                state.capture_enabled = state.recording
                self._send(state.start_status, state.start_payload)
            elif parsed.path in {"/api/episodes/capture/pause", "/api/episodes/capture/resume"}:
                state.capture_enabled = parsed.path.endswith("/resume")
                self._send(200, {
                    "state": "recording" if state.recording else "stopped",
                    "capture_enabled": state.capture_enabled,
                })
            elif parsed.path == "/api/episodes/stop":
                state.recording = False
                self._send(200, {"state": "stopped", "active": False, "completion_state": "complete"})
            elif parsed.path == "/api/episodes/discard":
                state.recording = False
                self._send(200, {"discarded": True, "episode_uuid": state.episode_uuid})
            elif parsed.path.endswith("/outcome"):
                self._send(200, {"labels_complete": True})
            else:
                self._send(404, {"detail": "missing"})

        def do_PUT(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            body = self._body()
            state.requests.append(("PUT", parsed.path, body))
            state.label_blocked = False
            self._send(200, {"labels_complete": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _identity():
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5EpisodeIdentity

    return Task5EpisodeIdentity(
        task_id="in_the_pot",
        model_id="openpi_rlt",
        checkpoint_id="step_4999",
        dataset_round="online_r1",
        data_root="/registered/jiaan/rlt/task5",
        max_timesteps=300,
    )


def test_task5_client_recovers_finalized_unlabeled_episode_before_start() -> None:
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client

    with _task5_server() as (state, url):
        state.label_blocked = True
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        ref = client.start_episode(_identity())

    assert ref.episode_index == state.episode_index
    recovery = next(
        body
        for method, path, body in state.requests
        if method == "PUT" and path.endswith("/labels")
    )
    assert recovery["episode_outcome"] == "aborted"
    assert recovery["episode_quality"] == "bad"
    assert recovery["keep_for_training"] == "false"
    assert sum(
        method == "POST" and path == "/api/storage/prepare"
        for method, path, _body in state.requests
    ) == 2


def test_task5_client_records_then_labels_success() -> None:
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    with _task5_server() as (state, url):
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        ref = client.start_episode(_identity())
        assert ref.episode_index == 7
        assert ref.episode_uuid == state.episode_uuid
        finalized = client.finish_episode(ref, EpisodeOutcome.SUCCESS)

    assert finalized.episode_uuid == state.episode_uuid
    start_body = next(body for method, path, body in state.requests if method == "POST" and path == "/api/episodes/start")
    assert start_body["episode_index"] == 7
    assert start_body["data_root"] == "/registered/jiaan/rlt/task5"
    outcome_body = next(body for method, path, body in state.requests if path.endswith("/outcome"))
    assert outcome_body == {"episode_uuid": state.episode_uuid, "outcome": "success"}


def test_task5_capture_gate_pause_resume_contract() -> None:
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client

    with _task5_server() as (state, url):
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        client.start_episode(_identity())
        assert client.set_capture_enabled(False)["capture_enabled"] is False
        assert client.set_capture_enabled(True)["capture_enabled"] is True
    assert [
        path for method, path, _ in state.requests
        if method == "POST" and "/capture/" in path
    ] == ["/api/episodes/capture/pause", "/api/episodes/capture/resume"]


def test_unlabelled_save_keeps_hdf5_without_outcome_or_training():
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome, terminal_rewards
    with _task5_server() as (state, url):
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        ref = client.start_episode(_identity())
        client.record_marker('marker')
        client.finish_episode(ref, EpisodeOutcome.SAVED)
    writes = [(method,path,body) for method,path,body in state.requests if method != 'GET']
    assert any(path == '/api/episodes/stop' for _,path,_ in writes)
    assert not any(path.endswith('/outcome') or path.endswith('/discard') for _,path,_ in writes)
    label = next(body for _,path,body in writes if path.endswith('/labels'))
    assert label['episode_outcome'] == 'unknown' and label['keep_for_training'] == 'false'
    assert label['termination_reason'] == 'operator_save'
    assert writes[-1][2]['operator_nodes'] == [{'frame_index':0,'node_kind':'marker'}]
    with pytest.raises(ValueError): terminal_rewards(1, EpisodeOutcome.SAVED)


def test_task5_client_discards_without_finalizing_or_saving_labels() -> None:
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    with _task5_server() as (state, url):
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        ref = client.start_episode(_identity())
        client.finish_episode(ref, EpisodeOutcome.ABORTED)

    update = next(body for method, path, body in state.requests if path == "/api/episodes/discard")
    assert update == {"episode_uuid": state.episode_uuid}
    assert not any(path == "/api/episodes/stop" or path.endswith("/labels") for method, path, body in state.requests)


def test_task5_client_rejects_pending_outcome() -> None:
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client, Task5EpisodeRef
    from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome

    with _task5_server() as (_state, url):
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        ref = Task5EpisodeRef(identity=_identity(), episode_index=7)
        with pytest.raises(ValueError, match="terminal outcome"):
            client.finish_episode(ref, EpisodeOutcome.PENDING)


def test_task5_client_preserves_http_error_detail() -> None:
    from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client, Task5ClientError

    with _task5_server() as (state, url):
        state.start_status = 422
        state.start_payload = {"detail": "invalid_start_request: ValueError: max_timesteps"}
        client = Task5Client(url, timeout_sec=1, poll_interval_sec=0)
        with pytest.raises(
            Task5ClientError,
            match=r"HTTP 422.*invalid_start_request: ValueError: max_timesteps",
        ):
            client.start_episode(_identity())
