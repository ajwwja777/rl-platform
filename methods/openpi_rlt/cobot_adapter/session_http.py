"""Operator-facing RLT session orchestration and dependency-light HTTP UI."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from methods.openpi_rlt.cobot_adapter.session import (
    RltSessionController,
    SessionConflict,
    SessionPhase,
    SessionSnapshot,
)
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


@dataclass(frozen=True)
class SessionHooks:
    is_policy_mode: Callable[[], bool]
    set_policy_paused: Callable[[bool], None]
    submit_outcome: Callable[[EpisodeOutcome], None]
    signal_episode_ready: Callable[[], None]
    request_front_home: Callable[[], None]
    signal_policy_armed: Callable[[], None] | None = None


class RltSessionApplication:
    """Order policy, recorder and replay-visible terminal operations safely."""

    def __init__(
        self,
        controller: RltSessionController,
        task5_client: Any,
        *,
        identity_factory: Callable[[int], Any],
        hooks: SessionHooks,
        home_after_terminal: bool = False,
    ) -> None:
        self._controller = controller
        self._task5 = task5_client
        self._identity_factory = identity_factory
        self._hooks = hooks
        self._home_after_terminal = bool(home_after_terminal)
        self._terminal_home_requested = False
        self._lock = threading.RLock()
        self._task5_ref: Any | None = None
        self._metrics: dict[str, Any] = {
            "step": 0,
            "chunk_count": 0,
            "last_inference_latency_sec": None,
            "actor_version": None,
            "learner_version": None,
            "shadow_mode": False,
        }

    def snapshot(self) -> SessionSnapshot:
        return self._controller.snapshot()

    def status(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        result = asdict(snapshot)
        result["phase"] = snapshot.phase.value
        result["outcome"] = None if snapshot.outcome is None else snapshot.outcome.value
        result["policy_paused"] = snapshot.policy_paused
        result["replay_eligible"] = snapshot.replay_eligible
        result["home_after_terminal"] = self._home_after_terminal and not bool(
            self._metrics["shadow_mode"]
        )
        result.update(self._metrics)
        if result["shadow_mode"]:
            result["replay_eligible"] = False
        if self._task5_ref is not None:
            result["task5_episode_index"] = self._task5_ref.episode_index
        return result

    def update_metrics(self, **values: Any) -> None:
        allowed = set(self._metrics)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown session metric: {min(unknown)}")
        with self._lock:
            self._metrics.update(values)

    def _pause_capture_if_recording(self) -> None:
        if str(self._task5.status().get("state")) == "recording":
            self._task5.set_capture_enabled(False)

    def ensure_recorder_active(self) -> bool:
        """Pause before a new chunk if Task5 has reached its recorder ceiling."""
        snapshot = self.snapshot()
        if snapshot.phase not in {SessionPhase.ROLLOUT, SessionPhase.HIL}:
            return True
        # Network status may take tens of milliseconds. Never hold the
        # operator lock across it; a pause must revoke authority immediately.
        try:
            status = self._task5.status()
        except Exception:
            current = self.snapshot()
            if (current.episode_id != snapshot.episode_id
                    or current.generation != snapshot.generation
                    or current.phase not in {SessionPhase.ROLLOUT, SessionPhase.HIL}):
                return True
            raise
        with self._lock:
            current = self.snapshot()
            if (current.episode_id != snapshot.episode_id
                    or current.generation != snapshot.generation
                    or current.phase not in {SessionPhase.ROLLOUT, SessionPhase.HIL}):
                return True
            if str(status.get("state")) == "recording":
                if status.get("capture_enabled") is False:
                    self.mark_terminal_pending("task5_capture_paused_unexpectedly")
                    return False
                return True
            self.mark_terminal_pending("task5_recorder_stopped")
            return False

    def arm_operator(self) -> SessionSnapshot:
        if self._hooks.signal_policy_armed is not None:
            self._hooks.signal_policy_armed()
        self._hooks.set_policy_paused(True)
        return self._controller.arm_operator()

    def arm(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        """Expose the operator arm transition with the same stale guards as other HTTP actions."""
        with self._lock:
            self._verify_request(episode_id, generation)
            return self.arm_operator()

    def _verify_request(self, episode_id: int, generation: int) -> SessionSnapshot:
        snapshot = self.snapshot()
        if int(episode_id) != snapshot.episode_id:
            raise SessionConflict("stale_episode", f"expected episode {episode_id}, current {snapshot.episode_id}")
        if int(generation) != snapshot.generation:
            raise SessionConflict("stale_generation", f"expected generation {generation}, current {snapshot.generation}")
        return snapshot

    def _start_recorded_episode(self, starting: SessionSnapshot) -> SessionSnapshot:
        try:
            self._task5_ref = self._task5.start_episode(self._identity_factory(starting.episode_id))
            rollout = self._controller.recording_ready(expected_generation=starting.generation)
            self._hooks.signal_episode_ready()
            self._hooks.set_policy_paused(False)
            return rollout
        except Exception as error:
            self._hooks.set_policy_paused(True)
            detail = " ".join(str(error).split())[:384]
            reason = f"task5_start_failed: {type(error).__name__}"
            if detail:
                reason += f": {detail}"
            self._controller.fail(reason)
            print(f"[rlt-session] {reason}", flush=True)
            raise

    def start(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        with self._lock:
            snapshot = self._verify_request(episode_id, generation)
            if not self._hooks.is_policy_mode():
                raise SessionConflict("task2_not_policy", "Task2 is not in policy mode")
            if snapshot.phase in {SessionPhase.DISARMED, SessionPhase.STOPPED}:
                generation = self.arm_operator().generation
            elif snapshot.phase is SessionPhase.WAITING_SCENE:
                return self.next_episode(episode_id=episode_id, generation=generation)
            starting = self._controller.start_session(expected_generation=generation)
            return self._start_recorded_episode(starting)

    def prepare(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        """Open a session without moving; episode start can also do this implicitly."""
        with self._lock:
            snapshot = self._verify_request(episode_id, generation)
            if snapshot.phase in {SessionPhase.DISARMED, SessionPhase.STOPPED}:
                return self.arm_operator()
            if snapshot.phase not in {SessionPhase.READY, SessionPhase.WAITING_SCENE}:
                raise SessionConflict("episode_active", "finish the episode first")
            return snapshot

    def marker(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        with self._lock:
            snapshot = self._verify_request(episode_id, generation)
            if snapshot.phase not in {SessionPhase.ROLLOUT, SessionPhase.HIL, SessionPhase.PAUSED}:
                raise SessionConflict("episode_not_active", "no active episode")
            self._task5.record_marker("marker")
            return snapshot

    def terminal(
        self,
        outcome: EpisodeOutcome,
        *,
        episode_id: int,
        generation: int,
        home_after_terminal: bool = True,
    ) -> SessionSnapshot:
        with self._lock:
            self._verify_request(episode_id, generation)
            finalizing = self._controller.request_terminal(
                outcome,
                expected_episode_id=episode_id,
                expected_generation=generation,
            )
            self._hooks.set_policy_paused(True)
            self._terminal_home_requested = bool(home_after_terminal)
            try:
                if self._task5_ref is None:
                    raise RuntimeError("Task5 episode reference is missing")
                self._pause_capture_if_recording()
                finalized = self._task5.finish_episode(self._task5_ref, EpisodeOutcome(outcome))
                self._task5_ref = finalized
                # Saved/unlabelled HDF5 stays on disk, but the driver must drop
                # its learning trace exactly as for a discarded episode.
                self._hooks.submit_outcome(EpisodeOutcome.ABORTED if outcome is EpisodeOutcome.SAVED else EpisodeOutcome(outcome))
                committed = self._controller.episode_finalized(
                    expected_episode_id=finalizing.episode_id,
                    task5_episode_uuid=finalized.episode_uuid,
                )
                # Aborted episodes are deliberately excluded from replay.  There is
                # therefore no replay work for EnvDriver to acknowledge.  Advance
                # immediately instead of leaving the operator UI in
                # ``replay_committing`` while EnvDriver unwinds the terminal sample.
                # The normal post-episode hook remains the authority for success and
                # failure, because those outcomes can add transitions to replay.
                if EpisodeOutcome(outcome) in {EpisodeOutcome.ABORTED, EpisodeOutcome.SAVED}:
                    return self.mark_replay_finalized()
                return committed
            except Exception:
                self._terminal_home_requested = False
                self._controller.fail("task5_finalize_failed")
                raise

    def next_episode(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        with self._lock:
            self._verify_request(episode_id, generation)
            if not self._hooks.is_policy_mode():
                raise SessionConflict("task2_not_policy", "Task2 is not in policy mode")
            starting = self._controller.start_next_episode(
                expected_episode_id=episode_id,
                expected_generation=generation,
            )
            return self._start_recorded_episode(starting)

    def pause(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        with self._lock:
            self._verify_request(episode_id, generation)
            paused = self._controller.pause_session(expected_generation=generation)
            self._hooks.set_policy_paused(True)
            try:
                self._task5.set_capture_enabled(False)
                if hasattr(self._task5, "record_marker"):
                    self._task5.record_marker("pause")
            except Exception:
                self._controller.fail("task5_capture_pause_failed")
                raise
            return paused

    def resume(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        with self._lock:
            self._verify_request(episode_id, generation)
            if not self._hooks.is_policy_mode():
                raise SessionConflict("task2_not_policy", "Task2 is not in policy mode")
            resumed = self._controller.resume_session(expected_generation=generation)
            try:
                if hasattr(self._task5, "record_marker"):
                    self._task5.record_marker("resume")
                self._task5.set_capture_enabled(True)
            except Exception:
                self._hooks.set_policy_paused(True)
                self._controller.fail("task5_capture_resume_failed")
                raise
            self._hooks.set_policy_paused(False)
            return resumed

    def update_takeover(self, *, left: bool, right: bool) -> SessionSnapshot:
        with self._lock:
            before = self.snapshot()
            entering_paused_hil = (
                before.phase is SessionPhase.PAUSED and (left or right)
            )
            leaving_hil = before.phase is SessionPhase.HIL and not (left or right)
            if leaving_hil:
                try:
                    self._pause_capture_if_recording()
                except Exception:
                    self._hooks.set_policy_paused(True)
                    self._controller.fail("task5_capture_pause_failed")
                    raise
            updated = self._controller.update_takeover(left=left, right=right)
            if entering_paused_hil:
                try:
                    self._task5.set_capture_enabled(True)
                except Exception:
                    self._hooks.set_policy_paused(True)
                    self._controller.fail("task5_capture_resume_failed")
                    raise
            elif leaving_hil and updated.phase is SessionPhase.ROLLOUT:
                try:
                    self._task5.set_capture_enabled(True)
                except Exception:
                    self._hooks.set_policy_paused(True)
                    self._controller.fail("task5_capture_resume_failed")
                    raise
            return updated

    def mark_terminal_pending(self, reason: str) -> SessionSnapshot:
        with self._lock:
            snapshot = self.snapshot()
            pending = self._controller.mark_terminal_pending(
                reason=reason,
                expected_episode_id=snapshot.episode_id,
                expected_generation=snapshot.generation,
            )
            self._hooks.set_policy_paused(True)
            self._pause_capture_if_recording()
            print(
                f"[rlt-session] 策略已暂停，等待操作员选择成功/失败/放弃；reason={reason}",
                flush=True,
            )
            return pending

    def mark_replay_finalized(self) -> SessionSnapshot:
        with self._lock:
            snapshot = self.snapshot()
            # The abort fast path can finish before EnvDriver returns from the
            # episode.  Its eventual post-episode hook must be harmless.
            if snapshot.phase is not SessionPhase.REPLAY_COMMITTING:
                return snapshot
            should_home = (
                self._home_after_terminal
                and self._terminal_home_requested
                and not bool(self._metrics["shadow_mode"])
            )
            if should_home:
                try:
                    self._hooks.request_front_home()
                except Exception:
                    self._terminal_home_requested = False
                    self._controller.fail("front_home_failed")
                    raise
            self._terminal_home_requested = False
            return self._controller.replay_finalized(expected_episode_id=snapshot.episode_id)

    def stop(self, *, episode_id: int, generation: int) -> SessionSnapshot:
        with self._lock:
            self._verify_request(episode_id, generation)
            self._hooks.set_policy_paused(True)
            self._terminal_home_requested = False
            snapshot = self.snapshot()
            if self._task5_ref is not None and snapshot.phase in {
                SessionPhase.ROLLOUT,
                SessionPhase.HIL,
                SessionPhase.PAUSED,
                SessionPhase.TERMINAL_PENDING,
            }:
                try:
                    self._pause_capture_if_recording()
                    self._task5.finish_episode(self._task5_ref, EpisodeOutcome.ABORTED)
                    self._hooks.submit_outcome(EpisodeOutcome.ABORTED)
                except Exception:
                    self._controller.fail("task5_stop_failed")
                    raise
            return self._controller.stop()


class _ThreadingServer(ThreadingHTTPServer):
    daemon_threads = True


class RltSessionHttpServer:
    def __init__(self, application: RltSessionApplication, *, host: str, port: int) -> None:
        self._application = application
        static_root = Path(__file__).resolve().parents[1] / "web" / "rlt_session"
        handler = partial(_SessionHandler, application=application, directory=str(static_root))
        self._server = _ThreadingServer((host, int(port)), handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("session HTTP server already started")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


class _SessionHandler(SimpleHTTPRequestHandler):
    server_version = "CobotRltSession/1"

    def __init__(self, *args: Any, application: RltSessionApplication, **kwargs: Any) -> None:
        self._application = application
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        if self.command == "GET" and self.path in {
            "/",
            "/api/session",
            "/app.js",
            "/styles.css",
            "/favicon.ico",
        }:
            return
        print(f"[rlt-session] {self.address_string()} {format % args}")

    def _send_json(self, status: int, payload: object) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 16384:
            raise ValueError("invalid request size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise TypeError("request body must be an object")
        return payload

    def do_GET(self) -> None:
        if self.path == "/api/session":
            self._send_json(200, self._application.status())
            return
        super().do_GET()

    def do_POST(self) -> None:
        routes: dict[str, Callable[..., SessionSnapshot]] = {
            "/api/session/prepare": self._application.prepare,
            "/api/session/arm": self._application.arm,
            "/api/session/start": self._application.start,
            "/api/session/pause": self._application.pause,
            "/api/session/resume": self._application.resume,
            "/api/session/stop": self._application.stop,
            "/api/episode/next": self._application.next_episode,
            "/api/episode/marker": self._application.marker,
            "/api/episode/save": partial(self._application.terminal, EpisodeOutcome.SAVED),
            "/api/episode/success": partial(self._application.terminal, EpisodeOutcome.SUCCESS),
            "/api/episode/failure": partial(self._application.terminal, EpisodeOutcome.FAILURE),
            "/api/episode/abort": partial(self._application.terminal, EpisodeOutcome.ABORTED),
        }
        callback = routes.get(self.path)
        if callback is None:
            self._send_json(404, {"error": "not_found"})
            return
        try:
            body = self._body()
            callback_args: dict[str, Any] = {
                "episode_id": int(body["episode_id"]),
                "generation": int(body["generation"]),
            }
            if self.path in {
                "/api/episode/success",
                "/api/episode/failure",
                "/api/episode/abort",
                "/api/episode/save",
            }:
                callback_args["home_after_terminal"] = bool(
                    body.get("home_after_terminal", True)
                )
            snapshot = callback(**callback_args)
            self._send_json(200, self._application.status() | {"generation": snapshot.generation})
        except SessionConflict as error:
            self._send_json(409, {"error": error.code, "detail": str(error)})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": "invalid_request", "detail": str(error)})
        except Exception:  # noqa: BLE001 - dependency failures must fail closed at HTTP boundary
            self._send_json(503, {"error": "session_operation_failed"})
