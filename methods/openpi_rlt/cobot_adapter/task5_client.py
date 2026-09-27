"""Small fail-closed client for the existing Task5 v1 HTTP boundary."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


class Task5ClientError(RuntimeError):
    """Task5 was unavailable or violated the expected recorder contract."""


@dataclass(frozen=True)
class Task5EpisodeIdentity:
    task_id: str
    model_id: str
    checkpoint_id: str
    dataset_round: str
    data_root: str
    max_timesteps: int


@dataclass(frozen=True)
class Task5EpisodeRef:
    identity: Task5EpisodeIdentity
    episode_index: int
    episode_uuid: str | None = None


class Task5Client:
    """Coordinate one Task5 recording without importing its historical source."""

    _MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float = 10.0,
        poll_interval_sec: float = 0.1,
        min_free_bytes: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = float(timeout_sec)
        self._poll_interval_sec = float(poll_interval_sec)
        self._sleep = sleep
        self._min_free_bytes = int(min_free_bytes)
        self._opener = build_opener(ProxyHandler({}))
        if self._timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if self._min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")

    def _request(self, method: str, path: str, body: object | None = None) -> object:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self._base_url + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self._timeout_sec) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            detail = ""
            try:
                error_raw = error.read(self._MAX_RESPONSE_BYTES + 1)
                if len(error_raw) <= self._MAX_RESPONSE_BYTES:
                    payload = json.loads(error_raw)
                    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
                        detail = " ".join(payload["detail"].split())[:512]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise Task5ClientError(
                f"Task5 {method} {path} returned HTTP {error.code}{suffix}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise Task5ClientError(f"Task5 {method} {path} unavailable") from error
        if len(raw) > self._MAX_RESPONSE_BYTES:
            raise Task5ClientError("Task5 response exceeded size limit")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Task5ClientError("Task5 returned invalid JSON") from error

    def health(self) -> None:
        payload = self._request("GET", "/healthz")
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise Task5ClientError("Task5 is not ready")

    def status(self) -> dict[str, object]:
        payload = self._request("GET", "/api/status")
        if not isinstance(payload, dict):
            raise Task5ClientError("Task5 status is not an object")
        return payload

    def set_capture_enabled(self, enabled: bool) -> dict[str, object]:
        operation = "resume" if enabled else "pause"
        payload = self._request("POST", f"/api/episodes/capture/{operation}")
        if (
            not isinstance(payload, dict)
            or payload.get("state") != "recording"
            or payload.get("capture_enabled") is not enabled
        ):
            raise Task5ClientError(f"Task5 capture {operation} was not confirmed")
        return payload

    def _wait_status(self, expected: set[str]) -> dict[str, object]:
        deadline = time.monotonic() + self._timeout_sec
        while True:
            payload = self._request("GET", "/api/status")
            if not isinstance(payload, dict):
                raise Task5ClientError("Task5 status is not an object")
            state = str(payload.get("state", "unknown"))
            if state in expected:
                return payload
            if state in {"error", "fatal"} or time.monotonic() >= deadline:
                raise Task5ClientError(f"Task5 status did not reach {sorted(expected)}; state={state}")
            self._sleep(self._poll_interval_sec)

    def _recover_orphaned_latest_episode(
        self,
        identity: Task5EpisodeIdentity,
        prepared: dict[str, object],
    ) -> dict[str, object]:
        """Exclude a finalized, unlabeled episode left by an interrupted RLT shutdown."""
        latest_uuid = prepared.get("latest_episode_uuid")
        latest_index = prepared.get("latest_episode_index")
        if not isinstance(latest_uuid, str):
            raise Task5ClientError("Task5 latest episode labels are incomplete")
        try:
            latest_index = int(latest_index)
        except (TypeError, ValueError) as error:
            raise Task5ClientError("Task5 latest episode labels are incomplete") from error

        query = urlencode({"data_root": identity.data_root})
        episodes = self._request("GET", f"/api/episodes?{query}")
        if not isinstance(episodes, list):
            raise Task5ClientError("Task5 episode list is not an array")
        matches = [
            row
            for row in episodes
            if isinstance(row, dict)
            and row.get("episode_uuid") == latest_uuid
            and row.get("task_id") == identity.task_id
            and row.get("model_id") == identity.model_id
            and row.get("checkpoint_id") == identity.checkpoint_id
            and row.get("dataset_round") == identity.dataset_round
            and row.get("episode_index") == latest_index
            and row.get("completion_state") == "complete"
            and not bool(row.get("has_labels"))
        ]
        if len(matches) != 1:
            raise Task5ClientError("Task5 latest episode labels are incomplete")
        self._request(
            "PUT",
            f"/api/episodes/{latest_uuid}/labels?{query}",
            {
                "episode_uuid": latest_uuid,
                "episode_outcome": "aborted",
                "episode_quality": "bad",
                "termination_reason": "operator_abort",
                "keep_for_training": "false",
                "operator_note": (
                    "Recovered before a new RLT episode after interrupted finalization; "
                    "excluded from training."
                ),
            },
        )
        refreshed = self._request(
            "POST",
            "/api/storage/prepare",
            {
                "data_root": identity.data_root,
                "task_id": identity.task_id,
                "model_id": identity.model_id,
                "checkpoint_id": identity.checkpoint_id,
                "dataset_round": identity.dataset_round,
            },
        )
        if not isinstance(refreshed, dict) or refreshed.get("label_blocked"):
            raise Task5ClientError("Task5 latest episode labels remain incomplete after recovery")
        return refreshed

    def start_episode(self, identity: Task5EpisodeIdentity) -> Task5EpisodeRef:
        self._operator_nodes = []
        if self._min_free_bytes:
            free = shutil.disk_usage(Path(identity.data_root)).free
            if free < self._min_free_bytes:
                raise Task5ClientError(
                    f"Task5 data filesystem has insufficient free space: {free} bytes"
                )
        self.health()
        prepared = self._request(
            "POST",
            "/api/storage/prepare",
            {
                "data_root": identity.data_root,
                "task_id": identity.task_id,
                "model_id": identity.model_id,
                "checkpoint_id": identity.checkpoint_id,
                "dataset_round": identity.dataset_round,
            },
        )
        if not isinstance(prepared, dict):
            raise Task5ClientError("Task5 storage response is not an object")
        if prepared.get("label_blocked"):
            prepared = self._recover_orphaned_latest_episode(identity, prepared)
        try:
            episode_index = int(prepared["next_episode_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise Task5ClientError("Task5 did not return next_episode_index") from error
        started = self._request(
            "POST",
            "/api/episodes/start",
            {
                "task_id": identity.task_id,
                "model_id": identity.model_id,
                "checkpoint_id": identity.checkpoint_id,
                "dataset_round": identity.dataset_round,
                "data_root": identity.data_root,
                "episode_index": episode_index,
                "max_timesteps": identity.max_timesteps,
            },
        )
        self._wait_status({"recording"})
        episode_uuid = started.get("episode_uuid") if isinstance(started, dict) else None
        return Task5EpisodeRef(identity=identity, episode_index=episode_index, episode_uuid=episode_uuid)

    def _resolve_uuid(self, ref: Task5EpisodeRef) -> str:
        query = urlencode({"data_root": ref.identity.data_root})
        payload = self._request("GET", f"/api/episodes?{query}")
        if not isinstance(payload, list):
            raise Task5ClientError("Task5 episode list is not an array")
        matches = [
            row
            for row in payload
            if isinstance(row, dict)
            and row.get("task_id") == ref.identity.task_id
            and row.get("model_id") == ref.identity.model_id
            and row.get("checkpoint_id") == ref.identity.checkpoint_id
            and row.get("dataset_round") == ref.identity.dataset_round
            and row.get("episode_index") == ref.episode_index
            and row.get("completion_state") == "complete"
        ]
        if len(matches) != 1 or not matches[0].get("episode_uuid"):
            raise Task5ClientError("Task5 finalized episode identity was not unique")
        return str(matches[0]["episode_uuid"])

    def record_marker(self, kind: str) -> None:
        if kind not in {"pause", "resume", "marker"}:
            raise ValueError("invalid marker kind")
        status = self.status()
        frame = max(0, int(status.get("frames_written", 0)) - 1)
        nodes = getattr(self, "_operator_nodes", [])
        if len(nodes) >= 4096:
            raise Task5ClientError("episode marker limit reached")
        nodes.append({"frame_index": frame, "node_kind": kind})
        self._operator_nodes = nodes

    def finish_episode(self, ref: Task5EpisodeRef, outcome: EpisodeOutcome) -> Task5EpisodeRef:
        terminal = EpisodeOutcome(outcome)
        if terminal not in (EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE, EpisodeOutcome.ABORTED, EpisodeOutcome.SAVED):
            raise ValueError("terminal outcome must be success, failure, or aborted")
        if terminal is EpisodeOutcome.ABORTED:
            if not ref.episode_uuid:
                raise Task5ClientError("Task5 discard requires the UUID returned by start")
            discarded = self._request("POST", "/api/episodes/discard", {"episode_uuid": ref.episode_uuid})
            if not isinstance(discarded, dict) or not discarded.get("discarded") or discarded.get("episode_uuid") != ref.episode_uuid:
                raise Task5ClientError("Task5 discard was not confirmed")
            return ref
        status = self._request("GET", "/api/status")
        if not isinstance(status, dict):
            raise Task5ClientError("Task5 status is not an object")
        state = str(status.get("state", "unknown"))
        if state in {"starting", "recording", "stopping"}:
            self._request("POST", "/api/episodes/stop")
            self._wait_status({"idle", "stopped"})
        elif state not in {"idle", "stopped"}:
            raise Task5ClientError(f"Task5 cannot finalize from state={state}")
        episode_uuid = ref.episode_uuid
        deadline = time.monotonic() + self._timeout_sec
        while episode_uuid is None:
            try:
                episode_uuid = self._resolve_uuid(ref)
            except Task5ClientError:
                if time.monotonic() >= deadline:
                    raise
                self._sleep(self._poll_interval_sec)
        query = urlencode({"data_root": ref.identity.data_root})
        if terminal in (EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE):
            self._request(
                "POST",
                f"/api/episodes/{episode_uuid}/outcome?{query}",
                {"episode_uuid": episode_uuid, "outcome": terminal.value},
            )
        else:
            self._request(
                "PUT",
                f"/api/episodes/{episode_uuid}/labels?{query}",
                {
                    "episode_uuid": episode_uuid,
                    "episode_outcome": "unknown",
                    "episode_quality": "uncertain",
                    "termination_reason": "operator_save",
                    "keep_for_training": "false",
                },
            )
        if getattr(self, "_operator_nodes", []):
            self._request("PUT", f"/api/episodes/{episode_uuid}/labels?{query}", {
                "episode_uuid": episode_uuid, "operator_nodes": self._operator_nodes,
            })
        return replace(ref, episode_uuid=episode_uuid)
