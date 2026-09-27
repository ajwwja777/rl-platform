"""Gracefully close an operator session through its localhost HTTP boundary."""

from __future__ import annotations

import json
import time
from urllib.request import ProxyHandler, Request, build_opener

ACTIVE_PHASES = {"rollout", "hil", "paused", "terminal_pending"}


def _request(opener, base_url: str, method: str, path: str, body=None, timeout_sec=5.0):
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    with opener.open(request, timeout=timeout_sec) as response:
        return json.load(response)


def shutdown_via_http(base_url: str, *, timeout_sec: float = 10.0) -> dict:
    """Abort any active episode, let replay/trace finalize, then stop the session."""
    opener = build_opener(ProxyHandler({}))
    status = _request(opener, base_url, "GET", "/api/session", timeout_sec=timeout_sec)
    if status.get("phase") in ACTIVE_PHASES:
        status = _request(
            opener,
            base_url,
            "POST",
            "/api/episode/abort",
            {
                "episode_id": status["episode_id"],
                "generation": status["generation"],
                "home_after_terminal": False,
            },
            timeout_sec=timeout_sec,
        )
        deadline = time.monotonic() + timeout_sec
        while status.get("phase") == "replay_committing" and time.monotonic() < deadline:
            time.sleep(0.05)
            status = _request(
                opener, base_url, "GET", "/api/session", timeout_sec=timeout_sec
            )
    if status.get("phase") not in {"disarmed", "stopped"}:
        status = _request(
            opener,
            base_url,
            "POST",
            "/api/session/stop",
            {"episode_id": status["episode_id"], "generation": status["generation"]},
            timeout_sec=timeout_sec,
        )
    return status
