"""Episode-scoped actor snapshot selection."""

from __future__ import annotations

from threading import Lock
from typing import Any


class EpisodeActorPin:
    """Keep one immutable parameter reference/version for each active episode."""

    def __init__(self) -> None:
        self._episode_id: int | None = None
        self._params: Any = None
        self._version = -1
        self._lock = Lock()

    def select(self, *, episode_id: int | None, latest_params: Any, latest_version: int):
        if episode_id is None:
            raise ValueError("episode_id is required for actor pinning")
        episode = int(episode_id)
        with self._lock:
            if self._episode_id is not None and episode < self._episode_id:
                raise RuntimeError(
                    f"stale actor request episode_id={episode} after {self._episode_id}"
                )
            if self._episode_id != episode:
                self._episode_id = episode
                self._params = latest_params
                self._version = int(latest_version)
            return self._params, self._version
