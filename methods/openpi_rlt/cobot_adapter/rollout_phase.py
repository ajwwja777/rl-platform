"""Episode-boundary warmup to online transition for the Cobot ROS 1 adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
import time
from typing import Any


class CobotRolloutPhaseController:
    """Mirror upstream's readiness gates without importing its ROS 2 module."""

    def __init__(
        self,
        *,
        replay_stats_getter: Callable[[], Mapping[str, Any]],
        actor_version_getter: Callable[[], int],
        learner_status_getter: Callable[[], Mapping[str, Any]],
        warmup_min_size: int,
        min_online_actor_version: int,
        shutdown_requested: Callable[[], bool] = lambda: False,
        sleep: Callable[[float], None] = time.sleep,
        wait_interval_sec: float = 0.25,
        logger: logging.Logger | None = None,
    ) -> None:
        self._replay_stats_getter = replay_stats_getter
        self._actor_version_getter = actor_version_getter
        self._learner_status_getter = learner_status_getter
        self._warmup_min_size = max(int(warmup_min_size), 0)
        self._min_online_actor_version = max(int(min_online_actor_version), 0)
        self._shutdown_requested = shutdown_requested
        self._sleep = sleep
        self._wait_interval_sec = float(wait_interval_sec)
        self._logger = logger or logging.getLogger(__name__)
        self._episode_phase = "online" if self._warmup_min_size == 0 else "warmup"

    @property
    def episode_phase(self) -> str:
        return self._episode_phase

    def begin_episode(self) -> str:
        if self._warmup_min_size == 0:
            self._episode_phase = "online"
            return self._episode_phase
        replay_size = self._safe_replay_size()
        if replay_size < self._warmup_min_size:
            self._episode_phase = "warmup"
            return self._episode_phase

        last_progress: tuple[int, int, bool] | None = None
        while True:
            actor_version = self._safe_actor_version()
            learner_status = self._safe_learner_status()
            ready = bool(learner_status.get("ready_for_online", False))
            global_step = int(learner_status.get("global_step", 0))
            progress = (actor_version, global_step, ready)
            if actor_version >= self._min_online_actor_version and ready:
                self._episode_phase = "online"
                return self._episode_phase
            if progress != last_progress:
                self._logger.info(
                    "Cobot RLT waiting between episodes: actor_version=%s/%s learner_step=%s ready=%s",
                    actor_version,
                    self._min_online_actor_version,
                    global_step,
                    ready,
                )
                last_progress = progress
            if self._shutdown_requested():
                raise RuntimeError("Cobot RLT shut down while waiting for online readiness")
            self._sleep(self._wait_interval_sec)

    def finish_episode(self) -> None:
        # Readiness is deliberately re-evaluated only on the next begin call,
        # after EnvDriver has atomically finalized replay for this episode.
        return

    def _safe_replay_size(self) -> int:
        try:
            return int(self._replay_stats_getter().get("size", 0))
        except (OSError, RuntimeError, TypeError, ValueError):
            return 0

    def _safe_actor_version(self) -> int:
        try:
            return int(self._actor_version_getter())
        except (OSError, RuntimeError, TypeError, ValueError):
            return -1

    def _safe_learner_status(self) -> dict[str, Any]:
        try:
            return dict(self._learner_status_getter())
        except (OSError, RuntimeError, TypeError, ValueError):
            return {}
