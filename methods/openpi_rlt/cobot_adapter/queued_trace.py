"""Bounded metadata-only trace writes, outside the physical publication thread.

The queue is not a Replay buffer. Storage errors/overflow remain fatal; terminal
Replay admission must flush successfully. Pending evidence is never relabeled.
"""
from collections import deque
import copy
import threading
import time


class QueuedEpisodeTraceWriter:
    def __init__(self, writer, *, capacity=64, max_pending_age=1.0, clock=time.monotonic):
        if capacity <= 0 or max_pending_age <= 0:
            raise ValueError("Trace capacity and maximum age must be positive")
        self.writer = writer
        self._root = writer._root
        self.capacity, self.max_pending_age, self.clock = capacity, max_pending_age, clock
        self._condition = threading.Condition()
        self._producer_lock = threading.RLock()
        self._queue = deque()
        self._inflight = None
        self._error = None
        self._closed = False
        self._accepted = self._written = 0
        self._last_write_ms = self._max_write_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="rlt-trace-writer", daemon=True)
        self._thread.start()

    def _check_locked(self):
        if self._error is not None:
            raise RuntimeError("trace_storage_failed: " + str(self._error)) from self._error
        if self._closed:
            raise RuntimeError("trace_storage_closed")
        oldest = self._inflight if self._inflight is not None else (self._queue[0][0] if self._queue else None)
        if oldest is not None and self.clock() - oldest > self.max_pending_age:
            self._error = RuntimeError("oldest pending metadata exceeded age limit")
            raise RuntimeError("trace_storage_backlog: " + str(self._error))

    def check_health(self):
        with self._condition:
            self._check_locked()

    def diagnostics(self):
        with self._condition:
            oldest = self._inflight if self._inflight is not None else (self._queue[0][0] if self._queue else None)
            return dict(accepted=self._accepted, written=self._written,
                        pending=len(self._queue) + int(self._inflight is not None),
                        oldest_pending_ms=0. if oldest is None else (self.clock()-oldest)*1000,
                        last_write_ms=self._last_write_ms, max_write_ms=self._max_write_ms,
                        error=None if self._error is None else str(self._error))

    def append(self, record):
        # Match the compact writer's schema. Do not queue/deep-copy camera RGB.
        payload = dict(record)
        for name in ("observation", "next_observation"):
            observation = dict(payload.get(name, {}))
            observation.pop("images", None)
            payload[name] = observation
        with self._producer_lock:
            payload = copy.deepcopy(payload)
            with self._condition:
                self._check_locked()
                if len(self._queue) + int(self._inflight is not None) >= self.capacity:
                    self._error = RuntimeError("metadata queue capacity exceeded")
                    raise RuntimeError("trace_storage_queue_full: no metadata dropped; stop Episode")
                self._queue.append((self.clock(), payload))
                self._accepted += 1
                self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue:
                    return
                enqueued, payload = self._queue.popleft()
                self._inflight = enqueued
            started = self.clock()
            try:
                self.writer.append(payload)
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._last_write_ms = (self.clock()-started)*1000
                self._max_write_ms = max(self._max_write_ms, self._last_write_ms)
                self._written += 1
                self._inflight = None
                self._condition.notify_all()

    def flush(self, timeout=5.0):
        # Called while execution is paused/terminal; waiting is forbidden in
        # append/check_health, which run on the publication path.
        deadline = time.monotonic() + timeout
        with self._producer_lock, self._condition:
            while self._queue or self._inflight is not None:
                if self._error is not None:
                    raise RuntimeError("trace_storage_failed: " + str(self._error)) from self._error
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("trace_storage_flush_timeout: Episode cannot enter Replay")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("trace_storage_failed: " + str(self._error)) from self._error

    def start_episode(self):
        with self._producer_lock:
            self.flush()
            self.writer.start_episode()

    def discard(self):
        with self._producer_lock:
            self.flush()
            self.writer.discard()

    def finalize(self, outcome, *, identity=None):
        with self._producer_lock:
            self.flush()
            self.writer.finalize(outcome, identity=identity)

    def close(self, timeout=5.0):
        try:
            self.flush(timeout)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
        self._thread.join(timeout=timeout)
