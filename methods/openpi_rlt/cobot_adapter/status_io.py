"""Bound derived learner-status disk writes without throttling learning."""
import atexit
import time
from functools import wraps


def install_status_write_throttle(learner_class=None, *, interval_sec=1.0,
                                  clock=time.monotonic, register_exit=atexit.register):
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    if learner_class is None:
        from rlt_online_rl.trainer import LearnerService
        learner_class = LearnerService
    original = learner_class._write_status
    if getattr(original, "_cobot_status_throttled", False):
        return

    @wraps(original)
    def write_status(self, progress):
        state = getattr(self, "_cobot_status_io_state", None)
        if state is None:
            state = {"last": float("-inf"), "critical": None, "pending": None}
            self._cobot_status_io_state = state
            def flush():
                if state["pending"] is not None:
                    original(self, state["pending"])
                    state["pending"] = None
            register_exit(flush)
        state["pending"] = dict(progress)
        now = clock()
        critical = (bool(progress.get("ready_for_online", False)),
                    getattr(self, "_warmup_ready_adds_total", None))
        if now - state["last"] < interval_sec and critical == state["critical"]:
            return
        original(self, state["pending"])
        state.update(last=now, critical=critical, pending=None)

    write_status._cobot_status_throttled = True
    learner_class._write_status = write_status
