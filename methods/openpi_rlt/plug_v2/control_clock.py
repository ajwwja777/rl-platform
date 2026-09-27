"""Monotonic control scheduling, with no late-command catch-up burst."""
import math

def next_deadline(previous_deadline, now, last_publish=None, period=1/30):
    if period <= 0 or not all(math.isfinite(v) for v in (previous_deadline, now, period)):
        raise ValueError("invalid control clock")
    deadline = previous_deadline + period
    if deadline <= now:
        deadline = now + period
    if last_publish is not None:
        if not math.isfinite(last_publish):
            raise ValueError("invalid publication timestamp")
        deadline = max(deadline, last_publish + period)
    return deadline
