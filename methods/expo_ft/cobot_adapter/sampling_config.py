"""Validated task-specific exploration scale; absent keeps the native recipe."""
import math


def edit_scale(value):
    if value is None:return None
    if isinstance(value,bool):raise ValueError('Edit scale must be a finite number')
    value=float(value)
    if not math.isfinite(value) or not 0<value<=1:
        raise ValueError('Edit scale must be finite and in (0,1]')
    return value
