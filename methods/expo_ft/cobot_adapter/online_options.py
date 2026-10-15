"""Load-time EXPO choices shared by Web, Actor and Learner launchers.

Pure validation only: no ROS, model imports, files or process side effects.
"""
import math

KEYS = frozenset(('rtc_delay', 'edit_scale'))


def normalize(values):
    if not isinstance(values, dict) or set(values) != KEYS:
        raise ValueError('EXPO options require rtc_delay and edit_scale')
    delay, scale = values['rtc_delay'], values['edit_scale']
    if type(delay) is not int or not 1 <= delay <= 8:
        raise ValueError('RTC d must be an integer from 1 to 8 (100 ms per step)')
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError('Edit scale must be a finite number')
    tick = round(scale * 100)
    if not 1 <= tick <= 10 or abs(scale * 100 - tick) > 1e-9:
        raise ValueError('Edit scale must be 0.01 to 0.10 in steps of 0.01')
    return dict(rtc_delay=delay, edit_scale=tick / 100.)


def suffix(values):
    values = normalize(values)
    return 'd%d-edit%03d' % (values['rtc_delay'], round(values['edit_scale'] * 100))


def configured_run(config, values):
    """Keep original defaults resumable, isolate every non-default pair."""
    from pathlib import Path
    selected = normalize(values)
    defaults = normalize({'rtc_delay': config['rtc_delay'], 'edit_scale': config['edit_scale']})
    result = {**config, **selected}
    if selected != defaults:
        root = Path(config['run_dir'])
        result['run_dir'] = str(root.parent / (root.name + '--' + suffix(selected)))
    return result
