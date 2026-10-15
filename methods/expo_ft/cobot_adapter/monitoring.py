"""Read-only EXPO monitoring data; never changes Replay, RNG or checkpoints."""
import json
import math
import os
from pathlib import Path
import threading
import time
from collections import deque


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


class LearnerHistory:
    """One point per completed native update call (20 critic steps), not heartbeat."""
    def __init__(self, root, committed_step):
        self.path = Path(root) / 'telemetry' / 'learner.json'
        old = read_json(self.path).get('points', [])
        self.points = [p for p in old if p.get('critic_step', -1) <= committed_step][-600:]
        self.last_metrics = None

    def observe(self, value, metrics):
        if metrics is self.last_metrics or not metrics:
            return
        self.last_metrics = metrics
        names = ('q', 'target_q_mean', 'critic_loss')
        if not all(finite(metrics.get(k)) for k in names):
            return
        step = value.get('critic_updates')
        if not isinstance(step, int) or step <= 0:
            return
        point = dict(critic_step=step, actor_version=value['actor_version'],
                     timestamp=value['updated_at'], **{k: metrics[k] for k in names})
        self.points = [p for p in self.points if p['critic_step'] < step][-599:] + [point]
        write_json(self.path, self.payload())

    def payload(self):
        return dict(schema=1, points=self.points, updated_at=self.points[-1]['timestamp'] if self.points else None)


_capture = threading.local()


def install_selection_observer(module):
    """Observe existing JIT outputs. Return the exact original result; no extra RNG/Q call."""
    original = module._jitted_fast_select
    if getattr(original, '_expo_monitoring', False):
        return
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        if getattr(_capture, 'enabled', False):
            _capture.selection = result
        return result
    observed._expo_monitoring = True
    module._jitted_fast_select = observed


def infer_with_q(actor, infer, route, data):
    _capture.selection = None
    _capture.enabled = route == '/select'
    try:
        result = infer(actor, route, data)
        selection = _capture.selection
        if route == '/select' and selection is not None and selection.get('qs') is not None:
            # actions are already synchronized by native infer; copy only its selected scalar.
            try:
                q = float(selection['qs'][selection['idx']])
                if math.isfinite(q):
                    result['selected_q'] = q
            except (ValueError, TypeError, IndexError, KeyError):
                pass  # Missing diagnostic values cannot change inference behavior.
        return result
    finally:
        _capture.enabled = False
        _capture.selection = None


class RolloutHistory:
    """No disk writes in the control loop. The caller's background thread flushes."""
    def __init__(self, root):
        self.path = Path(root) / 'telemetry' / 'rollout.json'
        self.lock = threading.Lock()
        self.identity = None
        self.points = deque(maxlen=1500)
        self.dirty = False
        self.version = None

    def selected(self, active, payload):
        if not active or not finite(payload.get('selected_q')):
            return
        with self.lock:
            if active['episode_uuid'] != self.identity:
                self.identity = active['episode_uuid']
                self.version = payload['actor_version']
                self.points.clear()
            self.points.append(dict(seconds=max(0., time.time() - active['started_at']),
                                    q=payload['selected_q'], timestamp=time.time()))
            self.dirty = True

    def flush(self):
        with self.lock:
            if not self.dirty:
                return
            value = dict(schema=1, episode_uuid=self.identity, actor_version=self.version,
                         points=list(self.points))
            self.dirty = False
        write_json(self.path, value)


def episode_summary(directory):
    """Historical HIL comes from actual valid journal rows, never from an outcome label."""
    import numpy as np
    directory = Path(directory)
    value = read_json(directory / 'finished.json')
    if not value or value.get('episode_uuid') != directory.name:
        return None
    count = value.get('count')
    hil = None
    if type(count) is int and 0 < count <= 3000:
        try:
            hil = 0
            for index in range(count):
                with np.load(directory / ('%06d.npz' % index), allow_pickle=False) as row:
                    flag = row['is_hil']
                    if flag.shape != () or flag.dtype != np.dtype('bool'):
                        raise ValueError('Unknown HIL schema')
                    hil += int(flag)
        except (OSError, ValueError, KeyError):
            hil = None
    return dict(episode_uuid=directory.name, outcome=value.get('outcome'),
                started_at=value.get('started_at'), finished_at=value.get('finished_at'),
                actor_version=value.get('actor_version'), count=count, hil_steps=hil)


def success_series(episodes):
    """Last20 labeled, nonempty attempts; HIL success stays in denominator, not numerator."""
    ordered = sorted(episodes, key=lambda p: (p.get('finished_at') or 0, p['episode_uuid']))
    window = deque(maxlen=20)
    points = []
    excluded = 0
    for index, item in enumerate(ordered, 1):
        eligible = item.get('outcome') in ('success', 'failure') and (item.get('count') or 0) > 0
        if not eligible or item.get('hil_steps') is None:
            excluded += 1
            continue
        window.append(item)
        successes = sum(p['outcome'] == 'success' and p['hil_steps'] == 0 for p in window)
        points.append(dict(episode=index, rate=100. * successes / len(window), successes=successes,
                           total=len(window), assisted=sum(p['hil_steps'] > 0 for p in window),
                           actor_version=item.get('actor_version'), episode_uuid=item['episode_uuid']))
    return dict(points=points[-600:], excluded=excluded, completed=len(ordered))
