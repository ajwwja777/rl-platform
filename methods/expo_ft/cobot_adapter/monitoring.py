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
    original = getattr(module, '_jitted_fast_select', None)
    if original is None:
        return  # Legacy actors without candidate Q remain fully operational.
    if getattr(original, '_expo_monitoring', False):
        return
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        if getattr(_capture, 'enabled', False):
            _capture.selection = result
            _capture.base_count = kwargs.get('N')
        return result
    observed._expo_monitoring = True
    module._jitted_fast_select = observed


def selection_diagnostics(selection, base_count=None):
    """CPU scalars from the native selection, without forward calls or RNG access.

    Native candidates are N unedited chunks followed by edits of the first base
    chunks. Q values already use the native target-critic ensemble reduction.
    edit_norm is the full-chunk L2 norm of the selected edit minus its paired
    base, in normalized action coordinates (dimensionless, not joint radians).
    """
    import numpy as np
    if selection is None or selection.get('qs') is None:
        return {}
    qs = np.asarray(selection['qs'])
    index = np.asarray(selection.get('idx'))
    if qs.ndim != 1 or index.shape != () or index.dtype.kind not in 'iu':
        return {}
    index = int(index)
    if not 0 <= index < len(qs):
        return {}
    q = float(qs[index])
    if not math.isfinite(q):
        return {}
    value = {'selected_q': q}
    if type(base_count) is not int or not 0 < base_count <= len(qs):
        return value  # Old/native variants without an explicit base count remain compatible.
    if np.isfinite(qs[:base_count]).all():
        value['best_unedited_q'] = float(qs[:base_count].max())
    edited = index >= base_count
    value['edit_selected'] = edited
    if not edited:
        value['edit_norm'] = 0.0  # Measured choice of an unedited candidate, not missing data.
        return value
    if selection.get('cand') is None:
        return value
    candidates = np.asarray(selection['cand'])
    paired = index - base_count
    if (candidates.ndim != 3 or candidates.shape[0] != len(qs)
            or paired >= base_count or not all(candidates.shape[1:])):
        return value
    delta = candidates[index].astype(np.float64) - candidates[paired]
    norm = float(np.linalg.norm(delta.reshape(-1)))
    if math.isfinite(norm):
        value['edit_norm'] = norm
    return value


def infer_with_q(actor, infer, route, data):
    _capture.selection = None
    _capture.base_count = None
    _capture.enabled = route == '/select'
    try:
        result = infer(actor, route, data)
        if route == '/select':
            # Native infer has already synchronized its actions. Diagnostics only copy
            # the existing small candidate outputs; failure cannot affect inference.
            try:
                result.update(selection_diagnostics(_capture.selection, _capture.base_count))
            except Exception:
                pass
        return result
    finally:
        _capture.enabled = False
        _capture.selection = None
        _capture.base_count = None


def rollout_diagnostics(payload):
    """Optional finite scalars only; absent timings are never replaced with zero.

    compute_seconds is measured by the Actor around this /select inference. It
    excludes the earlier pi0.5 /precache phase and transport/queue waiting.
    wait_ms measures a completed RTC precache waiting until select begins; it
    is not robot stalling or network latency. precache_ms is separate compute.
    """
    value = {}
    if finite(payload.get('best_unedited_q')):
        value['best_unedited_q'] = payload['best_unedited_q']
    if type(payload.get('edit_selected')) is bool:
        value['edit_selected'] = payload['edit_selected']
    for key in ('edit_norm', 'inference_ms', 'wait_ms', 'precache_ms'):
        if finite(payload.get(key)) and payload[key] >= 0:
            value[key] = payload[key]
    if 'inference_ms' not in value and finite(payload.get('compute_seconds')):
        milliseconds = payload['compute_seconds'] * 1000.0
        if math.isfinite(milliseconds) and milliseconds >= 0:
            value['inference_ms'] = milliseconds
    return value


def precache_timing(payload, completed_monotonic):
    """Local per-result timing, carried with the cache and never sent to Actor."""
    value = {}
    if finite(completed_monotonic):
        value['precache_completed_monotonic'] = completed_monotonic
    compute = payload.get('compute_seconds')
    if finite(compute) and compute >= 0 and math.isfinite(compute * 1000.0):
        value['precache_ms'] = compute * 1000.0
    return value


def selection_timing(cached, select_monotonic):
    """Use this cache's completion, never a shared/previous request timestamp."""
    source = cached.get('_rollout_telemetry')
    if not isinstance(source, dict):
        return {}
    value = {}
    completed = source.get('precache_completed_monotonic')
    if finite(completed) and finite(select_monotonic) and select_monotonic >= completed:
        elapsed = (select_monotonic - completed) * 1000.0
        if math.isfinite(elapsed):
            value['wait_ms'] = elapsed
    if finite(source.get('precache_ms')) and source['precache_ms'] >= 0:
        value['precache_ms'] = source['precache_ms']
    return value


class RolloutHistory:
    """No disk writes in the control loop. The caller's background thread flushes."""
    def __init__(self, root):
        self.path = Path(root) / 'telemetry' / 'rollout.json'
        self.lock = threading.Lock()
        self.identity = None
        self.points = deque(maxlen=1500)
        self.dirty = False
        self.version = None
        self.pending = []

    def selected(self, active, payload):
        if not active or not finite(payload.get('selected_q')):
            return
        with self.lock:
            if active['episode_uuid'] != self.identity:
                if self.identity and self.points:
                    self.pending.append(dict(schema=1, episode_uuid=self.identity, actor_version=self.version, points=list(self.points)))
                self.identity = active['episode_uuid']
                self.version = payload['actor_version']
                self.points.clear()
            # Same-host monotonic time aligns future points to recorded frame
            # timestamps; old points intentionally have no synthesized anchor.
            monotonic_seconds = time.monotonic()
            timestamp = time.time()
            self.points.append(dict(seconds=max(0., timestamp - active['started_at']),
                                    q=payload['selected_q'], timestamp=timestamp,
                                    monotonic_seconds=monotonic_seconds,
                                    **rollout_diagnostics(payload)))
            self.dirty = True

    def flush(self):
        with self.lock:
            if not self.dirty:
                return
            value = dict(schema=1, episode_uuid=self.identity, actor_version=self.version,
                         points=list(self.points))
            self.dirty = False
            pending, self.pending = self.pending, []
        # All archive I/O stays on the existing telemetry flush thread.
        try:
            import runpy
            archive = runpy.run_path(str(Path(__file__).resolve().parents[3]/'integrations/cobot_runtime/rollout_archive.py'))
            for snapshot in [*pending, value]:
                archive['save_snapshot'](self.path.parent.parent, snapshot)
        except Exception:
            # Retry complete snapshots later; telemetry failure never changes control.
            with self.lock:
                self.pending = pending + [value] + self.pending
                self.dirty = True
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
