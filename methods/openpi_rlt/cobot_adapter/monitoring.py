"""Dependency-light, read-only RLT chart contract; no model/ROS imports."""
import json
import math
from pathlib import Path
import time
from collections import deque


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def tail_rows(path, limit=600, byte_limit=2_000_000):
    try:
        with Path(path).open('rb') as f:
            f.seek(0, 2)
            start = max(0, f.tell()-byte_limit)
            f.seek(start)
            if start:
                f.readline()
            lines = f.read(byte_limit).splitlines()
        rows = []
        for line in lines:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except ValueError:
                continue  # A concurrently appended partial line is not a sample.
        return rows[-limit:]
    except OSError:
        return []


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def training_series(root):
    root = Path(root)
    status = read_json(root/'metrics/learner_status.json')
    points = {}
    boundary = status.get('warmup_required_updates')
    for row in tail_rows(root/'metrics/learner_metrics.jsonl'):
        if not all(finite(row.get(k)) for k in ('global_step', 'q1_mean', 'target_q_mean', 'critic_loss')):
            continue
        step = int(row['global_step'])
        # Restoration may leave a longer historical log: don't show future steps.
        if step > status.get('global_step', step):
            continue
        if finite(boundary) and step <= boundary:
            continue
        points[step] = dict(critic_step=step, q=row['q1_mean'], target_q_mean=row['target_q_mean'],
                            critic_loss=row['critic_loss'], actor_version=row.get('actor_version'),
                            timestamp=row.get('timestamp'))
    result = sorted(points.values(), key=lambda p: p['critic_step'])
    return dict(points=result, updated_at=result[-1]['timestamp'] if result else None,
                warmup_end_step=boundary, scope='online', q_semantics='q1_mean'), status


def success_series(rows):
    # Only latched collection episodes enter this series. Warmup/evaluation do not.
    unique = {p['episode_uuid']: p for p in rows if p.get('episode_uuid') and
              p.get('collection_phase') == 'online' and p.get('session_use') == 'collection'}
    ordered = sorted(unique.values(), key=lambda p: (p.get('finished_at', 0), p['episode_uuid']))
    window, points, excluded = deque(maxlen=20), [], 0
    for index, row in enumerate(ordered, 1):
        if row.get('outcome') not in ('success', 'failure') or not row.get('count') or row.get('hil_steps') is None:
            excluded += 1
            continue
        window.append(row)
        success = sum(p['outcome'] == 'success' and p['hil_steps'] == 0 for p in window)
        points.append(dict(episode=index, rate=100.*success/len(window), successes=success,
                           total=len(window), assisted=sum(p['hil_steps'] > 0 for p in window),
                           actor_version=row.get('actor_version'), episode_uuid=row['episode_uuid']))
    return dict(points=points[-600:], completed=len(ordered), excluded=excluded)


def read_history(root):
    """Called in Web's background pool; cache/polling is a Web responsibility."""
    root = Path(root)
    training, status = training_series(root)
    rows = tail_rows(root/'telemetry/episodes.jsonl', limit=100000, byte_limit=16_000_000)
    rows.extend(read_json(p) for p in (root/'telemetry/episodes').glob('*.json'))
    known = {p.get('episode_uuid') for p in rows}
    # Legacy finalized collection traces contain true outcomes and HIL flags.
    for path in (root/'traces').glob('*.jsonl'):
        if '.pending.' in path.name or not path.name.endswith(('_success.jsonl', '_failure.jsonl', '_aborted.jsonl')):
            continue
        trace = tail_rows(path, limit=100000, byte_limit=32_000_000)
        if not trace:
            continue
        last = trace[-1]
        uid = last.get('task5_episode_uuid') or last.get('episode_uuid') or path.stem
        if uid in known:
            continue
        hil = sum(bool(p['human_controlled']) for p in trace) if all('human_controlled' in p for p in trace) else None
        rows.append(dict(episode_uuid=uid, outcome=last.get('outcome'), count=len(trace), hil_steps=hil,
                         collection_phase=str(last.get('collection_phase', last.get('phase', 'unknown'))).split(':', 1)[0], session_use='collection',
                         finished_at=path.stat().st_mtime, actor_version=last.get('actor_param_version')))
    age = time.time()-status.get('timestamp', 0)
    return dict(training=training, success=success_series(rows),
                learner=dict(supported=True, connected=True, running=bool(0 <= age < 15),
                             phase='running' if 0 <= age < 15 else 'stale',
                             learner_version=status.get('global_step'), observed_at=status.get('timestamp')),
                rollout=read_json(root/'telemetry/rollout.json'))
