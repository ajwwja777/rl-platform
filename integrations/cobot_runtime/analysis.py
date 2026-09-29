"""Read-only RLT telemetry aggregation; no JAX, ROS or model imports."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

MAX_BYTES = 24 * 1024 * 1024
SERIES_KEYS = ("global_step", "timestamp", "actor_version", "critic_loss", "actor_loss",
    "did_actor_update", "q1_mean", "q2_mean", "target_q_mean", "weighted_bc", "weighted_q",
    "weighted_delta", "human_mask_ratio", "sample_recent_online_ratio",
    "sample_warmup_demo_ratio", "sample_human_intervention_ratio",
    "sample_source_base_ratio", "sample_source_rl_ratio", "sample_source_human_ratio",
    "sample_source_mixed_ratio")

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value

def read_json(path):
    try:
        with Path(path).open() as f: return clean(json.load(f))
    except (OSError, ValueError): return {}

def read_rows(path):
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > MAX_BYTES:
                f.seek(size - MAX_BYTES)
                f.readline()
            raw = f.read(MAX_BYTES).splitlines()
        rows, invalid = [], 0
        for line in raw:
            try:
                item = json.loads(line)
                if isinstance(item, dict): rows.append(clean(item))
                else: invalid += 1
            except (ValueError, UnicodeDecodeError): invalid += 1
        return rows, {"path": str(path), "bytes": size, "truncated": size > MAX_BYTES,
                      "invalid_lines": invalid, "rows": len(rows), "mtime": path.stat().st_mtime}
    except OSError as exc:
        return [], {"path": str(path), "error": str(exc), "rows": 0}

def segments(rows):
    groups = []
    for row in rows:
        if not finite(row.get("global_step")): continue
        if not groups or row["global_step"] <= groups[-1][-1]["global_step"]:
            groups.append([])
        groups[-1].append(row)
    return groups

def decimate(rows, limit=320):
    if len(rows) <= limit: return rows
    return [rows[round(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)]

def wilson(successes, count):
    if not count: return [None, None]
    z = 1.96
    p = successes / count
    scale = 1 + z*z/count
    center = (p + z*z/(2*count))/scale
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/scale
    return [max(0., center-half), min(1., center+half)]

def episode_groups(rows):
    grouped, accepted, excluded = defaultdict(list), [], 0
    seen = set()
    for row in rows:
        identity = (row.get("timestamp"), row.get("episode_id"))
        if identity in seen: continue
        seen.add(identity)
        if not finite(row.get("transitions_written")) or row["transitions_written"] <= 0:
            excluded += 1
            continue
        if row.get("success") not in (0, 1, False, True): continue
        item = dict(row)
        item["assisted"] = bool(row.get("human_steps", 0) or row.get("mixed_steps", 0)
                                or row.get("intervention_count", 0))
        item["version"] = (str(row.get("actor_version_start", "?")) if
            row.get("actor_version_start") == row.get("actor_version_end") else
            str(row.get("actor_version_start", "?")) + "→" + str(row.get("actor_version_end", "?")))
        item["cohort"] = str(row.get("collection_phase") or row.get("phase") or "unknown") + (
            " / deterministic" if row.get("actor_deterministic") else " / stochastic")
        item["key"] = str(row.get("timestamp")) + ":" + str(row.get("episode_id"))
        grouped[(item["cohort"], item["version"])].append(item)
        accepted.append(item)
    groups = []
    for (cohort, version), values in grouped.items():
        n = len(values)
        autonomous = sum(bool(x["success"]) and not x["assisted"] for x in values)
        assisted = sum(bool(x["success"]) and x["assisted"] for x in values)
        groups.append({"cohort": cohort, "version": version, "count": n,
            "autonomous_successes": autonomous, "assisted_successes": assisted,
            "failures": n-autonomous-assisted, "hil_count": sum(x["assisted"] for x in values),
            "autonomous_rate": autonomous/n, "autonomous_ci": wilson(autonomous,n),
            "first_timestamp": values[0].get("timestamp"), "last_timestamp": values[-1].get("timestamp")})
    return groups, accepted, excluded

def snapshot(run_root, config_path, run=-1):
    root = Path(run_root)
    metrics = root / "online/metrics"
    learner, ls = read_rows(metrics / "learner_metrics.jsonl")
    rollouts, rs = read_rows(metrics / "rollout_metrics.jsonl")
    runs = segments(learner)
    index = run if 0 <= run < len(runs) else len(runs)-1
    selected = runs[index] if runs else []
    actors = [x for x in selected if x.get("did_actor_update") == 1]
    recent = selected[-100:]
    averages = {}
    for key in SERIES_KEYS:
        values = [x[key] for x in recent if finite(x.get(key))]
        if values: averages[key] = sum(values)/len(values)
    groups, episodes, excluded = episode_groups(rollouts)
    config, config_error = {}, None
    try:
        import yaml
        config = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception as exc: config_error = str(exc)
    status = read_json(metrics / "learner_status.json")
    age = max(0, time.time()-status["timestamp"]) if finite(status.get("timestamp")) else None
    projection = read_json(root / "analysis/replay_projection.json")
    batches, batch_source = read_rows(metrics / "batch_composition.jsonl")
    batch_views = [{k: v for k,v in row.items() if k not in ("identities", "cross")} for row in batches[-128:]]
    replay_composition = read_json(root / "analysis/replay_composition.json")
    diagnosis = read_json(root / "analysis/learning_diagnosis.json")
    sensitivity = read_json(root / "analysis/rl_sensitivity.json")
    return clean({"schema": 1, "generated_at": time.time(), "status": status,
        "status_age_sec": age, "stale": age is None or age > 30,
        "selected_run": index, "runs": [{"id": i, "start": s[0]["global_step"],
            "end": s[-1]["global_step"], "rows": len(s), "timestamp": s[-1].get("timestamp")}
            for i,s in enumerate(runs)],
        "series": [{k: x[k] for k in SERIES_KEYS if finite(x.get(k))} for x in decimate(selected)],
        "actor_series": [{k: x[k] for k in SERIES_KEYS if finite(x.get(k))} for x in decimate(actors)],
        "recent_means": averages, "recent_count": len(recent),
        "episode_groups": groups, "episodes": episodes[-500:], "episode_count": len(episodes),
        "excluded_uncommitted": excluded, "config": config, "config_path": str(config_path),
        "config_error": config_error, "sources": [ls, rs, batch_source], "projection": projection, "replay_composition": replay_composition, "batches": batch_views, "learning_diagnosis": diagnosis, "sensitivity": sensitivity})
