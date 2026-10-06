"""CPU-only numeric trace audit; missing measurements never become zeros.

Per-file complete Episode is the statistical unit. Physical publication receipts
and logical20 macro rows are different series. Position/split identities must be
supplied externally and are never inferred from failure or trajectory thirds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

DIMENSIONS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
SCALES = np.array([1000.] * 7)  # radians -> mrad; gripper metres -> mm


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value):
    if value is None:
        return None
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if x.shape == (14,):
        x = x[7:14]
    return x if x.shape == (7,) and np.isfinite(x).all() else None


def number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def mean_vectors(values):
    return (np.mean(values, axis=0) * SCALES).tolist() if values else None


def scalar_summary(values):
    if not values:
        return None
    return dict(n=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def audit_episode(path, identity=None):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty trace: " + str(path))
    final = rows[-1]
    issues = []
    if identity and identity.get("sha256") != sha256(path):
        raise ValueError("Trace identity hash mismatch: " + str(path))
    split = (identity or {}).get("split", "unassigned")
    if split not in {"train", "development", "independent_test", "unassigned"}:
        raise ValueError("Unknown split")
    condition = (identity or {}).get("condition")
    if split == "independent_test" and not (identity or {}).get("independent_test_provenance"):
        issues.append("Independent test declared without provenance; treated as unassigned")
        split = "unassigned"
    outcome = final.get("outcome")
    complete = bool(final.get("done")) and outcome in {"success", "failure"}
    aborted = outcome == "aborted"
    hil = any(bool(r.get("human_controlled")) or r.get("source") in [2, 3] for r in rows)
    outcome_class = ("assisted_success" if hil else "autonomous_success") if complete and outcome == "success" else "failure" if complete else "aborted" if aborted else "unlabelled_or_incomplete"
    versions = sorted({r.get("actor_param_version") for r in rows
                       if isinstance(r.get("actor_param_version"), int) and r.get("actor_param_version") >= 0})
    if len(versions) > 1:
        issues.append("Actor changed inside Episode; not frozen")
    if any(r.get("replay_eligible") is not False for r in rows):
        issues.append("Trace has no verified evaluation-only Replay exclusion")
    if any(r.get("trace_purpose") != "evaluation_diagnostic" for r in rows):
        issues.append("Not an opt-in evaluation diagnostic trace")
    settings = {json.dumps(r["execution_settings"], sort_keys=True) for r in rows if r.get("execution_settings")}
    if len(settings) > 1:
        issues.append("Execution configuration changed inside Episode")
    models = sorted({r.get("deployment_model_id") for r in rows if r.get("deployment_model_id")})
    if len(models) != 1:
        issues.append("Single actual model ID missing")
    if identity and identity.get("actor_version") is not None and versions != [identity["actor_version"]]:
        issues.append("Actual Actor versions do not match condition manifest")
    uid = final.get("task5_episode_uuid") or (identity or {}).get("episode_uuid")
    if not uid and final.get("session_id") is not None and final.get("session_episode_id") is not None:
        uid = str(final["session_id"]) + ":" + str(final["session_episode_id"])
    if not uid:
        issues.append("Stable complete Episode identity absent")
    actor_ref, command_proposal, end_tracking = [], [], []
    physical, logical, events = [], [], []
    inference_ms, publish_duration_ms, lateness_ms, feedback_age_ms = [], [], [], []
    missing = dict(actor_proposal_rows=0, independent_reference_rows=0,
                   end_feedback_rows=0, scheduled_publications=0)
    for index, row in enumerate(rows):
        human = bool(row.get("human_controlled")) or row.get("source") in [2, 3]
        action = vector(row.get("action"))
        proposal = vector(row.get("planned_action"))
        # HIL ref_action can be replaced by executed feedback/command. It must
        # never be interpreted as the unexecuted VLA alternative.
        ref = vector(row.get("logical_ref_action", row.get("ref_action"))) if not human else None
        feedback = vector(row.get("next_observation", {}).get("state"))
        if proposal is None:
            missing["actor_proposal_rows"] += 1
        if ref is None:
            missing["independent_reference_rows"] += 1
        if not human and proposal is not None and ref is not None:
            actor_ref.append(np.abs(proposal - ref))
        if not human and action is not None and proposal is not None:
            command_proposal.append(np.abs(action - proposal))
        end_time = number(row.get("sample_received_monotonic"))
        pubs = row.get("publications") or []
        if not pubs and number(row.get("command_publish_finished_monotonic")) is not None and not human:
            pubs = [dict(action=row.get("action"),
                         publish_started_monotonic=row.get("command_publish_started_monotonic"),
                         publish_finished_monotonic=row.get("command_publish_finished_monotonic"),
                         execution_epoch="synchronous_unassigned")]
        for item in pubs:
            command = vector(item.get("action"))
            t = number(item.get("publish_finished_monotonic", item.get("monotonic_timestamp")))
            start = number(item.get("publish_started_monotonic"))
            scheduled = number(item.get("scheduled_monotonic"))
            received = number(item.get("feedback_received_monotonic"))
            if command is None or t is None:
                issues.append("Invalid physical command receipt")
                continue
            epoch = item.get("execution_epoch")
            physical.append(dict(t=t, action=command.tolist(), row=index, epoch=epoch,
                                 actor_version=item.get("actor_param_version", row.get("actor_param_version")),
                                 publish_hz=row.get("publish_hz")))
            if start is not None:
                if t < start:
                    issues.append("Publish completion precedes start")
                else:
                    publish_duration_ms.append((t-start)*1000)
            if start is not None and scheduled is not None:
                lateness_ms.append((start-scheduled)*1000)
            else:
                missing["scheduled_publications"] += 1
            if received is not None and start is not None:
                if start < received:
                    issues.append("Feedback receipt occurs after publication starts")
                else:
                    feedback_age_ms.append((start-received)*1000)
        if not human and feedback is not None and pubs and end_time is not None:
            last_command = vector(pubs[-1].get("action"))
            last_time = number(pubs[-1].get("publish_finished_monotonic", pubs[-1].get("monotonic_timestamp")))
            if last_command is not None and last_time is not None and end_time >= last_time:
                end_tracking.append(np.abs(feedback-last_command))
            else:
                missing["end_feedback_rows"] += 1
        else:
            missing["end_feedback_rows"] += 1
        logical.append(dict(row=index, t=end_time, action=None if action is None else action.tolist(),
            proposal=None if proposal is None else proposal.tolist(),
            reference=None if ref is None else ref.tolist(),
            feedback=None if feedback is None else feedback.tolist(), human=human))
        for event in row.get("inference_events") or []:
            events.append(event)
            start = number(event.get("request_started_monotonic"))
            end = number(event.get("request_finished_monotonic"))
            if start is not None and end is not None and end >= start:
                inference_ms.append((end-start)*1000)
    intervals, velocities, velocity_segments = [], [], []
    for edge, (left, right) in enumerate(zip(physical, physical[1:])):
        dt = right["t"]-left["t"]
        # Never join a HIL/pause/replan epoch or an unrecorded command gap.
        hz = number(right["publish_hz"])
        connected = (left["epoch"] is not None and left["epoch"] == right["epoch"]
                     and right["row"]-left["row"] <= 1 and dt > 0
                     and (hz is None or dt <= 1.5/hz))
        if connected:
            intervals.append(dt*1000)
            v = (np.array(right["action"])-left["action"])/dt
            velocities.append(np.abs(v))
            velocity_segments.append((v, right["t"], right["epoch"], edge))
        elif dt <= 0:
            issues.append("Non-increasing publication receipt time")
    flips = np.zeros(7, int)
    for left, right in zip(velocity_segments, velocity_segments[1:]):
        if left[2] != right[2] or right[3] != left[3] + 1:
            continue
        # Ignore numerical sign flips smaller than .1mrad/s or .1mm/s.
        active = (np.abs(left[0]) > 1e-4) & (np.abs(right[0]) > 1e-4)
        flips += active & (left[0]*right[0] < 0)
    return dict(path=str(path), sha256=sha256(path), episode_identity=uid,
        split=split, condition=condition, measured_offset_mm=(identity or {}).get("measured_offset_mm"),
        outcome_class=outcome_class, complete_episode=complete, hil_present=hil,
        actor_versions=versions, model_ids=models, rows=len(rows), physical_publications=len(physical),
        runtime_provenance=rows[0].get("runtime_provenance"),
        execution_settings=[json.loads(s) for s in sorted(settings)],
        issues=sorted(set(issues)), missing=missing,
        metrics=dict(actor_reference_mae=mean_vectors(actor_ref),
                     command_queue_target_mae=mean_vectors(command_proposal),
                     endpoint_feedback_command_mae=mean_vectors(end_tracking),
                     publication_interval_ms=scalar_summary(intervals),
                     publication_call_duration_ms=scalar_summary(publish_duration_ms),
                     scheduled_start_lateness_ms=scalar_summary(lateness_ms),
                     feedback_receipt_to_publish_start_ms=scalar_summary(feedback_age_ms),
                     inference_request_duration_ms=scalar_summary(inference_ms),
                     command_speed_mean=mean_vectors(velocities),
                     command_direction_flips=flips.tolist() if velocity_segments else None,
                     rejected_deadline_miss_count=None),
        series=dict(logical=logical, physical=physical, inference_events=events))


def episode_interval(values, seed=42):
    # Complete Episodes, not ticks/windows. Single Episode gets no CI.
    values = np.asarray(values, float)
    if not len(values):
        return None
    mean = np.mean(values, axis=0)
    ci = None
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        means = np.mean(values[rng.integers(0, len(values), (2000, len(values)))], axis=1)
        ci = np.percentile(means, [2.5, 97.5], axis=0).tolist()
    return dict(n_episodes=len(values), mean=mean.tolist(), episode_bootstrap95=ci)


def audit_traces(paths, identities=None):
    identities = identities or {}
    episodes = [audit_episode(path, identities.get(str(Path(path).resolve()))) for path in paths]
    hashes = [e["sha256"] for e in episodes]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate trace snapshots; not independent Episodes")
    known = [e["episode_identity"] for e in episodes if e["episode_identity"]]
    if len(known) != len(set(known)):
        raise ValueError("Duplicate Episode identities; never count repeated snapshots as independent")
    groups = {}
    for e in episodes:
        key = e["split"] + ":" + str(e["condition"]) + ":" + e["outcome_class"]
        groups.setdefault(key, []).append(e)
    summary = {}
    for key, group in groups.items():
        entry = dict(n_episodes=len(group), complete=sum(e["complete_episode"] for e in group), metrics={})
        for metric in ["actor_reference_mae", "command_queue_target_mae", "endpoint_feedback_command_mae"]:
            eligible = [e["metrics"][metric] for e in group if e["complete_episode"] and e["metrics"][metric] is not None]
            entry["metrics"][metric] = episode_interval(eligible)
        summary[key] = entry
    return dict(status="verified" if episodes and all(not e["issues"] for e in episodes) else "insufficient_evidence",
        capability_verdict="insufficient_evidence", dimensions=DIMENSIONS,
        dimension_units=["mrad"]*6+["mm"], groups=summary, episodes=episodes,
        boundaries=["Numeric diagnostic only; no visual target inference/TCP/contact proof",
            "Cached/queued proposal is not fresh same-state HIL counterfactual",
            "Endpoint feedback residual has finite reaction delay, not settled tracking accuracy",
            "Success label is operator supplied; no autonomous gain or Online convergence proven",
            "Missing rejected deadline trace remains null; emitted lateness is not all deadline misses",
            "Physical publications are not independent statistical samples"])
