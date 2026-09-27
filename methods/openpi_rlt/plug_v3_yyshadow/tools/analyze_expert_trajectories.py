#!/usr/bin/env python3
"""Summarize right-arm expert trajectory consistency from an audit selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def robust_outliers(values: np.ndarray, threshold: float = 4.0) -> list[int]:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, np.inf)
    score = np.max(np.abs(values - median) / scale, axis=1)
    return np.flatnonzero(score > threshold).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    rows: list[dict] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for item in selection["episodes"]:
        if not item["decision"].startswith("accept"):
            continue
        end = int(item["selected_frame_end_exclusive"])
        with h5py.File(item["source"], "r") as episode:
            action = np.asarray(episode["action"][:end, 7:14], dtype=np.float64)
            qpos = np.asarray(episode["observations/qpos"][:end, 7:14], dtype=np.float64)
            timestamps = np.asarray(episode["rollout/sample_timestamp"][:end], dtype=np.float64)
        dt = np.diff(timestamps)
        action_step = np.diff(action[:, :6], axis=0)
        qpos_step = np.diff(qpos[:, :6], axis=0)
        starts.append(qpos[0])
        ends.append(qpos[-1])
        rows.append(
            {
                "episode_index": int(item["episode_index"]),
                "frames": end,
                "duration_sec": float(timestamps[-1] - timestamps[0]),
                "sample_hz": float(1.0 / np.median(dt)),
                "joint_path_length_rad": float(np.linalg.norm(qpos_step, axis=1).sum()),
                "joint_step_p99_rad": float(np.percentile(np.abs(qpos_step), 99)),
                "joint_step_max_rad": float(np.abs(qpos_step).max()),
                "action_step_p99_rad": float(np.percentile(np.abs(action_step), 99)),
                "action_step_max_rad": float(np.abs(action_step).max()),
                "tracking_abs_error_p99_rad": float(np.percentile(np.abs(action[:, :6] - qpos[:, :6]), 99)),
                "gripper_action_range": float(np.ptp(action[:, 6])),
                "gripper_qpos_range": float(np.ptp(qpos[:, 6])),
                "start_qpos": starts[-1].tolist(),
                "end_qpos": ends[-1].tolist(),
            }
        )

    starts_array = np.stack(starts)
    ends_array = np.stack(ends)
    scalar_keys = (
        "duration_sec",
        "sample_hz",
        "joint_path_length_rad",
        "joint_step_p99_rad",
        "joint_step_max_rad",
        "action_step_p99_rad",
        "action_step_max_rad",
        "tracking_abs_error_p99_rad",
        "gripper_action_range",
        "gripper_qpos_range",
    )
    summary = {}
    for key in scalar_keys:
        values = np.asarray([row[key] for row in rows])
        summary[key] = {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    features = np.asarray(
        [
            [
                row["duration_sec"],
                row["joint_path_length_rad"],
                row["joint_step_p99_rad"],
                row["tracking_abs_error_p99_rad"],
            ]
            for row in rows
        ]
    )
    outlier_rows = robust_outliers(features)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_selection": str(args.selection.resolve()),
        "episode_count": len(rows),
        "right_arm_dim": 7,
        "summary": summary,
        "start_qpos_range": np.ptp(starts_array, axis=0).tolist(),
        "end_qpos_range": np.ptp(ends_array, axis=0).tolist(),
        "robust_trajectory_outlier_episodes": [rows[index]["episode_index"] for index in outlier_rows],
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("episode_count", "start_qpos_range", "end_qpos_range", "robust_trajectory_outlier_episodes")}, indent=2))


if __name__ == "__main__":
    main()
