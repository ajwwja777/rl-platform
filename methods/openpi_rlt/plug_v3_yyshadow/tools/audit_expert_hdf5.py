#!/usr/bin/env python3
"""Audit plug_v3 expert HDF5s and emit a deterministic training selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_TOPICS = ("camera_high", "camera_left", "camera_right")
REQUIRED_VALID = ("action", "qpos", "camera_high", "camera_left", "camera_right")


def contiguous_bad_suffix(mask: np.ndarray) -> int | None:
    bad = np.flatnonzero(~mask)
    if not len(bad):
        return len(mask)
    first = int(bad[0])
    return first if np.array_equal(bad, np.arange(first, len(mask))) else None


def audit(path: Path, sidecar_root: Path, min_frames: int) -> dict:
    issues: list[str] = []
    with h5py.File(path, "r") as episode:
        index = int(episode.attrs["episode_index"])
        episode_uuid = str(episode.attrs["episode_uuid"])
        completion = str(episode.attrs["completion_state"])
        action = episode["action"][:]
        qpos = episode["observations/qpos"][:]
        frames = len(action)
        data_valid = np.ones(frames, dtype=bool)
        for key in REQUIRED_VALID:
            data_valid &= episode[f"rollout/valid_mask/{key}"][:].astype(bool)
        data_valid &= np.isfinite(action[:, 7:14]).all(axis=1)
        data_valid &= np.isfinite(qpos[:, 7:14]).all(axis=1)
        right_teach = episode["rollout/teach_active_right"][:].astype(bool)
        right_action_qpos_error_max = float(
            np.nanmax(np.abs(action[:, 7:13] - qpos[:, 7:13]))
        )
        sample_timestamp = episode["rollout/sample_timestamp"][:]
        frame_index = episode["rollout/frame_index"][:]
        camera_timestamp = np.stack(
            [episode[f"rollout/topic_timestamp/{key}"][:] for key in CAMERA_TOPICS],
            axis=1,
        )
        camera_skew_p99_ms = float(np.percentile(np.ptp(camera_timestamp, axis=1), 99) * 1000)
        finite_left_qpos = np.isfinite(qpos[:, :7]).all(axis=1)
        finite_left_action = np.isfinite(action[:, :7]).all(axis=1)
        left_qpos_range = float(np.ptp(qpos[finite_left_qpos, :7], axis=0).max())
        left_action_range = float(np.ptp(action[finite_left_action, :7], axis=0).max())
        if completion != "complete":
            issues.append("hdf5_not_complete")
        if action.shape != (frames, 14) or qpos.shape != (frames, 14):
            issues.append("raw_shape_not_14d")
        if not np.array_equal(frame_index, np.arange(frames)):
            issues.append("frame_index_not_contiguous")
        if frames < min_frames or not np.all(np.diff(sample_timestamp) > 0):
            issues.append("timeline_invalid")
        if camera_skew_p99_ms > 120:
            issues.append("camera_skew")
        if left_qpos_range > 1e-6 or left_action_range > 1e-6:
            issues.append("fixed_left_arm_changed")
        # Healthy recordings in this cohort stay below 0.034 rad.  A much
        # larger one-frame mismatch is a recorder/command glitch rather than
        # demonstrated motion and must not enter behavior cloning.
        if right_action_qpos_error_max > 0.1:
            issues.append("right_action_state_disagreement")
        for camera in CAMERAS:
            if episode[f"observations/images/{camera}"].shape != (frames, 480, 640, 3):
                issues.append(f"camera_shape:{camera}")

    sidecar_path = sidecar_root / episode_uuid / "sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    if sidecar.get("commit_state") != "complete" or int(sidecar.get("training_frame_count", -1)) != frames:
        issues.append("sidecar_invalid")

    selected_end = frames
    # A single post-release frame can be fully usable and is common at the end
    # of segmented demonstrations.  Teach state is therefore diagnostic, while
    # cropping is reserved for genuinely invalid sensor/action data.
    right_teach_ratio = float(np.mean(right_teach)) if frames else 0.0
    last_teach = int(np.flatnonzero(right_teach)[-1]) if np.any(right_teach) else -1
    if right_teach_ratio < 0.90:
        issues.append("right_teach_coverage_low")
    if last_teach >= 0 and np.any(~right_teach[: last_teach + 1]):
        issues.append("right_teach_interrupted")

    bad_suffix = contiguous_bad_suffix(data_valid)
    if bad_suffix is None:
        issues.append("invalid_training_frame_inside_episode")
    elif bad_suffix < frames:
        if bad_suffix < min_frames:
            issues.append("valid_prefix_too_short")
        else:
            selected_end = bad_suffix

    rejected = bool(issues)
    return {
        "episode_index": index,
        "episode_uuid": episode_uuid,
        "source": str(path),
        "raw_frames": frames,
        "selected_frame_start": 0,
        "selected_frame_end_exclusive": None if rejected else selected_end,
        "selected_frames": 0 if rejected else selected_end,
        "trimmed_suffix_frames": 0 if rejected else frames - selected_end,
        "right_arm_indices": list(range(7, 14)),
        "implicit_outcome": "success",
        "camera_skew_p99_ms": camera_skew_p99_ms,
        "right_teach_ratio": right_teach_ratio,
        "right_action_qpos_error_max": right_action_qpos_error_max,
        "left_qpos_range_max": left_qpos_range,
        "left_action_range_max": left_action_range,
        "decision": "reject" if rejected else ("accept_trim_suffix" if selected_end < frames else "accept"),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=60)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    episodes = [audit(path, root / ".segments", args.min_frames) for path in sorted(root.glob("episode_*.hdf5"))]
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "task_contract": "single_right_arm_7d_three_cameras",
        "expert_label_rule": "complete validated expert episodes are implicitly success",
        "raw_episode_count": len(episodes),
        "accepted_count": sum(item["decision"].startswith("accept") for item in episodes),
        "rejected_count": sum(item["decision"] == "reject" for item in episodes),
        "trimmed_count": sum(item["decision"] == "accept_trim_suffix" for item in episodes),
        "accepted_frame_count": sum(
            int(item["selected_frames"])
            for item in episodes
            if item["decision"].startswith("accept")
        ),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("raw_episode_count", "accepted_count", "rejected_count", "trimmed_count")}, indent=2))
    for item in episodes:
        if item["decision"] != "accept":
            print(item["episode_index"], item["decision"], item["issues"], item["selected_frame_end_exclusive"])


if __name__ == "__main__":
    main()
