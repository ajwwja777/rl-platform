#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def load_json(name: str) -> dict:
    path = ROOT / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_reproduction(contract: dict) -> list[str]:
    errors: list[str] = []
    upstream = contract["upstream"]
    repo = Path(upstream["relay_clone"])
    if not repo.is_dir():
        return [f"missing upstream clone: {repo}"]
    try:
        head = git_output(repo, "rev-parse", "HEAD")
        status = git_output(repo, "status", "--short")
    except subprocess.CalledProcessError as exc:
        return [f"git inspection failed: {exc.output}"]
    if head != upstream["commit"]:
        errors.append(f"upstream HEAD mismatch: {head}")
    if status:
        errors.append("upstream clone is dirty")
    for relative, expected in upstream["files"].items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"missing upstream file: {relative}")
        elif sha256(path) != expected:
            errors.append(f"upstream hash mismatch: {relative}")

    rl = contract["online_rl"]
    expected = {
        "action_dim": 7,
        "proprio_dim": 7,
        "chunk_len": 10,
        "z_dim": 2048,
        "gamma": 0.99,
        "fixed_std": 0.002,
        "reference_dropout_prob": 0.5,
        "delta_weight": 10.0,
        "warmup_bc_weight": 10.0,
        "warmup_q_weight": 0.1,
        "online_bc_weight": 5.0,
        "online_q_weight": 0.1,
        "actor_lr": 0.0001,
        "critic_lr": 0.0001,
        "target_tau": 0.005,
        "warmup_min_size": 600,
        "warmup_post_collect_updates": 20000,
        "grad_updates_per_cycle": 5,
        "sample_batch_size": 128,
        "control_frequency_hz": 20.0,
        "step_trace_stride": 0,
    }
    for key, value in expected.items():
        if rl.get(key) != value:
            errors.append(f"locked online parameter mismatch: {key}={rl.get(key)!r}")
    return errors


def validate_scene(scene: dict, require_reference: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    pending: list[str] = []
    if scene.get("experiment_id") != "plug_v3_yyshadow":
        errors.append("scene experiment_id mismatch")
    required_views = scene.get("camera_contract", {}).get("required_views")
    if required_views != ["mid", "left", "right"]:
        errors.append("camera views must be mid/left/right")
    if scene.get("camera_contract", {}).get("excluded_views") != []:
        errors.append("no selected camera may be excluded")
    if len(scene.get("state_contract", {}).get("raw_platform_evidence_order", [])) != 14:
        errors.append("optional raw platform evidence order must contain 14 dimensions")
    if len(scene.get("state_contract", {}).get("training_proprio_order", [])) != 7:
        errors.append("training proprio order must contain 7 right-arm dimensions")
    if len(scene.get("state_contract", {}).get("online_active_order", [])) != 7:
        errors.append("online active order must contain 7 dimensions")

    if scene.get("reference_status") != "captured":
        pending.append("onsite scene reference has not been read back and verified")
    geometry = scene.get("fixed_geometry", {})
    for key in (
        "right_gripper_plug_depth_marked",
        "right_gripper_plug_yaw_marked",
        "power_strip_orientation_marked",
        "camera_mounts_locked",
        "lighting_profile_recorded",
    ):
        if geometry.get(key) is not True:
            pending.append(key)
    if geometry.get("arm_home_pose_name") in (None, "pending_onsite_confirmation"):
        pending.append("arm_home_pose_name")
    if geometry.get("mid_camera_pose_name") in (None, "pending_onsite_confirmation"):
        pending.append("mid_camera_pose_name")
    cameras = scene.get("camera_contract", {}).get("reference_images", {})
    for view in ("mid", "left", "right"):
        if not cameras.get(view):
            pending.append(f"reference image: {view}")
    distribution = scene.get("scene_distribution", {})
    if not distribution.get("position_ids"):
        pending.append("scene position_ids")
    if not distribution.get("nominal_position_id"):
        pending.append("nominal_position_id")
    if not distribution.get("heldout_position_ids"):
        pending.append("heldout_position_ids")
    evidence = scene.get("reference_evidence", {})
    for key in ("captured_at", "operator", "initial_state_file", "scene_photos_manifest"):
        if not evidence.get(key):
            pending.append(f"reference_evidence.{key}")
    if require_reference and pending:
        errors.extend(f"pending: {item}" for item in pending)
    return errors, sorted(set(pending))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-scene-reference", action="store_true")
    args = parser.parse_args()
    reproduction = load_json("reproduction_contract.json")
    scene = load_json("scene_contract.json")
    errors = validate_reproduction(reproduction)
    scene_errors, pending = validate_scene(scene, args.require_scene_reference)
    errors.extend(scene_errors)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("PASS stage0 upstream identity, hashes, isolation, and locked defaults")
    print("PASS stage1 scene contract schema")
    if pending:
        print("PENDING onsite scene reference:")
        for item in pending:
            print(f"- {item}")
    else:
        print("PASS onsite scene reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())

