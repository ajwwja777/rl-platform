#!/usr/bin/env python3
"""Prepare, but never launch, a π0.5/RLT plug-insertion training run."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from methods.openpi_rlt.scripts.build_plug_training_release import atomic_write_json


def _q(path: str | Path) -> str:
    return shlex.quote(str(Path(path).resolve()))


def build_run_manifest(
    *,
    dataset_manifest: Path,
    dataset_root: Path,
    project_root: Path,
    upstream_root: Path,
    base_params: Path,
    run_id: str,
    full_steps: int,
) -> dict:
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if dataset.get("status") != "frozen":
        raise ValueError("dataset manifest must be frozen before preparing a training run")
    if not run_id or "/" in run_id or full_steps <= 2:
        raise ValueError("run_id and full_steps are invalid")
    script = project_root / "methods/openpi_rlt/scripts/stage1.py"
    common = (
        f"{_q(script)} train --project-root {_q(project_root)} "
        f"--upstream-root {_q(upstream_root)} --dataset-root {_q(dataset_root)} "
        f"--base-params {_q(base_params)} --exp-name {shlex.quote(run_id)} "
        "--batch-size 32 --num-workers 8 --fsdp-devices 4 --rlt-alpha 1.0"
    )
    return {
        "schema_version": 1,
        "status": "prepared-not-trained",
        "run_id": run_id,
        "dataset": {
            "manifest": str(dataset_manifest.resolve()),
            "release_sha256": dataset.get("release_sha256"),
            "dataset_id": dataset.get("dataset_id"),
            "total_episodes": dataset.get("total_episodes"),
            "total_frames": dataset.get("total_frames"),
            "task_prompt": dataset.get("task_prompt"),
        },
        "stage1": {
            "initialization": "fixed_pi05_base",
            "train_scope": "joint_vlm_and_rlt_token",
            "rlt_alpha": 1.0,
            "seed": 42,
            "full_steps_candidate": int(full_steps),
            "freeze_after_validation": ["vlm", "rlt_encoder", "rl_token"],
        },
        "stage2": {
            "train_scope": "actor_critic_only",
            "actor_version_policy": "fixed_for_whole_episode",
            "evaluation": "deterministic",
            "exploration": "linear_endpoints_chunk_noise_active_arm_only",
        },
        "commands": {
            "validate": common.replace(" train ", " validate ", 1).split(" --base-params", 1)[0],
            "smoke": f"{common} --num-train-steps 2 --overwrite",
            "full": f"{common} --num-train-steps {int(full_steps)} --overwrite",
        },
        "launch_gate": [
            "real dataset files match the frozen manifest",
            "single batch is finite and camera/action schema is audited",
            "smoke checkpoint saves and restores",
            "throughput probe fixes micro/global batch before full training",
            "user authorizes the GPU training run",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--base-params", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-steps", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_run_manifest(
        dataset_manifest=args.dataset_manifest.resolve(),
        dataset_root=args.dataset_root.resolve(),
        project_root=args.project_root.resolve(),
        upstream_root=args.upstream_root.resolve(),
        base_params=args.base_params.resolve(),
        run_id=args.run_id,
        full_steps=args.full_steps,
    )
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

