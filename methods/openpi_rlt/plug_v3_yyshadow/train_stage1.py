#!/usr/bin/env python3
"""Validate, compute stats, and train the plug_v3 faithful Stage-1 model."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_UPSTREAM_COMMIT = "c1e40ac360185778c98cf20da2820e22d2d415e7"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()


def add_import_roots(project: Path, upstream: Path) -> None:
    for path in (project, upstream / "src", upstream / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def inspect_dataset(dataset: Path) -> dict:
    from methods.openpi_rlt.stage1_entry import validate_dataset_root

    root = validate_dataset_root(dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    if features["observation.state"]["shape"] != [7] or features["action"]["shape"] != [7]:
        raise ValueError("plug_v3 Stage-1 requires 7D state and 7D action")
    required = {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    if not required.issubset(features):
        raise ValueError(f"missing camera features: {sorted(required - set(features))}")
    if int(info["fps"]) != 30:
        raise ValueError("plug_v3 dataset must be 30 Hz")
    return {
        "dataset_root": str(root),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "state_shape": features["observation.state"]["shape"],
        "action_shape": features["action"]["shape"],
        "cameras": sorted(required),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("validate", "stats", "train"))
    result.add_argument("--project-root", type=Path, required=True)
    result.add_argument("--upstream-root", type=Path, required=True)
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--base-params", required=True)
    result.add_argument("--exp-name", default="stage1-faithful-s42")
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--fsdp-devices", type=int, default=4)
    result.add_argument("--num-train-steps", type=int, default=5000)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    project = args.project_root.resolve()
    upstream = args.upstream_root.resolve()
    dataset = args.dataset_root.resolve()
    add_import_roots(project, upstream)
    if git(upstream, "rev-parse", "HEAD") != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("unexpected upstream commit")
    if git(upstream, "status", "--porcelain"):
        raise ValueError("fixed upstream worktree is dirty")
    from methods.openpi_rlt.stage1_entry import prepare_environment, run_upstream_train
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_right_arm_config import build_config

    prepare_environment(dataset, project)
    dataset_info = inspect_dataset(dataset)
    config = build_config(
        dataset_repo_id=dataset.name,
        base_params=args.base_params,
        project_root=project,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        num_train_steps=args.num_train_steps,
    )
    summary = {
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "dataset": dataset_info,
        "config_name": config.name,
        "exp_name": config.exp_name,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "fsdp_devices": config.fsdp_devices,
        "num_train_steps": config.num_train_steps,
        "action_horizon": config.model.action_horizon,
        "model_action_dim": config.model.action_dim,
        "rlt_alpha": config.rlt_alpha,
        "checkpoint_dir": str(config.checkpoint_dir),
        "assets_dir": str(config.assets_dirs),
    }
    print(json.dumps(summary, indent=2))
    if args.command == "validate":
        return
    if args.command == "stats":
        os.environ["JAX_PLATFORMS"] = "cpu"
        module = importlib.import_module("compute_norm_stats")
        module._config._CONFIGS_DICT[config.name] = config
        module.main(config.name, None)
        return
    config = dataclasses.replace(config, resume=args.resume, overwrite=args.overwrite)
    module = importlib.import_module("train_rlt")
    run_upstream_train(module, config, project / "cache" / "jax")


if __name__ == "__main__":
    main()
