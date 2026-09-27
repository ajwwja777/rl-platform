#!/usr/bin/env python3
"""Project-owned entrypoint around the fixed openpi-RLT Stage-1 source."""

import argparse
import dataclasses
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_UPSTREAM_COMMIT = "c1e40ac360185778c98cf20da2820e22d2d415e7"
PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

from methods.openpi_rlt.stage1_entry import prepare_environment  # noqa: E402
from methods.openpi_rlt.stage1_entry import run_upstream_train  # noqa: E402
from methods.openpi_rlt.stage1_entry import validate_dataset_root  # noqa: E402


def _git(upstream_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(upstream_root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def inspect_inputs(upstream_root: str | Path, dataset_root: str | Path) -> dict[str, object]:
    upstream = Path(upstream_root).resolve()
    dataset = validate_dataset_root(dataset_root)
    commit = _git(upstream, "rev-parse", "HEAD")
    if commit != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError(f"unexpected openpi-RLT commit: {commit}")
    status = _git(upstream, "status", "--porcelain")
    info = json.loads((dataset / "meta" / "info.json").read_text())
    features = info["features"]
    camera_keys = [key for key in features if key.startswith("observation.images.")]
    return {
        "upstream_root": str(upstream),
        "upstream_commit": commit,
        "upstream_clean": status == "",
        "dataset_root": str(dataset),
        "dataset_id": dataset.name,
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "fps": info["fps"],
        "state_shape": features["observation.state"]["shape"],
        "action_shape": features["action"]["shape"],
        "camera_keys": camera_keys,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--project-root", required=True, type=Path)
    validate.add_argument("--upstream-root", required=True, type=Path)
    validate.add_argument("--dataset-root", required=True, type=Path)
    for name in ("stats", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--project-root", required=True, type=Path)
        command.add_argument("--upstream-root", required=True, type=Path)
        command.add_argument("--dataset-root", required=True, type=Path)
        command.add_argument("--base-params", required=True)
        command.add_argument("--exp-name", required=True)
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--num-workers", type=int, default=8)
        command.add_argument("--fsdp-devices", type=int, default=1)
        command.add_argument("--num-train-steps", type=int, default=5_000)
        command.add_argument("--rlt-alpha", type=float, default=1.0)
        command.add_argument("--debug-model", action="store_true")
        run_mode = command.add_mutually_exclusive_group()
        run_mode.add_argument("--resume", action="store_true")
        run_mode.add_argument("--overwrite", action="store_true")
        if name == "stats":
            command.add_argument("--max-frames", type=int)
    return parser


def _build_config(args: argparse.Namespace):
    upstream = Path(args.upstream_root).resolve()
    sys.path.insert(0, str(upstream / "src"))
    from methods.openpi_rlt.stage1_config import build_stage1_config

    project = Path(args.project_root).resolve()
    return build_stage1_config(
        exp_name=args.exp_name,
        dataset_repo_id=Path(args.dataset_root).resolve().name,
        base_params=args.base_params,
        assets_base_dir=project / "assets" / "openpi-rlt",
        checkpoint_base_dir=project / "checkpoints" / "openpi-rlt",
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        rlt_alpha=args.rlt_alpha,
        debug_model=args.debug_model,
    )


def _run_spec(command: str, config) -> dict[str, object]:
    return {
        "command": command,
        "config_name": config.name,
        "dataset_repo_id": config.data.repo_id,
        "base_params": getattr(config.weight_loader, "params_path", None),
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "fsdp_devices": config.fsdp_devices,
        "num_train_steps": config.num_train_steps,
        "rlt_alpha": config.rlt_alpha,
        "assets_base_dir": config.assets_base_dir,
        "checkpoint_base_dir": config.checkpoint_base_dir,
        "exp_name": config.exp_name,
        "debug_model": config.model.paligemma_variant == "dummy",
    }


def _load_upstream_script(upstream_root: Path, name: str):
    scripts = upstream_root / "scripts"
    path = scripts / f"{name}.py"
    if not path.is_file():
        raise RuntimeError(f"cannot load fixed upstream script: {path}")
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module(name)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate":
        prepare_environment(args.dataset_root, args.project_root)
        print(json.dumps(inspect_inputs(args.upstream_root, args.dataset_root), indent=2))
        return

    prepare_environment(args.dataset_root, args.project_root)
    inputs = inspect_inputs(args.upstream_root, args.dataset_root)
    if not inputs["upstream_clean"]:
        raise ValueError("fixed openpi-RLT upstream is dirty")
    if args.command == "stats":
        os.environ["JAX_PLATFORMS"] = "cpu"
    config = _build_config(args)
    if args.dry_run:
        print(json.dumps(_run_spec(args.command, config), indent=2))
        return
    if args.command == "stats":
        module = _load_upstream_script(Path(args.upstream_root).resolve(), "compute_norm_stats")
        module._config._CONFIGS_DICT[config.name] = config
        module.main(config.name, args.max_frames)
        return
    if args.command == "train":
        config = dataclasses.replace(config, resume=args.resume, overwrite=args.overwrite)
        module = _load_upstream_script(Path(args.upstream_root).resolve(), "train_rlt")
        run_upstream_train(module, config, Path(args.project_root) / "cache" / "jax")
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
