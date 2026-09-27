#!/usr/bin/env python3
"""Serve the frozen Cobot RLT representation/VLA policy (Machine A)."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

from methods.openpi_rlt.cobot_adapter.machine_a import (
    ACTION_DIM,
    CHUNK_LEN,
    PROPRIO_DIM,
    configure_serve_module,
)
from methods.openpi_rlt.scripts.stage1 import inspect_inputs
from methods.openpi_rlt.stage1_entry import prepare_environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--default-prompt",
        default="Open the pot lid, put the object into the pot, then close the lid.",
    )
    parser.add_argument(
        "--no-shared-prefix-inference",
        action="store_false",
        dest="shared_prefix_inference",
        default=True,
    )
    return parser


def _build_config(args: argparse.Namespace):
    upstream = args.upstream_root.resolve()
    if str(upstream / "src") not in sys.path:
        sys.path.insert(0, str(upstream / "src"))
    from methods.openpi_rlt.stage1_config import build_stage1_config

    return build_stage1_config(
        exp_name="machine-a",
        dataset_repo_id=args.dataset_root.resolve().name,
        base_params=args.base_params,
        assets_base_dir=args.project_root.resolve() / "assets" / "openpi-rlt",
        checkpoint_base_dir=args.project_root.resolve() / "checkpoints" / "openpi-rlt",
        batch_size=32,
        num_train_steps=5_000,
        num_workers=0,
        fsdp_devices=1,
        rlt_alpha=1.0,
    )


def _validate_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint = checkpoint_dir.resolve()
    if not (checkpoint / "params").is_dir():
        raise ValueError(f"checkpoint params directory is missing: {checkpoint / 'params'}")
    return checkpoint


def _load_runtime_and_config(args: argparse.Namespace):
    """Load the inference runtime before the training config.

    The legacy Cobot π0.5 environment crashes inside Arrow's jemalloc thread
    when the training-config import precedes ``serve_rlt_policy``.  The reverse
    order is deterministic on that host and does not alter either module.
    """
    upstream = args.upstream_root.resolve()
    scripts = upstream / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    source = upstream / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("serve_rlt_policy")
    return module, _build_config(args)


def main() -> None:
    args = _parser().parse_args()
    project = args.project_root.resolve()
    upstream = args.upstream_root.resolve()
    dataset = args.dataset_root.resolve()
    checkpoint = _validate_checkpoint(args.checkpoint_dir)
    prepare_environment(dataset, project)
    inputs = inspect_inputs(upstream, dataset)
    if not inputs["upstream_clean"]:
        raise ValueError("fixed openpi-RLT upstream is dirty")
    if args.dry_run:
        config = _build_config(args)
        spec = {
            "upstream_commit": inputs["upstream_commit"],
            "checkpoint_dir": str(checkpoint),
            "config_name": config.name,
            "proprio_dim": PROPRIO_DIM,
            "action_dim": ACTION_DIM,
            "chunk_len": CHUNK_LEN,
            "port": args.port,
            "shared_prefix_inference": args.shared_prefix_inference,
        }
        print(json.dumps(spec, indent=2))
        return

    module, config = _load_runtime_and_config(args)
    configure_serve_module(module, config)
    module.main(
        module.Args(
            config=config.name,
            checkpoint_dir=str(checkpoint),
            port=args.port,
            default_prompt=args.default_prompt,
            shared_prefix_inference=args.shared_prefix_inference,
        )
    )


if __name__ == "__main__":
    main()
