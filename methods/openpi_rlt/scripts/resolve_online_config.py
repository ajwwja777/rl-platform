#!/usr/bin/env python3
"""Resolve all Cobot online runtime outputs into one registered project root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

RUN_RELATIVE = Path("runs/openpi-rlt/online-r1-legacy40-v2.1")


def _validate_run_relative(run_relative: Path) -> Path:
    if run_relative.is_absolute() or ".." in run_relative.parts:
        raise ValueError("run-relative must stay below the project root")
    return run_relative


def build_runtime_payload(
    template: Path,
    project_root: Path,
    *,
    run_relative: Path = RUN_RELATIVE,
) -> dict:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("online config must contain a mapping")
    project = project_root.resolve()
    run = project / _validate_run_relative(run_relative)
    experiment = payload["experiment"]
    runtime = payload["runtime"]
    experiment["rl"]["action_norm_stats_path"] = str(
        project / "methods/openpi_rlt/configs/stats/legacy40-v2.1-action-delta-chunk10.json"
    )
    snapshot = str(run / "actor_snapshot/actor_snapshot.pkl")
    runtime["actor_service"]["snapshot_path"] = snapshot
    runtime["learner_service"]["checkpoint_dir"] = str(run / "checkpoints")
    runtime["learner_service"]["actor_snapshot_path"] = snapshot
    runtime["replay"]["journal_path"] = str(run / "replay/replay_journal.pkl")
    runtime["monitoring"]["wandb_dir"] = str(run / "wandb")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-relative", default=RUN_RELATIVE, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_runtime_payload(
        args.template.resolve(),
        args.project_root.resolve(),
        run_relative=args.run_relative,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
