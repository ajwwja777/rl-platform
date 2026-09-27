#!/usr/bin/env python3
"""Create a deterministic dry transition plan from a frozen data manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from methods.openpi_rlt.cobot_adapter.segmented_dataset import build_transition_plan


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_transition_plan(
        args.dataset_manifest.resolve(), chunk_len=args.chunk_len, stride=args.stride
    )
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps({key: payload[key] for key in ("status", "unique_episodes", "transition_count")}))


if __name__ == "__main__":
    main()

