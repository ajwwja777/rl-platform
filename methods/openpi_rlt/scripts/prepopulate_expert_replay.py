#!/usr/bin/env python3
"""Validate an expert replay prepopulation plan without fabricating features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from methods.openpi_rlt.scripts.build_plug_training_release import atomic_write_json


def build_prepopulate_payload(plan: dict, *, min_transitions: int, max_updates: int) -> dict:
    if plan.get("status") != "planned-not-materialized":
        raise ValueError("transition plan has not passed frozen-release validation")
    transition_count = int(plan.get("transition_count", 0))
    if transition_count < int(min_transitions):
        raise ValueError(
            f"expert transition count {transition_count} is below warmup minimum {min_transitions}"
        )
    if max_updates <= 0 or max_updates > 2_000:
        raise ValueError("warmup update cap must be in [1, 2000]")
    return {
        "schema_version": 1,
        "status": "prepared-not-materialized",
        "dataset_id": plan.get("dataset_id"),
        "source_manifest_sha256": plan.get("source_manifest_sha256"),
        "unique_episodes": int(plan.get("unique_episodes", 0)),
        "transition_count": transition_count,
        "chunk_len": int(plan.get("chunk_len", 0)),
        "stride": int(plan.get("stride", 0)),
        "source": "expert",
        "warmup_update_cap": int(max_updates),
        "materialization_gate": (
            "Run Machine A feature extraction against the frozen data release, then verify "
            "z_rl/ref_chunk/schema before appending to a new replay journal."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-plan", required=True, type=Path)
    parser.add_argument("--min-transitions", required=True, type=int)
    parser.add_argument("--max-updates", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    plan = json.loads(args.transition_plan.read_text(encoding="utf-8"))
    payload = build_prepopulate_payload(
        plan, min_transitions=args.min_transitions, max_updates=args.max_updates
    )
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

