#!/usr/bin/env python3
"""Verify faithful online resume on private copies, with no ROS or robot connection."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "third_party/openpi-rlt/rlt_online_rl/src"),
               str(ROOT / "integrations/stage1-client/src")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing directories are refused")
    args = parser.parse_args()
    sources = {name: getattr(args, name).resolve() for name in ("checkpoint", "replay", "norm")}
    for path in sources.values():
        if not path.is_file():
            parser.error("Missing source: " + str(path))
    out = args.output.resolve()
    if out.exists():
        parser.error("Output already exists; use a fresh validation directory")
    hashes = {name: digest(path) for name, path in sources.items()}
    out.mkdir(parents=True)
    (out / "checkpoints").mkdir()
    shutil.copy2(sources["checkpoint"], out / "checkpoints/latest.pkl")
    shutil.copy2(sources["replay"], out / "replay.pkl")
    shutil.copy2(sources["norm"], out / "action_norm_stats.json")

    import numpy as np
    import yaml
    from rlt_online_rl.config import system_config_from_mapping
    from rlt_online_rl.replay import ReplayManager
    from rlt_online_rl.trainer import LearnerService

    config = yaml.safe_load((ROOT / "configs/rlt/plug_v3_yyshadow/online_rl.yaml").read_text())
    config["experiment"]["rl"]["action_norm_stats_path"] = str(out / "action_norm_stats.json")
    cfg = system_config_from_mapping(config)
    service = dataclasses.replace(cfg.learner_service,
        checkpoint_dir=str(out / "checkpoints"),
        actor_snapshot_path=str(out / "actor_snapshot/actor_snapshot.pkl"))
    replay = ReplayManager(cfg.replay.capacity, journal_path=str(out / "replay.pkl"),
                           seed=cfg.replay.seed, sample_strategy=cfg.replay.sample_strategy)
    learner = LearnerService(cfg.rl, service, replay, metrics_path=str(out / "metrics/learner.jsonl"))
    initial = {"step": int(learner.state.global_step), "actor": int(learner.state.actor_version)}
    assert initial == {"step": 5000, "actor": 2500}, initial
    assert replay.stats()["adds_total"] == 2567, replay.stats()
    assert learner.train_once() is None, "Existing Replay unexpectedly triggered warmup updates"
    assert int(learner.state.global_step) == 5000
    with (out / "replay.pkl").open("rb") as stream:
        transition = pickle.load(stream)
    transition["episode_id"] = replay.stats()["max_episode_id"] + 1
    transition["collection_phase"] = "online"
    replay.add_transition(transition)
    updates = []
    for _ in range(5):
        metrics = learner.train_once()
        assert metrics is not None
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        updates.append(metrics)
    assert learner.train_once() is None, "Update budget was exceeded"
    final = {"step": int(learner.state.global_step), "actor": int(learner.state.actor_version)}
    assert final == {"step": 5005, "actor": 2502}, final
    learner.flush_artifacts()
    restored = LearnerService(cfg.rl, service, replay, metrics_path=str(out / "metrics/restart.jsonl"))
    assert int(restored.state.global_step) == 5005
    assert restored.train_once() is None
    assert hashes == {name: digest(path) for name, path in sources.items()}, "Source files changed"
    report = {"ok": True, "initial": initial, "final": final, "updates": len(updates),
              "new_transition_simulated": 1, "restart_stable": True, "robot_publishers": 0,
              "sources": {name: str(path) for name, path in sources.items()},
              "source_sha256": hashes, "output": str(out)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
