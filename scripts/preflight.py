#!/usr/bin/env python3
"""Read-only validation of the registered RLT release; never publishes to ROS."""
import argparse
import json
import pickle
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COHORT = "plug_v3_yyshadow"

def check(root=ROOT):
    root = Path(root).resolve()
    config_dir = root / "configs/rlt" / COHORT
    config = yaml.safe_load((config_dir / "online_rl.yaml").read_text())
    manifest = json.loads((config_dir / "manifest.json").read_text())
    model_root = Path(manifest.get("model_root", root / "models/rlt" / COHORT))
    checkpoint = Path(manifest.get("stage1_root", model_root / ("reference_4999" if manifest.get("model_root") else "stage1/4999"))).resolve()
    warmup = model_root / ("warmup_5000" if manifest.get("model_root") else "warmup-5000")
    # The manifest is a site deployment registry. Refuse accidental cross-release use.
    if Path(manifest["checkpoint"]).resolve() != checkpoint:
        raise ValueError("manifest points outside this release: " + manifest["checkpoint"])
    required = [
        Path(manifest.get("tokenizer_home", str(root / "cache/openpi"))) / "big_vision/paligemma_tokenizer.model",
        checkpoint / "params",
        checkpoint / "assets/plug_v3_yyshadow_demonstrations/norm_stats.json",
        warmup / "actor_snapshot/actor_snapshot.pkl",
        warmup / "action_norm_stats.json",
        root / "third_party/openpi-rlt/rlt_online_rl/scripts/run_online_rl.py",
    ]
    import runpy
    guard = runpy.run_path(str(root.parent / 'cobot-control/robot/asset_storage.py'))['require_storage']
    guard(model_root, write=True)
    runtime = config["runtime"]
    rl = config["experiment"]["rl"]
    paths = {
        "actor": runtime["actor_service"]["snapshot_path"],
        "learner": runtime["learner_service"]["checkpoint_dir"] + "/latest.pkl",
        "replay": runtime["replay"]["journal_path"],
        "normalization": rl["action_norm_stats_path"],
    }
    resolved = {key: (config_dir / value).resolve() for key, value in paths.items()}
    for path in resolved.values():
        guard(path, write=True)
    for path in required + list(resolved.values()):
        if not path.exists():
            raise FileNotFoundError(str(path))
    # Only trusted local checkpoints produced by this project are read.
    with resolved["learner"].open("rb") as stream:
        saved = pickle.load(stream)
    state = saved["state"]
    step = int(state["global_step"])
    actor_version = int(state["actor_version"])
    warmup_anchor = saved.get("progress", {}).get("warmup_ready_adds_total")
    if step < int(rl["warmup_post_collect_updates"]) or warmup_anchor is None:
        raise ValueError("online resume requires a completed warmup checkpoint")
    expected = {"action_dim": 7, "chunk_len": 10, "z_dim": 2048,
                "proprio_dim": 7, "warmup_post_collect_updates": 5000}
    if any(rl[key] != value for key, value in expected.items()):
        raise ValueError("registered plug_v3 contract changed; review release before use")
    return {"ok": True, "project": str(root), "checkpoint_step": step,
            "actor_version": actor_version, "warmup_ready_adds_total": int(warmup_anchor),
            "paths": {key: str(value) for key, value in resolved.items()},
            "stage1_checkpoint": str(checkpoint), "robot_publishers": 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.root), ensure_ascii=False, indent=2))
    except Exception as error:
        raise SystemExit("RLT preflight failed: " + str(error))
