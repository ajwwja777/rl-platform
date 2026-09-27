#!/usr/bin/env python3
"""Isolated plug_v2 data preparation and matched pi05/RLT Stage-1 entry."""
import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "code/openpi-rlt"
DATA = ROOT / "data/rlt/plug_v2/demonstrations/lerobot"
RUN = ROOT / "runs/plug_v2"
EXPECTED_COMMIT = "c1e40ac360185778c98cf20da2820e22d2d415e7"

def imports(cpu=False):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(ROOT), str(UPSTREAM / "src"), str(UPSTREAM / "scripts")]
    from methods.openpi_rlt.stage1_entry import prepare_environment
    prepare_environment(DATA, RUN)
    # pyarrow/torch before JAX: preserve the verified native-library import order.
    import pyarrow.parquet
    import torch
    from openpi.training import data_loader
    from methods.openpi_rlt.stage1_config import build_stage1_config
    return data_loader, build_stage1_config

def selection():
    payload = json.loads((DATA / "selection_manifest.json").read_text())
    assert payload["schema_version"] == "cobot-lerobot-training-selection-v1"
    assert payload["status"] == "frozen" and payload["action_horizon"] == 50
    entries = payload["episodes"]
    train = {e["source_uuid"] for e in entries if e["split"] == "train"}
    val = {e["source_uuid"] for e in entries if e["split"] == "val"}
    assert not train & val
    assert len(train) == 74 and len(val) == 8
    assert len(entries) == 83 and sum(e["frame_count"] for e in entries) == 9119
    for i, e in enumerate(entries):
        assert e["episode_index"] == i
        anchors = e["anchor_frames"]
        assert len(set(anchors)) == len(anchors)
        assert all(0 <= a and a + 50 <= e["frame_count"] for a in anchors)
    return payload

def inspect():
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert commit == EXPECTED_COMMIT
    assert not subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    assert (ROOT / "cache/openpi-assets/pi05_base/params").is_dir()
    expected = json.loads((ROOT / "configs/plug_v2/transfer_checksums.json").read_text())
    actual = {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DATA.rglob("*") if p.is_file()}
    assert actual == expected, "materialized dataset checksums differ"
    manifest = selection()
    return {"upstream_commit": commit, "dataset": str(DATA), "files": len(actual),
            "original_episodes": 82, "derived_episodes": 83, "valid_frames": 9119,
            "splits": {split: {"source_episodes": len({e["source_uuid"] for e in manifest["episodes"] if e["split"] == split}),
                              "anchors": sum(len(e["anchor_frames"]) for e in manifest["episodes"] if e["split"] == split)}
                       for split in ["train", "val"]}}

def config(profile, *, steps=4000, workers=8, batch=32):
    _, builder = imports()
    cfg = builder(exp_name=profile, dataset_repo_id=DATA.name,
                  base_params=str(ROOT / "cache/openpi-assets/pi05_base/params"),
                  assets_base_dir=RUN / "assets/openpi-rlt",
                  checkpoint_base_dir=RUN / "checkpoints",
                  batch_size=batch, num_train_steps=steps, num_workers=workers,
                  fsdp_devices=4, rlt_alpha=1.0, seed=42)
    # alpha=0 in train_rlt freezes VLA; use train.py for the pi05-only baseline.
    data = dataclasses.replace(cfg.data, assets=dataclasses.replace(cfg.data.assets, assets_dir=str(RUN / "assets/openpi-rlt")))
    return dataclasses.replace(cfg, name="plug_v2_" + profile, keep_period=1999, data=data)

def specification(profile, cfg):
    return {"profile": profile, "entry": "train" if profile == "pi05_reference" else "train_rlt",
            "config_name": cfg.name, "seed": cfg.seed, "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers, "fsdp_devices": cfg.fsdp_devices,
            "num_train_steps": cfg.num_train_steps, "action_horizon": cfg.model.action_horizon,
            "physical_action_dim": 14, "model_action_dim": cfg.model.action_dim,
            "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "lr_schedule": dataclasses.asdict(cfg.lr_schedule),
            "optimizer": dataclasses.asdict(cfg.optimizer), "ema_decay": cfg.ema_decay,
            "base_params": cfg.weight_loader.params_path,
            "checkpoint_dir": str(cfg.checkpoint_dir), "norm_stats": str(RUN / "assets/openpi-rlt/lerobot/norm_stats.json"),
            "keep_period": cfg.keep_period, "save_interval": cfg.save_interval,
            "checkpoint_labels": {"phase1_2000_updates": 1999, "phase2_resume_to_4000_updates": 3999},
            "execution_plan": "train --steps 2000, review, then train --resume --steps 4000",
            "rlt_alpha": cfg.rlt_alpha if profile == "rlt_stage1" else None,
            "rlt_loss": "token reconstruction + alpha * VLA action loss" if profile == "rlt_stage1" else None,
            "note": "No online RL/critic/warmup is run by these Stage-1 entries."}

def prepare():
    print("PREPARE inventory", flush=True)
    summary = inspect()
    print("PREPARE imports", flush=True)
    data_loader, _ = imports(cpu=True)
    import numpy as np
    import pyarrow.parquet as pq
    from openpi import transforms
    from openpi.shared import normalize
    from methods.openpi_rlt.cobot_adapter.upstream_data import build_cobot_transform_groups
    from methods.openpi_rlt.plug_v2.selected_anchor import install_selected_anchor_patch
    cfg = config("rlt_stage1")
    print("PREPARE numeric statistics", flush=True)
    repack, groups = build_cobot_transform_groups(cfg.model)
    stats = {k: normalize.RunningStats() for k in ["state", "actions"]}
    comparisons = 0
    fake_images = {k: np.zeros((480, 640, 3), np.uint8) for k in
                   ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]}
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    delta = transforms.DeltaActions(mask)
    for e in selection()["episodes"]:
        if e["split"] != "train" or not e["anchor_frames"]:
            continue
        i = e["episode_index"]
        frame = pq.read_table(DATA / f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet")
        state = np.asarray(frame["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(frame["action"].to_pylist(), dtype=np.float32)
        anchors = np.asarray(e["anchor_frames"], dtype=np.int64)
        batch = {"state": state[anchors], "actions": action[anchors[:, None] + np.arange(50)].copy()}
        delta(batch)
        # Compare the numeric fast path to the actual configured data transforms.
        sample = {"images": fake_images, "state": state[anchors[0]].copy(),
                  "actions": action[anchors[0]:anchors[0] + 50].copy(), "prompt": "Insert the held plug into the socket."}
        for transform in groups.inputs:
            sample = transform(sample)
        np.testing.assert_array_equal(sample["state"], batch["state"][0])
        np.testing.assert_array_equal(sample["actions"], batch["actions"][0])
        comparisons += 1
        for key in stats:
            assert np.isfinite(batch[key]).all()
            stats[key].update(batch[key])
    norm_path = RUN / "assets/openpi-rlt/lerobot"
    norm_path.mkdir(parents=True, exist_ok=True)
    computed = {k: v.get_statistics() for k, v in stats.items()}
    if (norm_path / "norm_stats.json").exists():
        assert (norm_path / "norm_stats.json").read_text() == normalize.serialize_json(computed), "existing statistics differ; refusing replacement"
    else:
        normalize.save(norm_path, computed)
    loaded = normalize.load(norm_path)
    print("PREPARE official reader", flush=True)
    for key, value in loaded.items():
        assert value.mean.shape == (14,)
        assert np.isfinite(value.mean).all() and np.isfinite(value.std).all()
        assert np.isfinite(value.q01).all() and np.isfinite(value.q99).all()
    install_selected_anchor_patch(data_loader, DATA / "selection_manifest.json", split="train")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_config.norm_stats is not None, "configured stats were not loaded"
    dataset = data_loader.create_torch_dataset(data_config, 50, cfg.model)
    assert len(dataset) == summary["splits"]["train"]["anchors"]
    for idx in sorted(set([0, len(dataset) // 2, len(dataset) - 1])):
        item = dataset[idx]
        assert tuple(item["action"].shape) == (50, 14)
        for key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            assert tuple(item["observation.images." + key].shape) == (3, 480, 640)
        assert np.isfinite(np.asarray(item["observation.state"])).all()
    summary.update(status="prepared-not-trained", train_only_stats=True,
                   numeric_transform_comparisons=comparisons,
                   reader_checked_indices=[0, len(dataset) // 2, len(dataset) - 1],
                   norm_stats_sha256=hashlib.sha256((norm_path / "norm_stats.json").read_bytes()).hexdigest())
    for profile in ["pi05_reference", "rlt_stage1"]:
        spec = specification(profile, config(profile))
        (ROOT / f"configs/plug_v2/{profile}.json").write_text(json.dumps(spec, indent=2))
    (RUN / "preparation.json").write_text(json.dumps(summary, indent=2))
    print("PREPARE_SUCCESS " + json.dumps(summary), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "dry-run", "train", "probe"])
    parser.add_argument("profile", nargs="?", choices=["pi05_reference", "rlt_stage1"], default="rlt_stage1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, choices=[2000, 4000], default=2000)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    cfg = config(args.profile, steps=20 if args.command == "probe" else args.steps)
    if args.command == "probe":
        cfg = dataclasses.replace(cfg, exp_name="probe-" + time.strftime("%Y%m%dT%H%M%S"))
    if args.command == "dry-run":
        print(json.dumps(specification(args.profile, cfg), indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU training must run inside the authorized allocation")
    import jax
    assert len(jax.devices("gpu")) == 4, "expected four allocated GPUs"
    inspect()
    assert (RUN / "preparation.json").is_file(), "prepare and review first"
    from openpi.training import data_loader
    from methods.openpi_rlt.plug_v2.selected_anchor import install_selected_anchor_patch
    from methods.openpi_rlt.stage1_entry import run_upstream_train
    install_selected_anchor_patch(data_loader, DATA / "selection_manifest.json", split="train")
    cfg = dataclasses.replace(cfg, resume=args.resume, overwrite=False)
    if not args.resume and cfg.checkpoint_dir.exists():
        raise RuntimeError("run already exists; explicit --resume required")
    module = importlib.import_module("train" if args.profile == "pi05_reference" else "train_rlt")
    run_upstream_train(module, cfg, RUN / "cache/jax")

if __name__ == "__main__":
    main()
