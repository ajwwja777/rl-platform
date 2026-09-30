#!/usr/bin/env python3
"""Deterministic Stage-1 checkpoint comparison on fixed plug_v3 episode cohorts."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch


def add_roots(project: Path, upstream: Path) -> None:
    for path in (project, upstream / "src", upstream / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def parse_episodes(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            result.extend(range(first, last + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def make_loader(config, dataset_root: Path, episodes: list[int], batches: int):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from openpi import transforms
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    metadata = LeRobotDatasetMetadata(data.repo_id, root=dataset_root)
    base_dataset = LeRobotDataset(
        data.repo_id,
        root=dataset_root,
        delta_timestamps={
            key: [step / metadata.fps for step in range(config.model.action_horizon)]
            for key in data.action_sequence_keys
        },
    )
    indices: list[int] = []
    for episode in episodes:
        start = int(base_dataset.episode_data_index["from"][episode])
        stop = int(base_dataset.episode_data_index["to"][episode])
        indices.extend(range(start, stop))
    dataset = torch.utils.data.Subset(base_dataset, indices)
    if data.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)])
    dataset = data_loader.transform_dataset(dataset, data)
    # Finite, non-repeating evaluation, including the final partial batch.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=data_loader._collate_fn,
    )
    return data_loader.DataLoaderImpl(data, loader), len(dataset)


class EvaluationModel(nnx.Module):
    def __init__(self, inner):
        self.inner = inner

    def evaluate(self, rng, observation, actions):
        total, info = self.inner.compute_rlt_loss(rng, observation, actions, 1.0, train=False)
        sampled = self.inner.vla.sample_actions(rng, observation, num_steps=10)
        first = sampled[:, :10, :7]
        target = actions[:, :10, :7]
        delta = first[:, 1:] - first[:, :-1]
        accel = delta[:, 1:] - delta[:, :-1]
        return {
            "total_loss": total,
            "vla_loss": info["vla_loss"],
            "rlt_loss": info["rlt_loss"],
            "rlt_mse": info["mse"],
            "action_mae_10_norm": jnp.mean(jnp.abs(first - target)),
            "action_mse_10_norm": jnp.mean(jnp.square(first - target)),
            "action_mae_per_dim_norm": jnp.mean(jnp.abs(first - target), axis=(0, 1)),
            "action_mse_per_dim_norm": jnp.mean(jnp.square(first - target), axis=(0, 1)),
            "sample_velocity_norm": jnp.mean(jnp.abs(delta)),
            "sample_acceleration_norm": jnp.mean(jnp.abs(accel)),
        }


def evaluate_cohort(model, loader, batches: int, seed: int) -> dict[str, float]:
    from methods.openpi_rlt.plug_v3_yyshadow.evaluation_summary import aggregate_batches

    evaluate = __import__("openpi.shared.nnx_utils", fromlist=["module_jit"]).module_jit(model.evaluate)
    rows = []
    started = time.perf_counter()
    for index, (observation, actions) in enumerate(loader):
        metrics = evaluate(jax.random.fold_in(jax.random.key(seed), index), observation, actions)
        rows.append({"samples": int(actions.shape[0]), "metrics": {
            key: np.asarray(value).tolist() for key, value in jax.device_get(metrics).items()
        }})
        if batches and len(rows) >= batches:
            break
    return {
        **aggregate_batches(rows),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=0, help="Per-Episode cap; 0 evaluates every frame. Capped results have no complete-Episode CI.")
    parser.add_argument("--early-episodes", default="0-13")
    parser.add_argument("--late-episodes", default="120-133")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches < 0:
        parser.error("batch-size must be positive; batches must be nonnegative")

    project = args.project_root.resolve()
    upstream = args.upstream_root.resolve()
    dataset = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    add_roots(project, upstream)
    from methods.openpi_rlt.plug_v3_yyshadow.evaluation_summary import summarize_episodes

    from methods.openpi_rlt.stage1_entry import prepare_environment

    prepare_environment(dataset, project)
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_right_arm_config import build_config
    from openpi.models import model as model_api
    from openpi.training import config as training_config
    import serve_rlt_policy
    import train_rlt

    if not (checkpoint / "params").is_dir():
        raise ValueError(f"checkpoint params missing: {checkpoint}")
    config = build_config(
        dataset_repo_id=dataset.name,
        base_params=str(checkpoint / "params"),
        project_root=project,
        exp_name="stage1-evaluation",
        batch_size=args.batch_size,
        num_workers=0,
        fsdp_devices=1,
        num_train_steps=1,
    )
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=training_config.AssetsConfig(assets_dir=str(checkpoint / "assets")),
        ),
    )
    if config.data.create(config.assets_dirs, config.model).norm_stats is None:
        raise ValueError("normalization assets missing")

    vla = nnx.eval_shape(config.model.create, jax.random.key(0))
    inner = train_rlt.RLTTrainModel(
        vla,
        serve_rlt_policy._create_rlt_config(config),
        rngs=nnx.Rngs(jax.random.key(1)),
        prefix_seq_len=serve_rlt_policy._infer_prefix_seq_len(config.model),
    )
    graph, state = nnx.split(inner)
    loaded = model_api.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    expected = jax.tree_util.tree_structure(state.to_pure_dict())
    actual = jax.tree_util.tree_structure(loaded)
    if expected != actual:
        raise ValueError("checkpoint parameter tree does not match Stage-1 model")
    state.replace_by_pure_dict(loaded)
    model = EvaluationModel(nnx.merge(graph, state))

    cohorts = {}
    for name, episode_spec in (("early", args.early_episodes), ("late", args.late_episodes)):
        episodes = parse_episodes(episode_spec)
        per_episode = []
        for episode in episodes:
            loader, frames = make_loader(config, dataset, [episode], args.batches)
            per_episode.append({
                "episode": episode, "available_frames": frames,
                **evaluate_cohort(model, loader, args.batches, seed=42 + episode),
            })
        cohorts[name] = {
            "requested_episodes": episodes,
            "available_frames": sum(row["available_frames"] for row in per_episode),
            **summarize_episodes(per_episode),
        }

    result = {
        "schema": 2,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "note": "Both cohorts were present in Stage-1 training; metrics are consistency diagnostics, not unseen generalization.",
        "cohorts": cohorts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
