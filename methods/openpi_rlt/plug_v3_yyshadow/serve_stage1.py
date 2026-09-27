#!/usr/bin/env python3
"""Serve the plug_v3 single-right-arm Stage-1 RLT policy without robot publishers."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


PROMPT = "Insert the plug held by the right gripper into the socket."


def initialize(project: Path):
    upstream = project / "third_party" / "openpi-rlt"
    sys.path[:0] = [str(project), str(upstream / "src"), str(upstream / "scripts")]
    # PyArrow/Torch must be initialized before JAX in this mixed Cobot
    # environment.  The inverse order can crash in libarrow while the
    # LeRobot data config is being constructed.
    import pyarrow  # noqa: F401
    import torch  # noqa: F401
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import flax.nnx.bridge as nnx_bridge
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_right_arm_config import build_config
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models.rl_token import RLTokenConfig, RLTokenModel
    from openpi.shared import nnx_utils
    from openpi.training import config as training_config

    return (
        jax,
        jnp,
        nnx,
        nnx_bridge,
        RLTokenConfig,
        RLTokenModel,
        build_config,
        transforms,
        model_api,
        nnx_utils,
        training_config,
    )


def load(project: Path, checkpoint: Path, *, num_steps: int = 10):
    checkpoint = checkpoint.resolve()
    asset_id = "plug_v3_yyshadow_demonstrations"
    norm_file = checkpoint / "assets" / asset_id / "norm_stats.json"
    if not (checkpoint / "params").is_dir() or not norm_file.is_file():
        raise ValueError(f"incomplete Stage-1 release: {checkpoint}")

    print("MODEL_LOAD initialize", flush=True)
    (
        jax,
        jnp,
        nnx,
        nnx_bridge,
        RLTokenConfig,
        RLTokenModel,
        builder,
        transforms,
        model_api,
        nnx_utils,
        training_config,
    ) = initialize(project)
    print("MODEL_LOAD config", flush=True)
    config = builder(
        dataset_repo_id=asset_id,
        base_params=str(checkpoint / "params"),
        project_root=project,
        exp_name="plug-v3-serve",
        batch_size=1,
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
    print("MODEL_LOAD data", flush=True)
    data = config.data.create(config.assets_dirs, config.model)
    norm = data.norm_stats
    if norm is None:
        raise ValueError("normalization assets are required")

    rlt_kwargs = {}
    if config.rlt_num_tokens is not None:
        rlt_kwargs["num_rl_tokens"] = config.rlt_num_tokens
    if config.rlt_num_layers is not None:
        rlt_kwargs["num_layers"] = config.rlt_num_layers
    if config.rlt_embed_dim is not None:
        rlt_kwargs["embed_dim"] = config.rlt_embed_dim
    if config.rlt_input_dim is not None:
        rlt_kwargs["input_dim"] = config.rlt_input_dim
    rlt_config = RLTokenConfig(**rlt_kwargs)

    def infer_prefix_len(model_config):
        def get_embeddings(rng):
            candidate = model_config.create(rng)
            observation = model_config.fake_obs(batch_size=1)
            embeddings, _ = candidate.extract_prefix_embeddings(rng, observation, image_only=True)
            return embeddings

        return jax.eval_shape(get_embeddings, jax.random.key(0)).shape[1]

    class InferenceModel(nnx.Module):
        def __init__(self, vla_model, rngs, prefix_seq_len):
            self.vla = vla_model
            linen_rlt = RLTokenModel(config=rlt_config)
            self.rlt_module = nnx_bridge.ToNNX(linen_rlt)
            dummy_prefix = jnp.zeros((1, prefix_seq_len, rlt_config.input_dim))
            dummy_mask = jnp.ones((1, prefix_seq_len), dtype=jnp.bool_)
            self.rlt_module.lazy_init(dummy_prefix, dummy_mask, rngs=rngs)

        def infer_fixed(self, rng, observation):
            cache = self.vla.prepare_prefix_for_inference(observation)
            z = self.rlt_module(cache.image_prefix_out.astype(jnp.float32), None, method="encode", train=False)
            actions = self.vla.sample_actions_from_prefix_cache(rng, cache, num_steps=num_steps)
            return actions, z

    print("MODEL_LOAD structure", flush=True)
    vla = nnx.eval_shape(config.model.create, jax.random.key(0))
    model = InferenceModel(
        vla,
        rngs=nnx.Rngs(jax.random.key(1)),
        prefix_seq_len=infer_prefix_len(config.model),
    )
    graph, state = nnx.split(model)
    print("MODEL_LOAD restore", flush=True)
    loaded = model_api.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    expected = jax.tree_util.tree_structure(state.to_pure_dict())
    actual = jax.tree_util.tree_structure(loaded)
    if expected != actual:
        raise ValueError("checkpoint parameter tree does not match plug_v3 Stage-1")
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graph, state)
    print("MODEL_LOAD transforms", flush=True)

    input_transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(PROMPT),
            *data.data_transforms.inputs,
            transforms.Normalize(norm, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ]
    )
    infer = nnx_utils.module_jit(model.infer_fixed)
    print("MODEL_LOAD ready_for_compile", flush=True)

    class Policy:
        def __init__(self):
            self.metadata = {
                "cohort": "plug_v3_yyshadow",
                "stage": "stage1",
                "arm": "right",
                "has_rl_token": True,
                "z_dim": 2048,
                "proprio_dim": 7,
                "action_dim": 7,
                "model_horizon": 50,
                "actor_chunk_len": 10,
                "control_hz": 30,
                "reference_sampling": "fixed_seed_42",
                "denoising_steps": num_steps,
                "checkpoint": str(checkpoint),
                "norm_stats_sha256": hashlib.sha256(norm_file.read_bytes()).hexdigest(),
            }

        def infer(self, observation: dict) -> dict:
            state = np.asarray(observation.get("state"), dtype=np.float32)
            if state.shape != (7,) or not np.all(np.isfinite(state)):
                raise ValueError(f"plug_v3 requires finite right-arm 7D state, got {state.shape}")
            images = observation.get("images", {})
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
                image = np.asarray(images.get(key))
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"missing/invalid RGB camera {key}")
            transformed = input_transform(
                {"state": state.copy(), "images": images, "prompt": observation.get("prompt", PROMPT)}
            )
            batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
            model_observation = model_api.Observation.from_dict(batched)
            started = time.perf_counter()
            actions, token = infer(jax.random.key(42), model_observation)
            actions = np.asarray(actions[0])
            token = np.asarray(token[0], dtype=np.float32).reshape(-1)
            physical = output_transform({"state": np.asarray(transformed["state"]), "actions": actions})["actions"]
            physical = np.asarray(physical, dtype=np.float32)
            if physical.shape != (50, 7) or token.shape != (2048,):
                raise ValueError(f"invalid model output {physical.shape}/{token.shape}")
            if not np.all(np.isfinite(physical)) or not np.all(np.isfinite(token)):
                raise ValueError("model output contains non-finite values")
            return {
                "ref_chunk": physical[:10],
                "z_rl": token,
                "proprio": state.copy(),
                "policy_timing": {"infer_ms": 1000.0 * (time.perf_counter() - started)},
            }

    return Policy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_steps <= 20:
        raise ValueError("num-steps must be within 1..20")

    project = args.project_root.resolve()
    policy = load(project, args.checkpoint, num_steps=args.num_steps)
    dummy = {
        "images": {
            key: np.zeros((224, 224, 3), dtype=np.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "state": np.zeros(7, dtype=np.float32),
        "prompt": PROMPT,
    }
    timings = []
    for index in range(3):
        print(f"MODEL_VALIDATE inference_{index + 1}/3", flush=True)
        timings.append(policy.infer(dummy)["policy_timing"]["infer_ms"])
    receipt = {
        "status": "passed",
        "pid": os.getpid(),
        "metadata": policy.metadata,
        "inference_ms": timings,
        "robot_publishers": 0,
    }
    output = project / "outputs" / "rlt" / "plug_v3_yyshadow" / "model-server" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("MODEL_VALIDATED", json.dumps(receipt, sort_keys=True), flush=True)
    if args.validate_only:
        return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    print("MODEL_READY", flush=True)
    WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
