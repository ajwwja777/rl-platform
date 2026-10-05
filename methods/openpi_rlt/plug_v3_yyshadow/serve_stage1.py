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


from methods.openpi_rlt.plug_v3_yyshadow.stage1_loading import LoadTiming, restore_inference_params


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
    cache = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR",
                                str(project / "runtime/cache/jax/stage1")))
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    print("MODEL_LOAD compilation_cache=" + str(cache), flush=True)
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


def load(project: Path, checkpoint: Path, *, num_steps: int = 10, rtc_overlay: Path | None = None,
         reference_hz: int = 30, default_prompt: str = PROMPT):
    if not isinstance(default_prompt, str) or not default_prompt.strip():
        raise ValueError('default_prompt must be an explicit nonempty task instruction')
    if reference_hz not in (20, 30):
        raise ValueError('reference_hz must be the explicit 20 or 30 Hz time base')
    if reference_hz != 30 and rtc_overlay is not None:
        raise ValueError('retimed reference requires a separate RTC prefix time-base validation')
    checkpoint = checkpoint.resolve()
    asset_id = "plug_v3_yyshadow_demonstrations"
    norm_file = checkpoint / "assets" / asset_id / "norm_stats.json"
    if not (checkpoint / "params").is_dir() or not norm_file.is_file():
        raise ValueError(f"incomplete Stage-1 release: {checkpoint}")

    rtc_sampler = None
    timing = LoadTiming()
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
    timing.mark("imports")
    if rtc_overlay is not None:
        from methods.openpi_rlt.experiments.rtc import load_sampler
        rtc_sampler = load_sampler(rtc_overlay)
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

    timing.mark("config_and_transforms")
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
            self.rlt_module.lazy_init(
                dummy_prefix, dummy_mask, rngs=rngs, method="encode", train=False)

        def infer_fixed(self, rng, observation):
            cache = self.vla.prepare_prefix_for_inference(observation)
            z = self.rlt_module(cache.image_prefix_out.astype(jnp.float32), None, method="encode", train=False)
            actions = self.vla.sample_actions_from_prefix_cache(rng, cache, num_steps=num_steps)
            return actions, z

        def infer_rtc(self, rng, observation, previous, delay, execution):
            cache = self.vla.prepare_prefix_for_inference(observation)
            z = self.rlt_module(cache.image_prefix_out.astype(jnp.float32), None, method="encode", train=False)
            actions = rtc_sampler(self.vla, rng, observation, previous,
                inference_delay=delay, execution_horizon=execution,
                num_steps=num_steps, prefix_cache=cache)
            return actions, z

    print("MODEL_LOAD structure", flush=True)
    prefix_len = infer_prefix_len(config.model)

    def create_structure():
        # Evaluate the WHOLE structure abstractly. The previous loader created
        # real random encoder/decoder arrays before replacing them with weights.
        return InferenceModel(
            config.model.create(jax.random.key(0)),
            rngs=nnx.Rngs(jax.random.key(1)),
            prefix_seq_len=prefix_len,
        )

    model = nnx.eval_shape(create_structure)
    graph, state = nnx.split(model)
    timing.mark("abstract_structure")
    print("MODEL_LOAD restore", flush=True)
    loaded = restore_inference_params(checkpoint / "params")
    jax.block_until_ready(loaded)
    timing.mark("checkpoint_restore")
    expected = jax.tree_util.tree_structure(state.to_pure_dict())
    actual = jax.tree_util.tree_structure(loaded)
    if expected != actual:
        raise ValueError("checkpoint parameter tree does not match plug_v3 Stage-1")
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graph, state)
    print("MODEL_LOAD transforms", flush=True)

    input_transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(default_prompt),
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
    infer_rtc = nnx_utils.module_jit(model.infer_rtc) if rtc_sampler is not None else None
    timing.mark("bind_inference")
    print("MODEL_LOAD ready_for_compile", flush=True)

    class Policy:
        def __init__(self):
            self.load_seconds = dict(timing.seconds)
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
                "control_hz": reference_hz,
                "model_action_sampling_hz": 30,
                "reference_sampling_hz": reference_hz,
                "default_prompt": default_prompt,
                "reference_sampling": "fixed_seed_42",
                "denoising_steps": num_steps,
                "rtc_prefix_supported": infer_rtc is not None,
                "checkpoint": str(checkpoint),
                "norm_stats_sha256": hashlib.sha256(norm_file.read_bytes()).hexdigest(),
            }

        def sample_model_observation(self, observation):
            """Share the exact compiled fixed sampler with read-only audits."""
            return infer(jax.random.key(42), observation)

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
                {"state": state.copy(), "images": images, "prompt": observation.get("prompt", default_prompt)}
            )
            batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
            model_observation = model_api.Observation.from_dict(batched)
            started = time.perf_counter()
            rtc_request = observation.get("rtc")
            if rtc_request is not None:
                if infer_rtc is None:
                    raise ValueError("RTC is not enabled for this Stage1 server")
                from methods.openpi_rlt.experiments.rtc import encode_prefix
                previous, delay, execution = encode_prefix(
                    {"state": state, "images": images, "prompt": observation.get("prompt", default_prompt)},
                    rtc_request, input_transform, action_dim=config.model.action_dim)
                actions, token = infer_rtc(jax.random.key(42), model_observation,
                    jnp.asarray(previous)[None], jnp.asarray(delay), jnp.asarray(execution))
            else:
                actions, token = self.sample_model_observation(model_observation)
            actions = np.asarray(actions[0])
            token = np.asarray(token[0], dtype=np.float32).reshape(-1)
            physical = output_transform({"state": np.asarray(transformed["state"]), "actions": actions})["actions"]
            physical = np.asarray(physical, dtype=np.float32)
            if physical.shape != (50, 7) or token.shape != (2048,):
                raise ValueError(f"invalid model output {physical.shape}/{token.shape}")
            if not np.all(np.isfinite(physical)) or not np.all(np.isfinite(token)):
                raise ValueError("model output contains non-finite values")
            return {
                "ref_chunk": sample_reference(physical, reference_hz),
                "rtc_used": rtc_request is not None,
                "z_rl": token,
                "proprio": state.copy(),
                "policy_timing": {"infer_ms": 1000.0 * (time.perf_counter() - started)},
            }

    return Policy()


def sample_reference(physical: np.ndarray, reference_hz: int = 30) -> np.ndarray:
    """Preserve legacy output exactly or interpolate the 30 Hz model horizon.

    The optional 20 Hz output changes the frozen policy input contract. Existing
    Actors were trained with legacy references; it needs its own acceptance.
    """
    if reference_hz == 30:
        return physical[:10]
    if reference_hz != 20:
        raise ValueError('unsupported reference sampling rate')
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import interpolate_chunk
    return interpolate_chunk(physical, 30, 20, 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rtc-overlay", type=Path)
    parser.add_argument("--warmup-rtc", action="store_true")
    parser.add_argument("--reference-hz", type=int, choices=(20, 30), default=30,
                        help="Optional physical-time reference contract; default preserves the original 30 Hz samples.")
    parser.add_argument("--prompt", default=PROMPT, help="Explicit default instruction; legacy default preserved.")
    args = parser.parse_args()
    if not 1 <= args.num_steps <= 20:
        raise ValueError("num-steps must be within 1..20")

    project = args.project_root.resolve()
    policy = load(project, args.checkpoint, num_steps=args.num_steps, rtc_overlay=args.rtc_overlay,
                  reference_hz=args.reference_hz, default_prompt=args.prompt)
    dummy = {
        "images": {
            key: np.zeros((224, 224, 3), dtype=np.uint8)
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "state": np.zeros(7, dtype=np.float32),
        "prompt": args.prompt,
    }
    timings = []
    for index in range(3):
        print(f"MODEL_VALIDATE inference_{index + 1}/3", flush=True)
        timings.append(policy.infer(dummy)["policy_timing"]["infer_ms"])
    rtc_timings = []
    if args.warmup_rtc:
        if args.rtc_overlay is None:
            raise ValueError("--warmup-rtc needs --rtc-overlay")
        request = dict(dummy, rtc=dict(previous_actions=np.zeros((5,7),np.float32),
                                      delay_steps=4, execution_horizon=5))
        for index in range(3):
            result = policy.infer(request)
            if result.get("rtc_used") is not True:
                raise RuntimeError("RTC warmup was not accepted")
            rtc_timings.append(result["policy_timing"]["infer_ms"])
    receipt = {
        "status": "passed",
        "pid": os.getpid(),
        "metadata": policy.metadata,
        "inference_ms": timings,
        "rtc_inference_ms": rtc_timings,
        "load_seconds": policy.load_seconds,
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
