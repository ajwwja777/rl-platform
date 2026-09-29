#!/usr/bin/env python3
"""CPU-only comparison of original and inference-only Stage-1 checkpoint reads.

No ROS, model server, robot publishers, Replay or checkpoint writes.
Run with the frozen Stage-1 environment. Timings are host/cache dependent;
they are not a Cobot cold-start benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project), str(project / "third_party/openpi-rlt/src")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import pyarrow  # noqa: F401 -- frozen environment requires this order
    import torch  # noqa: F401
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.model import restore_params
    from openpi.models.rl_token import RLTokenConfig, RLTokenModel
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_loading import inference_tree, restore_inference_params

    checkpoint = args.checkpoint.resolve()
    metadata = checkpoint / "params/_METADATA"
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    started = time.perf_counter()
    print("VALIDATE original full restore", flush=True)
    original = restore_params(checkpoint / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
    original_seconds = time.perf_counter() - started
    print("VALIDATE inference-only restore", flush=True)
    started = time.perf_counter()
    selected = restore_inference_params(checkpoint / "params", restore_type=np.ndarray)
    selected_seconds = time.perf_counter() - started
    old_leaves, old_tree = jax.tree.flatten(inference_tree(original))
    new_leaves, new_tree = jax.tree.flatten(selected)
    assert old_tree == new_tree, "Inference parameter tree changed"
    for before, after in zip(old_leaves, new_leaves):
        assert before.dtype == after.dtype
        np.testing.assert_array_equal(before, after)
    print("VALIDATE every retained parameter matches bit-for-bit", flush=True)

    encoder = selected["rlt_module"]["encoder"]
    prefix_len, width = encoder["y_pos_enc"].shape
    count = prefix_len * width
    observation = jnp.asarray(np.linspace(-0.25, 0.25, count, dtype=np.float32).reshape(1, prefix_len, width))
    model = RLTokenModel(config=RLTokenConfig(
        num_rl_tokens=1, num_layers=2, embed_dim=2048, input_dim=2048))
    infer = jax.jit(lambda parameters, x: model.apply(
        {"params": parameters}, x, None, method="encode", train=False))
    print("VALIDATE real encoder output, CPU only", flush=True)
    before = np.asarray(infer(original["rlt_module"], observation))
    after = np.asarray(infer(selected["rlt_module"], observation))
    np.testing.assert_array_equal(before, after)
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == metadata_sha
    receipt = {
        "status": "passed", "checkpoint": str(checkpoint), "backend": jax.default_backend(),
        "checkpoint_metadata_sha256": metadata_sha,
        "retained_parameters": sum(x.size for x in new_leaves),
        "skipped_decoder_parameters": sum(x.size for x in jax.tree.leaves(original["rlt_module"]["decoder"])),
        "parameter_values": "bitwise_equal", "encoder_output": "bitwise_equal",
        "encoder_output_shape": list(after.shape),
        "original_restore_seconds": original_seconds,
        "inference_restore_seconds": selected_seconds,
        "timing_limit": "CPU host, full restore first; OS cache not controlled. Not a cold-start speedup claim.",
        "robot_publishers": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
