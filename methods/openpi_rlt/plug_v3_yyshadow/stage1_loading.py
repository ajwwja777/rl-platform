"""Inference-only checkpoint restore, preserving upstream parameter values."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


class LoadTiming:
    """Report completed phases, not estimates or fake progress."""

    def __init__(self):
        self.started = self.previous = time.perf_counter()
        self.seconds = {}

    def mark(self, phase):
        now = time.perf_counter()
        self.seconds[phase] = now - self.previous
        self.previous = now
        print("MODEL_LOAD_TIMING " + json.dumps({
            "phase": phase, "seconds": round(self.seconds[phase], 3),
            "elapsed_seconds": round(now - self.started, 3),
        }), flush=True)


def inference_tree(params):
    """Select exactly the branches executed by Stage-1 inference.

    The RL-token decoder reconstructs embeddings during Stage-1 training.
    Serving calls encode only. Reject unknown roots instead of silently
    accepting a different model/checkpoint contract.
    """
    if set(params) != {"vla", "rlt_module"}:
        raise ValueError("Expected Stage-1 vla and rlt_module parameter roots")
    token = params["rlt_module"]
    if "encoder" not in token or set(token) - {"encoder", "decoder"}:
        raise ValueError("Expected RL-token encoder with optional training decoder")
    return {"vla": params["vla"], "rlt_module": {"encoder": token["encoder"]}}


def restore_inference_params(params_path: Path, *, restore_type=None, sharding=None):
    # initialize() imports PyArrow/Torch before these libraries in the frozen
    # environment; keep this helper free of heavy module-level imports.
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from flax import traverse_util

    if restore_type is None:
        restore_type = jax.Array
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(params_path)
        selected = inference_tree(metadata["params"])
        leaves = jax.tree.leaves(selected)
        skipped = jax.tree.leaves(metadata["params"]["rlt_module"].get("decoder", {}))
        summary = {
            "restored_parameters": sum(math.prod(x.shape) for x in leaves),
            "skipped_decoder_parameters": sum(math.prod(x.shape) for x in skipped),
            "source_selected_bytes": sum(math.prod(x.shape) * x.dtype.itemsize for x in leaves),
            "restore_dtype": "bfloat16",
        }
        print("MODEL_LOAD_PARAMETERS " + json.dumps(summary), flush=True)
        item = {"params": selected}
        # transforms={} is required by frozen Orbax 0.11.13 to deserialize
        # only this subtree. Dropping decoder AFTER restore would not save I/O.
        loaded = checkpointer.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item, transforms={},
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=restore_type, dtype=jnp.bfloat16),
                    item,
                ),
            ),
        )["params"]
    flat = traverse_util.flatten_dict(loaded)
    if all(path[-1] == "value" for path in flat):
        flat = {path[:-1]: value for path, value in flat.items()}
    return traverse_util.unflatten_dict(flat)
