"""Cobot contract patch for the fixed upstream Machine A server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PROPRIO_DIM = 14
ACTION_DIM = 14
CHUNK_LEN = 50
Z_DIM = 2048


def configure_serve_module(module: Any, config: Any) -> None:
    """Register the project config and replace upstream's 7D demo constants."""
    module.PROPRIO_DIM = PROPRIO_DIM
    module.ACTION_DIM = ACTION_DIM
    module.CHUNK_LEN = CHUNK_LEN
    # The live Cobot client always submits one observation.  Compiling larger
    # batches only consumes startup time and scarce RTX 4090 memory.
    module.RLTPolicy.COMPILED_BATCH_SIZES = [1]
    module._config._CONFIGS_DICT[config.name] = config


def validate_machine_a_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "has_rl_token": True,
        "z_dim": Z_DIM,
        "proprio_dim": PROPRIO_DIM,
        "action_dim": ACTION_DIM,
        "chunk_len": CHUNK_LEN,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Machine A {name} expected {value!r}, got {metadata.get(name)!r}")
    return metadata
