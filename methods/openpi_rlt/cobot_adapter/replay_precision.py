"""Opt-in FP32 action storage for all Replay serialization boundaries.

Install before creating Replay buffers, in each participating process. Existing
FP16 journals stay quantized when restored; casting cannot recover raw actions.
References/features retain the fixed upstream format for a single-factor change.
"""
from __future__ import annotations

import os
import numpy as np


def install_action_precision_patch(*, enabled: bool | None = None) -> bool:
    if enabled is None:
        mode = os.environ.get("COBOT_RLT_REPLAY_ACTION_PRECISION", "legacy")
        if mode not in {"legacy", "float32"}:
            raise ValueError("Replay action precision must be legacy or float32")
        enabled = mode == "float32"
    if not enabled:
        return False
    from rlt_online_rl.replay import RLTTransition

    if getattr(RLTTransition.to_numpy, "_cobot_action_float32", False):
        return True
    original = RLTTransition.to_numpy

    def to_numpy_float32(self):
        record = original(self)
        # Use the pre-serialization source, not the already quantized record.
        record["action_chunk"] = np.asarray(self.action_chunk, dtype=np.float32)
        return record

    to_numpy_float32._cobot_action_float32 = True
    RLTTransition.to_numpy = to_numpy_float32
    return True
