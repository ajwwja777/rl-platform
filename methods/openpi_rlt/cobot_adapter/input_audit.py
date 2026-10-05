"""Optional Replay input receipts, separate from compact executed-step traces."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

RUN_ID = uuid.uuid4().hex
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def audit_mode() -> str:
    mode = os.environ.get("COBOT_RLT_INPUT_AUDIT", "off")
    if mode not in {"off", "record", "strict"}:
        raise ValueError("COBOT_RLT_INPUT_AUDIT must be off, record or strict")
    return mode


def array_receipt(value, dtype=None) -> dict:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot provide stable input fingerprints")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            "finite": bool(np.isfinite(array).all())}


def conditional_input_receipt(value):
    if isinstance(value, dict):
        return {str(key): conditional_input_receipt(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [conditional_input_receipt(item) for item in value]
    if isinstance(value, np.ndarray):
        return array_receipt(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported conditional input receipt type: {type(value).__name__}")


def observation_receipt(observation) -> dict:
    # Hash actual raw RGB inputs, not screenshots or an inferred nearby frame.
    result = {"state_fp32": array_receipt(observation["state"], np.float32),
              "prompt": str(observation.get("prompt", "")),
              "images": {key: array_receipt(value) for key, value in observation.get("images", {}).items()}}
    if "rtc" in observation:
        # RTC is conditional model input and must not disappear from identity.
        result["rtc"] = conditional_input_receipt(observation["rtc"])
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    return result


def window_receipt(raw_episode, raw_indices, transition) -> dict:
    first, last = raw_episode.steps[raw_indices[0]], raw_episode.steps[raw_indices[-1]]
    current = raw_episode.observations[first.observation_idx]
    following = raw_episode.observations[last.next_observation_idx]
    checks = {
        "current_state_equals_feature_proprio": np.array_equal(
            np.asarray(current["state"], np.float32), np.asarray(transition.proprio, np.float32)),
        "next_state_equals_feature_proprio": np.array_equal(
            np.asarray(following["state"], np.float32), np.asarray(transition.next_proprio, np.float32)),
    }
    arrays = {}
    for key in ("z_rl", "proprio", "ref_chunk", "action_chunk", "rewards", "next_z_rl", "next_proprio", "next_ref_chunk", "source_chunk"):
        value = getattr(transition, key)
        arrays[key] = {"native": array_receipt(value), "fp32": array_receipt(value, np.float32)}
        if key in {"z_rl", "ref_chunk", "action_chunk", "next_z_rl", "next_ref_chunk"}:
            arrays[key]["fp16"] = array_receipt(value, np.float16)
        if not arrays[key]["fp32"]["finite"]:
            checks["finite_"+key] = False
    # Use the actual installed serializer, including its opt-in precision
    # patch. Do not infer the transport dtype from configuration text.
    serialized = {key: array_receipt(value) for key, value in transition.to_numpy().items()}
    checks["finite_serialized_replay_arrays"] = all(value["finite"] for value in serialized.values())
    return {
        "schema": "rlt_replay_input_receipt_v1", "run_id": RUN_ID, "audit_source_sha256": SOURCE_SHA256,
        "episode_id": int(transition.episode_id), "step_id": int(transition.step_id),
        "raw_step_indices": list(map(int, raw_indices)),
        "current_observation_index": int(first.observation_idx),
        "next_observation_index": int(last.next_observation_idx),
        "current_input": observation_receipt(current), "next_input": observation_receipt(following),
        "training_arrays_before_storage": arrays, "checks": checks,
        "serialized_replay_arrays": serialized,
        "checks_passed": bool(all(checks.values())),
        "done": bool(transition.done), "terminal_success_flag": int(transition.success),
        "source": int(transition.source), "collection_phase": str(transition.collection_phase),
        "recorded_actor_versions": sorted({int(getattr(raw_episode.steps[i], "actor_param_version", -1)) for i in raw_indices}),
        "captured_unix": time.time(), "captured_monotonic": time.perf_counter(),
        "boundary": "Raw input/returned feature/training-array fingerprints before Replay submission; not checkpoint load, GPU preprocessing or hardware execution proof. Submission success requires the separate Replay acknowledgement.",
    }


def append_receipt(trace_root: Path, receipt: dict) -> Path:
    root = Path(trace_root) / "replay_inputs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"episode_{receipt['episode_id']}_run_{RUN_ID}.jsonl"
    line = (json.dumps(receipt, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)
    return path
