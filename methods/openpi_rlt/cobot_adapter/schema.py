"""Pure schema transforms for Cobot observations and actions."""

from collections.abc import Mapping

import numpy as np

_CAMERA_ROUTES = (
    ("cam_high", "base_0_rgb"),
    ("cam_left_wrist", "left_wrist_0_rgb"),
    ("cam_right_wrist", "right_wrist_0_rgb"),
)
PHYSICAL_ACTION_DIM = 14
MODEL_ACTION_DIM = 32
JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def _as_uint8_hwc(name: str, image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {array.dtype}")
    if array.ndim != 3:
        raise ValueError(f"{name} must have rank 3, got shape {array.shape}")
    if array.shape[-1] == 3:
        return np.ascontiguousarray(array)
    if array.shape[0] == 3:
        return np.ascontiguousarray(np.moveaxis(array, 0, -1))
    raise ValueError(f"{name} must be RGB HWC or CHW, got shape {array.shape}")


def map_cameras(images: Mapping[str, object]) -> dict[str, np.ndarray]:
    missing = [source for source, _ in _CAMERA_ROUTES if source not in images]
    if missing:
        raise ValueError(f"missing physical camera views: {', '.join(missing)}")

    source_images = [images[source] for source, _ in _CAMERA_ROUTES]
    if len({id(image) for image in source_images}) != len(source_images):
        raise ValueError("physical camera views must be distinct decoded frame objects")

    return {
        destination: _as_uint8_hwc(source, images[source])
        for source, destination in _CAMERA_ROUTES
    }


def _finite_float32(name: str, value: object, *, last_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != last_dim:
        raise ValueError(f"{name} must end in dimension {last_dim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _physical_state(state: object) -> np.ndarray:
    array = _finite_float32("state", state, last_dim=PHYSICAL_ACTION_DIM)
    if array.shape != (PHYSICAL_ACTION_DIM,):
        raise ValueError(f"state must have shape ({PHYSICAL_ACTION_DIM},), got {array.shape}")
    return array


def encode_state_14_to_32(state: object) -> np.ndarray:
    physical = _physical_state(state)
    encoded = np.zeros((MODEL_ACTION_DIM,), dtype=np.float32)
    encoded[:PHYSICAL_ACTION_DIM] = physical
    return encoded


def encode_action_chunk_14_to_32(actions: object, state: object) -> np.ndarray:
    physical_state = _physical_state(state)
    physical_actions = _finite_float32("actions", actions, last_dim=PHYSICAL_ACTION_DIM)
    if physical_actions.ndim < 2:
        raise ValueError(f"actions must contain a chunk axis, got shape {physical_actions.shape}")

    encoded = np.zeros((*physical_actions.shape[:-1], MODEL_ACTION_DIM), dtype=np.float32)
    encoded[..., :PHYSICAL_ACTION_DIM] = physical_actions
    encoded[..., JOINT_INDICES] -= physical_state[JOINT_INDICES]
    return encoded


def decode_action_chunk_32_to_14(actions: object, state: object) -> np.ndarray:
    physical_state = _physical_state(state)
    model_actions = _finite_float32("model actions", actions, last_dim=MODEL_ACTION_DIM)
    if model_actions.ndim < 2:
        raise ValueError(f"model actions must contain a chunk axis, got shape {model_actions.shape}")

    decoded = model_actions[..., :PHYSICAL_ACTION_DIM].copy()
    decoded[..., JOINT_INDICES] += physical_state[JOINT_INDICES]
    return decoded


def absolute_to_delta_chunk_14(actions: object, state: object) -> np.ndarray:
    """Convert absolute Cobot targets to joint-delta/absolute-gripper actions."""
    physical_state = _physical_state(state)
    absolute = _finite_float32("actions", actions, last_dim=PHYSICAL_ACTION_DIM)
    if absolute.ndim < 2:
        raise ValueError(f"actions must contain a chunk axis, got shape {absolute.shape}")

    delta = absolute.copy()
    zero_rows = np.all(np.isclose(absolute, 0.0), axis=-1, keepdims=True)
    delta[..., JOINT_INDICES] -= physical_state[JOINT_INDICES]
    return np.where(zero_rows, 0.0, delta).astype(np.float32, copy=False)


def delta_to_absolute_chunk_14(actions: object, state: object) -> np.ndarray:
    """Invert the Cobot joint-delta/absolute-gripper online representation."""
    physical_state = _physical_state(state)
    delta = _finite_float32("actions", actions, last_dim=PHYSICAL_ACTION_DIM)
    if delta.ndim < 2:
        raise ValueError(f"actions must contain a chunk axis, got shape {delta.shape}")

    absolute = delta.copy()
    absolute[..., JOINT_INDICES] += physical_state[JOINT_INDICES]
    return absolute


def validate_model_action_targets(actions: object) -> None:
    targets = _finite_float32("model action targets", actions, last_dim=MODEL_ACTION_DIM)
    if np.any(targets[..., PHYSICAL_ACTION_DIM:] != 0):
        raise ValueError("model action target padding dimensions must be exactly zero")
