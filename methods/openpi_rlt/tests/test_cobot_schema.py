import importlib.util

import numpy as np
import pytest


def test_cobot_schema_module_is_importable() -> None:
    """Catches a missing project-owned Cobot schema boundary."""
    assert importlib.util.find_spec("methods.openpi_rlt.cobot_adapter.schema") is not None


def test_camera_mapping_preserves_physical_view_order() -> None:
    """Catches swapped head, left-wrist, and right-wrist camera routing."""
    from methods.openpi_rlt.cobot_adapter import schema

    map_cameras = getattr(schema, "map_cameras", None)
    assert callable(map_cameras)
    head = np.full((4, 5, 3), 10, dtype=np.uint8)
    left = np.full((4, 5, 3), 20, dtype=np.uint8)
    right = np.full((4, 5, 3), 30, dtype=np.uint8)

    mapped = map_cameras(
        {
            "cam_high": head,
            "cam_left_wrist": left,
            "cam_right_wrist": right,
        }
    )

    assert tuple(mapped) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    np.testing.assert_array_equal(mapped["base_0_rgb"], head)
    np.testing.assert_array_equal(mapped["left_wrist_0_rgb"], left)
    np.testing.assert_array_equal(mapped["right_wrist_0_rgb"], right)


def test_camera_mapping_converts_chw_to_hwc() -> None:
    """Catches passing deployment CHW tensors to an HWC model boundary unchanged."""
    from methods.openpi_rlt.cobot_adapter.schema import map_cameras

    chw = np.stack(
        [np.full((4, 5), 11, dtype=np.uint8), np.full((4, 5), 22, dtype=np.uint8), np.full((4, 5), 33, dtype=np.uint8)]
    )
    mapped = map_cameras(
        {
            "cam_high": chw,
            "cam_left_wrist": chw.copy(),
            "cam_right_wrist": chw.copy(),
        }
    )

    assert mapped["base_0_rgb"].shape == (4, 5, 3)
    np.testing.assert_array_equal(mapped["base_0_rgb"][0, 0], np.array([11, 22, 33], dtype=np.uint8))


def test_camera_mapping_rejects_missing_physical_view() -> None:
    """Catches silently padding a missing physical camera."""
    from methods.openpi_rlt.cobot_adapter.schema import map_cameras

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="cam_right_wrist"):
        map_cameras({"cam_high": image, "cam_left_wrist": image.copy()})


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((4, 5), dtype=np.uint8),
        np.zeros((4, 5, 3), dtype=np.float32),
        np.zeros((4, 5, 4), dtype=np.uint8),
    ],
)
def test_camera_mapping_rejects_invalid_image_schema(bad_image: np.ndarray) -> None:
    """Catches wrong rank, dtype, or channel count at the model boundary."""
    from methods.openpi_rlt.cobot_adapter.schema import map_cameras

    good_left = np.zeros((4, 5, 3), dtype=np.uint8)
    good_right = np.zeros((4, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        map_cameras(
            {
                "cam_high": bad_image,
                "cam_left_wrist": good_left,
                "cam_right_wrist": good_right,
            }
        )


def test_camera_mapping_rejects_aliasing_two_physical_views() -> None:
    """Catches wiring two physical camera keys to the same decoded frame object."""
    from methods.openpi_rlt.cobot_adapter.schema import map_cameras

    shared = np.zeros((4, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="distinct"):
        map_cameras(
            {
                "cam_high": shared,
                "cam_left_wrist": shared,
                "cam_right_wrist": shared.copy(),
            }
        )


def test_bimanual_action_encoding_uses_joint_delta_and_absolute_grippers() -> None:
    """Catches applying delta conversion to either gripper or only one arm."""
    from methods.openpi_rlt.cobot_adapter import schema

    encode = getattr(schema, "encode_action_chunk_14_to_32", None)
    assert callable(encode)
    state = np.array([1, 2, 3, 4, 5, 6, 0.01, 11, 12, 13, 14, 15, 16, 0.02], dtype=np.float32)
    action = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.04, 10.9, 11.8, 12.7, 13.6, 14.5, 15.4, 0.05], dtype=np.float32)

    encoded = encode(action[None, :], state)

    expected = np.zeros((1, 32), dtype=np.float32)
    expected[0, :6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    expected[0, 6] = 0.04
    expected[0, 7:13] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    expected[0, 13] = 0.05
    np.testing.assert_allclose(encoded, expected, atol=1e-6)


def test_bimanual_action_round_trip_preserves_physical_targets() -> None:
    """Catches a mismatched encode/decode convention at deployment."""
    from methods.openpi_rlt.cobot_adapter.schema import (
        decode_action_chunk_32_to_14,
        encode_action_chunk_14_to_32,
    )

    state = np.linspace(-0.6, 0.7, 14, dtype=np.float32)
    actions = np.stack([state, state + 0.01, state - 0.02]).astype(np.float32)
    actions[:, [6, 13]] = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]], dtype=np.float32)

    decoded = decode_action_chunk_32_to_14(encode_action_chunk_14_to_32(actions, state), state)

    np.testing.assert_allclose(decoded, actions, atol=1e-6)


def test_online_bimanual_delta_round_trip_uses_both_joint_groups() -> None:
    """Catches upstream's single-arm ``:6`` delta rule leaking into Cobot online RL."""
    from methods.openpi_rlt.cobot_adapter.schema import (
        absolute_to_delta_chunk_14,
        delta_to_absolute_chunk_14,
    )

    state = np.array([1, 2, 3, 4, 5, 6, 0.01, 11, 12, 13, 14, 15, 16, 0.02], dtype=np.float32)
    absolute = np.array(
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.04, 10.9, 11.8, 12.7, 13.6, 14.5, 15.4, 0.05]],
        dtype=np.float32,
    )

    delta = absolute_to_delta_chunk_14(absolute, state)

    np.testing.assert_allclose(delta[0, :6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(delta[0, 7:13], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], atol=1e-6)
    np.testing.assert_array_equal(delta[0, [6, 13]], absolute[0, [6, 13]])
    np.testing.assert_allclose(delta_to_absolute_chunk_14(delta, state), absolute, atol=1e-6)


def test_online_delta_preserves_zero_padding_rows() -> None:
    """Catches zero terminal padding becoming a nonzero negative current-state delta."""
    from methods.openpi_rlt.cobot_adapter.schema import absolute_to_delta_chunk_14

    state = np.linspace(-0.7, 0.6, 14, dtype=np.float32)
    chunk = np.stack([state, np.zeros(14, dtype=np.float32)])

    delta = absolute_to_delta_chunk_14(chunk, state)

    np.testing.assert_array_equal(delta[1], np.zeros(14, dtype=np.float32))


def test_state_encoding_places_14d_state_in_front_of_zero_padding() -> None:
    """Catches nonzero or shifted π0.5 state padding."""
    from methods.openpi_rlt.cobot_adapter.schema import encode_state_14_to_32

    state = np.arange(14, dtype=np.float32)
    encoded = encode_state_14_to_32(state)

    assert encoded.shape == (32,)
    np.testing.assert_array_equal(encoded[:14], state)
    np.testing.assert_array_equal(encoded[14:], np.zeros(18, dtype=np.float32))


@pytest.mark.parametrize(
    ("state", "actions"),
    [
        (np.zeros(13, dtype=np.float32), np.zeros((2, 14), dtype=np.float32)),
        (np.zeros(14, dtype=np.float32), np.zeros((2, 13), dtype=np.float32)),
        (np.full(14, np.nan, dtype=np.float32), np.zeros((2, 14), dtype=np.float32)),
        (np.zeros(14, dtype=np.float32), np.full((2, 14), np.inf, dtype=np.float32)),
    ],
)
def test_action_encoding_rejects_bad_shape_or_nonfinite_values(state: np.ndarray, actions: np.ndarray) -> None:
    """Catches malformed state/action labels entering normalization or loss."""
    from methods.openpi_rlt.cobot_adapter.schema import encode_action_chunk_14_to_32

    with pytest.raises(ValueError):
        encode_action_chunk_14_to_32(actions, state)


def test_model_target_validation_rejects_padding_pollution() -> None:
    """Catches accidental supervision on π0.5 padding dimensions."""
    from methods.openpi_rlt.cobot_adapter import schema

    validate = getattr(schema, "validate_model_action_targets", None)
    assert callable(validate)
    targets = np.zeros((2, 50, 32), dtype=np.float32)
    targets[1, 17, 20] = 1e-3

    with pytest.raises(ValueError, match="padding"):
        validate(targets)
