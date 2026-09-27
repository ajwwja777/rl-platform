import numpy as np


def _apply(transforms, value):
    for transform in transforms:
        value = transform(value)
    return value


def test_cobot_data_transforms_route_three_physical_cameras_and_both_arms() -> None:
    """Catches using the upstream single-arm camera/action defaults for Cobot."""
    from openpi.models.pi0_config import Pi0Config

    from methods.openpi_rlt.cobot_adapter.upstream_data import build_cobot_transform_groups

    model = Pi0Config(pi05=True, action_dim=32, action_horizon=50)
    repack, data = build_cobot_transform_groups(model)
    state = np.array([1, 2, 3, 4, 5, 6, 0.01, 11, 12, 13, 14, 15, 16, 0.02], dtype=np.float32)
    actions = np.array(
        [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.04, 10.9, 11.8, 12.7, 13.6, 14.5, 15.4, 0.05]],
        dtype=np.float32,
    )
    sample = {
        "observation.images.cam_high": np.full((3, 4, 5), 0.1, dtype=np.float32),
        "observation.images.cam_left_wrist": np.full((3, 4, 5), 0.2, dtype=np.float32),
        "observation.images.cam_right_wrist": np.full((3, 4, 5), 0.3, dtype=np.float32),
        "observation.state": state,
        "action": actions,
        "prompt": "Open the pot lid, put the object into the pot, then close the lid.",
    }

    transformed = _apply((*repack.inputs, *data.inputs), sample)

    assert tuple(transformed["image"]) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert transformed["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert int(transformed["image"]["base_0_rgb"][0, 0, 0]) == 25
    assert int(transformed["image"]["left_wrist_0_rgb"][0, 0, 0]) == 51
    assert int(transformed["image"]["right_wrist_0_rgb"][0, 0, 0]) == 76
    expected = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.05]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(transformed["actions"], expected, atol=1e-6)


def test_cobot_output_transforms_restore_absolute_14d_targets() -> None:
    """Catches returning 32D deltas or applying delta conversion to grippers."""
    from openpi.models.pi0_config import Pi0Config

    from methods.openpi_rlt.cobot_adapter.upstream_data import build_cobot_transform_groups

    model = Pi0Config(pi05=True, action_dim=32, action_horizon=50)
    _, data = build_cobot_transform_groups(model)
    state = np.array([1, 2, 3, 4, 5, 6, 0.01, 11, 12, 13, 14, 15, 16, 0.02] + [0] * 18, dtype=np.float32)
    model_actions = np.zeros((1, 32), dtype=np.float32)
    model_actions[0, :14] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.04, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.05]

    transformed = _apply(data.outputs, {"state": state, "actions": model_actions})

    assert transformed["actions"].shape == (1, 14)
    np.testing.assert_allclose(
        transformed["actions"][0],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 0.04, 10.9, 11.8, 12.7, 13.6, 14.5, 15.4, 0.05],
        atol=1e-6,
    )


def test_cobot_data_factory_uses_lerobot_action_sequence_and_task_prompt(tmp_path) -> None:
    """Catches a loader that samples the wrong field or drops the dataset task."""
    from openpi.models.pi0_config import Pi0Config

    from methods.openpi_rlt.cobot_adapter.upstream_data import CobotLeRobotDataConfig

    factory = CobotLeRobotDataConfig(repo_id="legacy40-v2.1")
    config = factory.create(tmp_path, Pi0Config(pi05=True, action_dim=32, action_horizon=50))

    assert config.repo_id == "legacy40-v2.1"
    assert tuple(config.action_sequence_keys) == ("action",)
    assert config.prompt_from_task is True
