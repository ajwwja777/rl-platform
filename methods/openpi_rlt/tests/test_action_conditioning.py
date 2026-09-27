import numpy as np


def test_eval_noise_is_zero_and_explore_noise_is_chunk_correlated() -> None:
    from methods.openpi_rlt.cobot_adapter.action_conditioning import (
        chunk_exploration_noise,
    )

    eval_noise = chunk_exploration_noise(10, mode="eval", active_arm="left", std=0.002, seed=7)
    np.testing.assert_array_equal(eval_noise, np.zeros((10, 14), dtype=np.float32))

    noise = chunk_exploration_noise(10, mode="explore", active_arm="left", std=0.002, seed=7)
    assert noise.shape == (10, 14)
    assert np.max(np.abs(np.diff(noise[:, :6], axis=0))) < 0.006
    np.testing.assert_array_equal(noise[:, 6:], np.zeros((10, 8), dtype=np.float32))


def test_conditioner_holds_passive_arm_and_grippers_and_limits_acceleration() -> None:
    from methods.openpi_rlt.cobot_adapter.action_conditioning import ActionConditioner

    conditioner = ActionConditioner(
        active_arm="left",
        hold_grippers=True,
        joint_step_limit=0.03,
        joint_accel_limit=0.01,
        gripper_step_limit=0.002,
    )
    state = np.arange(14, dtype=np.float32) * 0.01
    requested = state.copy()
    requested[:6] += 0.5
    requested[6] += 0.02
    requested[7:13] -= 0.5
    requested[13] -= 0.02

    first, report1 = conditioner.condition(requested, state)
    second, report2 = conditioner.condition(requested, first)

    np.testing.assert_allclose(first[7:14], state[7:14])
    np.testing.assert_allclose(first[6], state[6])
    np.testing.assert_allclose(first[:6] - state[:6], 0.01, atol=1e-7)
    np.testing.assert_allclose(second[:6] - first[:6], 0.02, atol=1e-7)
    assert report1.acceleration_clipped == 6
    assert report2.acceleration_clipped == 6


def test_conditioner_reset_reanchors_to_latest_measured_state() -> None:
    from methods.openpi_rlt.cobot_adapter.action_conditioning import ActionConditioner

    conditioner = ActionConditioner(
        active_arm="both",
        hold_grippers=False,
        joint_step_limit=0.03,
        joint_accel_limit=0.01,
        gripper_step_limit=0.002,
    )
    state = np.zeros(14, dtype=np.float32)
    conditioner.condition(np.ones(14, dtype=np.float32), state)
    conditioner.reset()
    moved = np.full(14, 0.5, dtype=np.float32)
    output, _ = conditioner.condition(moved - 0.5, moved)
    np.testing.assert_allclose(output[[0, 7]], [0.49, 0.49])

