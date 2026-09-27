"""Migration must not enable optional shaping or change current action clipping."""
import numpy as np
from methods.openpi_rlt.cobot_adapter.task2_runtime import Task2PolicyRuntime

def test_default_runtime_keeps_deployed_clipping():
    rng = np.random.default_rng(42)
    runtime = Task2PolicyRuntime(joint_step_limit=.03, gripper_step_limit=.004)
    state = runtime.arm()
    runtime.install_fresh_plan(state.generation)
    limits = np.array([.03]*6 + [.004] + [.03]*6 + [.004])
    for _ in range(100):
        measured = rng.uniform(-1, 1, 14).astype(np.float32)
        proposed = measured + rng.uniform(-.2, .2, 14).astype(np.float32)
        expected = measured + np.clip(proposed - measured, -limits, limits).astype(np.float32)
        np.testing.assert_array_equal(runtime.safe_policy_target(proposed, measured), expected)
