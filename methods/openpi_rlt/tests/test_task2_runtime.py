from __future__ import annotations

import numpy as np
import pytest


def _runtime():
    from methods.openpi_rlt.cobot_adapter.task2_runtime import Task2PolicyRuntime

    return Task2PolicyRuntime(
        joint_step_limit=0.03,
        gripper_step_limit=0.004,
    )


def test_task2_mode_drives_per_arm_hil_without_task5_recording() -> None:
    from methods.openpi_rlt.cobot_adapter.trace import ControlSource

    runtime = _runtime()
    started = runtime.arm()
    runtime.install_fresh_plan(started.generation)

    left_hil = runtime.observe_mode("manual:left")
    assert left_hil.policy_paused is True
    assert left_hil.expert_mask == (True, False)
    assert runtime.control_source(ControlSource.RL) is ControlSource.MIXED

    both_hil = runtime.observe_mode("manual:left+right")
    assert both_hil.expert_mask == (True, True)
    assert runtime.control_source(ControlSource.RL) is ControlSource.HUMAN

    resumed = runtime.observe_mode("policy")
    assert resumed.policy_paused is False
    assert resumed.fresh_plan_required is True
    assert runtime.accepts_policy_result(started.generation) is False


def test_unknown_or_fault_task2_mode_fails_closed() -> None:
    runtime = _runtime()
    runtime.arm()

    with pytest.raises(RuntimeError, match="unknown Task2 mode"):
        runtime.observe_mode("manual:center")
    assert runtime.snapshot().phase.value == "fault"
    assert runtime.snapshot().policy_paused is True


def test_policy_target_is_rate_limited_and_reports_actual_executed_action() -> None:
    runtime = _runtime()
    started = runtime.arm()
    runtime.install_fresh_plan(started.generation)
    current = np.zeros(14, dtype=np.float32)
    requested = np.full(14, 1.0, dtype=np.float32)

    executed = runtime.safe_policy_target(requested, current)

    expected = np.array([0.03] * 6 + [0.004] + [0.03] * 6 + [0.004], dtype=np.float32)
    np.testing.assert_allclose(executed, expected)


def test_policy_target_is_never_emitted_during_hil_or_before_fresh_plan() -> None:
    runtime = _runtime()
    started = runtime.arm()
    current = np.zeros(14, dtype=np.float32)

    with pytest.raises(RuntimeError, match="fresh policy plan"):
        runtime.safe_policy_target(current, current)

    runtime.install_fresh_plan(started.generation)
    runtime.observe_mode("manual:right")
    with pytest.raises(RuntimeError, match="paused"):
        runtime.safe_policy_target(current, current)


def test_nonfinite_or_wrong_shape_action_fails_closed() -> None:
    runtime = _runtime()
    started = runtime.arm()
    runtime.install_fresh_plan(started.generation)

    with pytest.raises(ValueError, match="14"):
        runtime.safe_policy_target(np.zeros(7), np.zeros(14))
    assert runtime.snapshot().phase.value == "fault"

