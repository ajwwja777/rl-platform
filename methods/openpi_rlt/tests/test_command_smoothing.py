import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.action_conditioning import ActionConditioner

def make():
    return ActionConditioner(active_arm="right", hold_grippers=True,
        joint_step_limit=.02, joint_accel_limit=.005, gripper_step_limit=.004,
        command_smoothing_alpha=.35, command_step_limit=.01)

def test_smoothing_published_step_and_passive_hold():
    c=make(); state=np.zeros(14, dtype=np.float32); previous=state.copy()
    for k in range(100):
        target=np.full(14, .3 if k%7 else -.3, dtype=np.float32)
        out,report=c.condition(target,state)
        assert np.max(np.abs(out[7:13]-previous[7:13])) <= .010001
        assert np.max(np.abs(out-state)) <= .020001
        np.testing.assert_array_equal(out[:7],state[:7])
        assert out[13]==state[13]
        previous=out; state=out.copy()

def test_reset_reanchors_at_new_measured_pose():
    c=make(); c.condition(np.ones(14),np.zeros(14)); c.reset()
    out,_=c.condition(np.ones(14),np.full(14,.5))
    assert .5 < out[7] <= .51

def test_discontinuous_measurement_fails_closed():
    c=make(); c.condition(np.ones(14),np.zeros(14))
    with pytest.raises(ValueError,match="tracking"):
        c.condition(np.ones(14),np.full(14,.5))

def test_smoothing_uses_previous_command():
    c=make()
    a,_=c.condition(np.ones(14),np.zeros(14))
    b,_=c.condition(np.ones(14),np.zeros(14))
    assert b[7] > a[7]
    assert b[7] < .01

@pytest.mark.parametrize("alpha", [0, -1, 1.1, float("nan")])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        ActionConditioner(active_arm="right",hold_grippers=True,joint_step_limit=.02,
            joint_accel_limit=.005,gripper_step_limit=.004,
            command_smoothing_alpha=alpha,command_step_limit=.01)

def test_runtime_hil_clears_history_and_plan_install_preserves_it():
    from methods.openpi_rlt.cobot_adapter.task2_runtime import Task2PolicyRuntime
    r=Task2PolicyRuntime(joint_step_limit=.02,gripper_step_limit=.004,
        joint_accel_limit=.005,active_arm="right",hold_grippers=True,
        command_smoothing_alpha=.35,command_step_limit=.01)
    snap=r.arm();r.install_fresh_plan(snap.generation)
    first=r.safe_policy_target(np.ones(14),np.zeros(14))
    r.install_fresh_plan(r.snapshot().generation)
    second=r.safe_policy_target(np.ones(14),np.zeros(14))
    assert second[7]>first[7]
    r.observe_mode("manual:right")
    snap=r.observe_mode("policy");r.install_fresh_plan(snap.generation)
    out=r.safe_policy_target(np.ones(14),np.full(14,.5))
    assert .5<out[7]<.51
    r.reset_action_conditioning()
    out=r.safe_policy_target(np.ones(14),np.full(14,.8))
    assert .8<out[7]<.81
