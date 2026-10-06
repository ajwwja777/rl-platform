import numpy as np
import pytest
import jax
import jax.numpy as jnp
from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic

class LinearCritic:
    def q_values(self,params,z,p,a):
        q=(a*jnp.arange(1.,8.)).sum((-1,-2))
        return q,2*q

@pytest.mark.parametrize('batch_shape',[(2,10,7),(10,7)])
def test_projection_units_and_invariance(batch_shape):
    wrapper=HeldGripperCritic(LinearCritic(),.03,.04)
    a=jnp.ones(batch_shape);p=jnp.zeros(batch_shape[:-2]+(7,)).at[...,6].set(.032)
    projected=wrapper.project(a,p)
    np.testing.assert_array_equal(projected[...,:6],a[...,:6])
    np.testing.assert_allclose(projected[...,6],(.032-.03)/(.01+1e-6)*2-1,atol=1e-6)
    q=wrapper.q_values(None,None,p,a)
    changed=wrapper.q_values(None,None,p,a.at[...,6].set(9999.))
    for x,y in zip(q,changed):np.testing.assert_array_equal(x,y)
    grad=jax.grad(lambda x:wrapper.q_values(None,None,p,x)[0].sum())(a)
    np.testing.assert_array_equal(grad[...,6],0.)
    assert np.all(np.asarray(grad[...,:6])>0)

def test_next_state_uses_next_held_grip_and_rejects_wrong_contract():
    wrapper=HeldGripperCritic(LinearCritic(),.03,.04)
    a=jnp.zeros((2,10,7));p=jnp.zeros((2,7)).at[...,6].set(.032)
    assert not np.array_equal(wrapper.project(a,p),wrapper.project(a,p.at[...,6].set(.036)))
    with pytest.raises(ValueError):wrapper.project(jnp.zeros((2,10,14)),jnp.zeros((2,14)))
    with pytest.raises(ValueError):HeldGripperCritic(LinearCritic(),.03,.03)


def test_explicit_control_hold_matches_task_without_changing_default():
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmPolicyRuntime
    measured=np.zeros(7,np.float32);measured[6]=.033
    requested=measured.copy();requested[:6]=.002;requested[6]=.1
    for hold in [False,True]:
        runtime=RightArmPolicyRuntime(joint_step_limit=.03,gripper_step_limit=.004,hold_gripper=hold)
        snap=runtime.arm();runtime.install_fresh_plan(snap.generation)
        out=runtime.safe_policy_target(requested,measured)
        np.testing.assert_array_equal(out[:6],requested[:6])
        assert out[6]==pytest.approx(measured[6]if hold else measured[6]+.004)
