from types import SimpleNamespace
import numpy as np
import pytest
from methods.openpi_rlt.realtime_expo.left_arm_env import LeftArmRosTask2IO
from methods.openpi_rlt.tests.test_cobot_ros1_io import FakeRos,FakeJointState,_response

@pytest.mark.parametrize('method',['sample','sample_control','sample_publication_control'])
def test_only_wrist_subscriptions_left_feedback_and_left_publication(tmp_path,method):
    ros=FakeRos();ros.Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=lambda:100.01))
    io=LeftArmRosTask2IO(shadow_mode=False,trace_dir=tmp_path,ros_api=ros,bridge=SimpleNamespace(imgmsg_to_cv2=lambda m,e:m.pixels.copy()),image_type=object,joint_state_type=FakeJointState,bool_type=object,string_type=object,set_bool_type=object,trigger_type=object,response_factory=_response)
    assert '/camera_f/color/image_raw' not in ros.subscribers
    for key in ('cam_left_wrist','cam_right_wrist'):
        io._image_callback(key)(SimpleNamespace(header=SimpleNamespace(stamp=100.),pixels=np.zeros((4,4,3),np.uint8)))
    for side,v in [('left',.1),('right',.9)]:io._joint_callback(side)(SimpleNamespace(header=SimpleNamespace(stamp=100.),position=np.full(7,v)))
    sample=getattr(io,method)()
    np.testing.assert_allclose(sample.observation['state'],.1)
    if method!='sample_publication_control':assert set(sample.observation['images'])=={'left_wrist_0_rgb','right_wrist_0_rgb'}
    assert not io.publish_policy_action(np.zeros(7))
    io.set_policy_paused(False);assert io.publish_policy_action(np.arange(7)*.001)
    assert len(io._publishers['left'].messages)==1 and len(io._publishers['right'].messages)==0

def test_left_hil_uses_left_epoch_and_command():
    from methods.openpi_rlt.cobot_adapter.hil_targets import command_at_step_start
    from methods.openpi_rlt.tests.test_hil_targets import sample
    s=sample(mode='manual:left');e=s.io_evidence
    e['left_takeover_started_monotonic']=199.98
    e['coordinator_commands']['left']={**e['coordinator_commands']['right'],'topic':'/master/joint_left'}
    target,receipt=command_at_step_start(s,(True,False),side='left')
    assert receipt['topic']=='/master/joint_left'
    np.testing.assert_allclose(target,[.03]*6+[.004])
    with pytest.raises(ValueError):command_at_step_start(s,(False,True),side='left')

def test_explicit_logical10_options_preserve_default20():
    from methods.openpi_rlt.cobot_adapter.execution_profiles import selected_profile
    _,old=selected_profile(environ={'COBOT_RLT_EXECUTION_PROFILE':'async_rtc50'})
    _,new=selected_profile(environ={'COBOT_RLT_EXECUTION_PROFILE':'async_rtc50','COBOT_RLT_LOGICAL_HZ':'10'})
    assert old.logical_hz==20 and new.logical_hz==10 and new.publish_hz==50


def test_left10_logical_steps_publish50_and_record_final_command(monkeypatch):
    import importlib.util
    from pathlib import Path
    from methods.openpi_rlt.realtime_expo.left_arm_env import LeftArmCobotOnlineEnv
    from methods.openpi_rlt.cobot_adapter.async_execution import AsyncExecution
    from methods.openpi_rlt.cobot_adapter.execution_profiles import ExecutionProfile
    spec=importlib.util.spec_from_file_location('async_fixtures',Path(__file__).with_name('test_async_execution.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE','faithful20')
    clock=f.Clock();io=f.IO(clock)
    env=LeftArmCobotOnlineEnv(io,chunk_exec_horizon=10,control_frequency_hz=10,max_episode_steps=None,joint_step_limit=.03,gripper_step_limit=.004,sleep=clock.sleep)
    env._runtime.arm()
    engine=AsyncExecution(env,'left10-test',ExecutionProfile(logical_hz=10,publish_hz=50),clock);env._execution=engine;engine.backend=f.Backend()
    try:
        _,rewards,done,info=env.execute_chunk(io.sample().observation)
        assert len(rewards)==10 and len(io.published)==50 and not done
        assert clock()==pytest.approx(1.)
        np.testing.assert_allclose(np.diff([t for t,_ in io.published]),.02,atol=1e-9)
        for record in io.records:np.testing.assert_array_equal(record['action'],record['publications'][-1]['action'])
    finally:engine.close()


def test_left_factory_rejects_old_task_before_ros(monkeypatch):
    from methods.openpi_rlt.realtime_expo.left_arm_env import create_left_arm_online_env
    monkeypatch.delenv('COBOT_RLT_TASK_CONTRACT',raising=False)
    with pytest.raises(ValueError,match='TASK_CONTRACT'):
        create_left_arm_online_env()


def test_left_home_never_falls_back_to_old_task(monkeypatch):
    monkeypatch.delenv('COBOT_RLT_HOME_POSE',raising=False)
    io=object.__new__(LeftArmRosTask2IO)
    with pytest.raises(RuntimeError,match='not been registered'):io.request_home()


def test_left_runtime_keeps_variable_gripper():
    from methods.openpi_rlt.realtime_expo.left_arm_env import LeftArmPolicyRuntime
    runtime=LeftArmPolicyRuntime(joint_step_limit=.03,gripper_step_limit=.004,hold_gripper=False)
    runtime.arm()
    runtime.install_fresh_plan(runtime.snapshot().generation)
    current=np.zeros(7,np.float32);target=current.copy();target[6]=.003
    np.testing.assert_allclose(runtime.safe_policy_target(target,current),target)


def test_policy_input_rejects_middle_camera():
    from methods.openpi_rlt.realtime_expo.policy import prepare_observation
    obs=dict(state=np.zeros(7),images={k:np.zeros((480,640,3),np.uint8) for k in ('left_wrist_0_rgb','right_wrist_0_rgb')})
    result=prepare_observation(obs)
    assert set(result['images'])==set(obs['images'])
    assert all(v.shape==(224,224,3) for v in result['images'].values())
    obs['images']['base_0_rgb']=np.zeros((224,224,3),np.uint8)
    with pytest.raises(ValueError,match='forbidden'):prepare_observation(obs)
