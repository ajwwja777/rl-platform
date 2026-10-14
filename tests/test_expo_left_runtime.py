import importlib.util
from pathlib import Path
import numpy as np
import pytest
from methods.expo_ft.cobot_adapter.core import Gate, limited_plan, execution_options, validate_observation


def test_load_is_paused_and_handback_cannot_release_operator_pause():
    gate = Gate()
    gate.update_mode('policy')
    assert gate.paused
    assert gate.request(False, '/cobot_deployment_command_123')[0]
    assert not gate.paused
    before = gate.generation
    gate.request(True, '/task2_teach_handover')
    gate.update_mode('manual:left')
    assert gate.paused and gate.generation > before and gate.intervention_count == 1
    gate.request(True, '/cobot_deployment_command')
    gate.request(False, '/task2_teach_handover')
    gate.update_mode('policy')
    assert gate.paused and gate.manual_pause
    gate.request(False, '/cobot_deployment_command')
    assert not gate.paused


def test_plain_hil_handback_resumes_with_a_different_generation():
    gate = Gate(); gate.update_mode('policy');gate.request(False, '/cobot_deployment_command')
    old = gate.generation
    gate.request(True, '/task2_teach_handover');gate.update_mode('manual:right')
    assert gate.paused
    gate.request(False, '/task2_teach_handover');gate.update_mode('policy')
    assert not gate.paused and gate.generation != old


def test_protective_resume_and_hil_operator_resume_are_rejected():
    gate = Gate();gate.update_mode('policy')
    assert not gate.request(False, '/rosservice')[0]
    gate.request(False, '/cobot_deployment_command')
    gate.update_mode('manual:left')
    assert not gate.request(False, '/cobot_deployment_command')[0]
    gate.fail('deadline')
    gate.request(False, '/task2_teach_handover');gate.update_mode('policy')
    assert gate.paused and gate.fault == 'deadline'


@pytest.mark.parametrize('smoothing',[False,True])
def test_physical_prefix_equals_the_future_guarded_endpoints(smoothing):
    anchor=np.array([0,1,-1,0,1,0,.02])
    actions=np.tile(anchor+np.array([.5]*6+[.03]),(8,1))
    guarded=limited_plan(actions,anchor,smoothing)
    steps=np.diff(np.vstack([anchor,guarded]),axis=0)
    assert np.all(np.abs(steps[:,:6])<=.0300001)
    assert np.all(np.abs(steps[:,6])<=.0040001)
    np.testing.assert_allclose(guarded[6:8],limited_plan(actions[6:8],guarded[5],smoothing),atol=1e-7)
    assert np.all((guarded[:,6]>=0)&(guarded[:,6]<=.08))


@pytest.mark.parametrize('shape',[(8,14),(8,6),(7,)])
def test_invalid_actions_do_not_reach_a_publisher(shape):
    with pytest.raises(ValueError):limited_plan(np.zeros(shape),np.zeros(7))


def test_nonfinite_targets_rejected():
    x=np.zeros((8,7));x[3,0]=np.nan
    with pytest.raises(ValueError):limited_plan(x,np.zeros(7))


def test_required_two_views_no_top_camera_and_fresh_format():
    obs=dict(state=np.zeros(7),left_wrist_image=np.full((180,320,3),10,np.uint8),right_wrist_image=np.full((180,320,3),20,np.uint8))
    assert validate_observation(obs) is obs
    obs['state']=np.zeros(14)
    with pytest.raises(ValueError):validate_observation(obs)
    obs['state']=np.zeros(7);obs['right_wrist_image']=np.zeros((180,320,3),np.uint8)
    with pytest.raises(ValueError):validate_observation(obs)


def test_publish_overrides_never_change_training_time():
    assert execution_options({'enabled':False})==dict(publish_hz=10,rtc=True,smoothing=False)
    for hz in (20,30,40,50):
        opts=execution_options(dict(enabled=True,publish_hz=hz,rtc=True,smoothing=False))
        assert opts['publish_hz']//10 in (2,3,4,5)
    with pytest.raises(ValueError):execution_options(dict(enabled=True,publish_hz=30,rtc='true',smoothing=False))


def _loop_fixture(monkeypatch, rtc=True, takeover_after=None, delay=2):
    import sys
    import threading
    from concurrent.futures import Future
    from types import SimpleNamespace
    from methods.expo_ft.cobot_adapter import core
    monkeypatch.setitem(sys.modules,'core',core)
    path=Path(core.__file__).with_name('ros_runtime.py')
    spec=importlib.util.spec_from_file_location('expo_runtime_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    clock=[100.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    value=object.__new__(module.Runtime)
    value.lock=threading.RLock();value.gate=Gate();value.gate.update_mode('policy');value.gate.request(False,'/cobot_deployment_command')
    value.options=dict(publish_hz=20,rtc=rtc,smoothing=False)
    value.rtc_delay=delay
    value.server=SimpleNamespace(poll=lambda:None)
    value.ready=True;value.chunk_ready=False;value.last_gate_save=100.;value.last_resume=99.
    value.future=None;value.published=0
    value.samples={'state':(None,100.)}
    calls=[];published=[]
    class Event:
        stopped=False
        def is_set(self):return self.stopped
        def set(self):self.stopped=True
        def wait(self,t):
            clock[0]+=max(t,.0001)
            value.samples['state']=(None,clock[0])
            return self.stopped
    value.stop=Event()
    value.ros=SimpleNamespace(is_shutdown=lambda:False,Time=SimpleNamespace(now=lambda:clock[0]))
    value.joint_type=lambda:SimpleNamespace(header=SimpleNamespace(stamp=None))
    def publish(message):
        assert not value.gate.paused
        published.append(np.asarray(message.position))
        if takeover_after and len(published)==takeover_after:
            value.gate.request(True,'/task2_teach_handover')
            value.gate.update_mode('manual:left')
            value.stop.set()
        elif len(published)==16:value.stop.set()
    value.publisher=SimpleNamespace(publish=publish)
    value.start_server=lambda:None
    value.save=lambda:None
    obs=dict(state=np.zeros(7),left_wrist_image=np.full((180,320,3),10,np.uint8),right_wrist_image=np.full((180,320,3),10,np.uint8))
    value.observation=lambda **_:obs.copy()
    def infer(observation,prefix):
        calls.append(np.asarray(prefix).copy())
        return np.ones((8,7),np.float32)
    class Pool:
        def submit(self,fn,*args):
            future=Future();future.set_result(infer(*args));return future
    value.pool=Pool();value.infer=infer
    return value,calls,published


def test_full_publish_loop_prefetches_actual_guarded_prefix(monkeypatch):
    value,calls,published=_loop_fixture(monkeypatch)
    value.run()
    assert len(published)==16 and len(calls)==2
    assert calls[0].shape==(0,7) and calls[1].shape==(2,7)
    np.testing.assert_allclose(calls[1],np.asarray(published)[[13,15]],atol=1e-7)


def test_hil_invalidates_in_the_middle_of_physical_substeps(monkeypatch):
    value,calls,published=_loop_fixture(monkeypatch,takeover_after=1)
    before=value.gate.generation
    value.run()
    assert len(published)==1 and value.gate.paused and value.gate.generation>before
    assert len(calls)==1


def test_fast_hil_handback_discards_pending_rtc_without_old_bootstrap_clock(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    value,calls,published=_loop_fixture(monkeypatch)
    pending=Future()
    class Pool:
        def submit(self,fn,observation,prefix):
            actions=fn(observation,prefix)
            if len(prefix):return pending
            f=Future();f.set_result(actions);return f
    value.pool=Pool()
    original_wait=value.stop.wait
    def wait(t):
        result=original_wait(t)
        if value.gate.intervention_count and not pending.done():
            pending.set_result(np.full((8,7),9.,np.float32))
        return result
    value.stop.wait=wait
    original_publish=value.publisher.publish
    def publish(message):
        original_publish(message)
        if len(published)==13:
            value.mode_callback(SimpleNamespace(data='manual:left'))
            value.mode_callback(SimpleNamespace(data='policy'))
    value.publisher.publish=publish
    reasons=[]
    original_fail=value.fail
    def fail(reason):
        reasons.append(str(reason));original_fail(reason);value.stop.set()
    value.fail=fail
    value.run()
    assert not reasons
    assert len(calls)==3 and calls[-1].shape==(0,7)
    assert len(published)==16 and value.gate.intervention_count==1


def test_native10_explicit_rtc_filter_overrides():
    for rtc in (False,True):
        for smoothing in (False,True):
            assert execution_options(dict(enabled=True,publish_hz=10,rtc=rtc,smoothing=smoothing))==dict(publish_hz=10,rtc=rtc,smoothing=smoothing)


def test_remote_five_step_prefix_uses_actual_future_commands(monkeypatch):
    value,calls,published=_loop_fixture(monkeypatch,delay=5)
    value.run()
    assert len(published)==16 and len(calls)==2
    assert calls[1].shape==(5,7)
    np.testing.assert_allclose(calls[1],np.asarray(published)[[7,9,11,13,15]],atol=1e-7)
