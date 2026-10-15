import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'methods/expo_ft/cobot_adapter'))
from gripper_feedback import FeedbackPlan
from core import limited_plan
from gripper_preview import PreviewRuntime

@pytest.mark.parametrize('smooth',[False,True])
def test_contact_opening_bypasses_closed_command_dead_travel(smooth):
    actions=np.zeros((8,7));actions[:,6]=.06
    p=FeedbackPlan(actions,np.zeros(7),smooth,True,5)
    result=p.prepare(0,np.zeros(7),.027)
    assert result[0,6]==pytest.approx(.031)
    value=p.interpolate(np.zeros(7),result[0],.2,0)
    assert value[6]==pytest.approx(.0278)
    assert result[0,6] <= .027+.00400001
    assert np.all(np.diff(result[:,6])<=.00400001)

@pytest.mark.parametrize('requested',[0.,.001,.02,.027])
@pytest.mark.parametrize('smooth',[False,True])
def test_contact_never_causes_unrequested_release(requested,smooth):
    actions=np.zeros((8,7));actions[:,6]=requested
    anchor=np.zeros(7)
    p=FeedbackPlan(actions,anchor,smooth,True,5)
    np.testing.assert_array_equal(p.prepare(0,anchor,.027),limited_plan(actions,anchor,smooth))

@pytest.mark.parametrize('rtc',[False,True])
@pytest.mark.parametrize('smooth',[False,True])
def test_all_arm_commands_unchanged_and_opening_does_not_overshoot(rtc,smooth):
    rng=np.random.RandomState(7);actions=rng.uniform(-2,2,(8,7));actions[:,6]=.03
    anchor=np.zeros(7);p=FeedbackPlan(actions,anchor,smooth,rtc,5)
    expected=limited_plan(actions,anchor,smooth)
    for i in range(8):
        last=anchor if i==0 else p.plan[i-1].copy()
        result=p.prepare(i,last,.027)
        np.testing.assert_array_equal(result[:,:6],expected[:,:6])
        assert np.all(result[:,6]<=.03000001)
        for f in [.2,.4,.6,.8,1.]:
            actual=p.interpolate(last,result[i],f,i)
            np.testing.assert_array_equal(actual[:6],(last+(result[i]-last)*f)[:6])

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.01,.09])
def test_invalid_feedback_rejected(bad):
    p=FeedbackPlan(np.zeros((8,7)),np.zeros(7),False,True,5)
    with pytest.raises(ValueError):p.prepare(0,np.zeros(7),bad)

def test_committed_rtc_prefix_is_immutable_despite_new_feedback():
    actions=np.zeros((8,7));actions[4:,6]=.06
    p=FeedbackPlan(actions,np.zeros(7),False,True,5)
    p.prepare(3,np.zeros(7),.027);prefix=p.plan[3:].copy()
    for i in range(4,8):p.prepare(i,p.plan[i-1],.04)
    np.testing.assert_array_equal(prefix,p.plan[3:])

def test_fresh_chunk_resets_commit_after_pause_or_replan():
    a=np.zeros((8,7));a[:,6]=.06
    old=FeedbackPlan(a,np.zeros(7),False,True,5);old.prepare(3,np.zeros(7),.027)
    new=FeedbackPlan(a,np.zeros(7),False,True,5)
    assert not new.committed
    assert new.prepare(0,np.zeros(7),.04)[0,6]==pytest.approx(.044)

def fixture(monkeypatch, takeover_after=None):
    spec=importlib.util.spec_from_file_location('loop_tests',ROOT/'tests/test_expo_left_runtime.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    v,calls,published=mod._loop_fixture(monkeypatch,delay=5,takeover_after=takeover_after)
    v.args=SimpleNamespace(model_id='expo-plug-actor11-gripper-feedback')
    for name in ('make_plan','prepare_plan','interpolate_command'):
        setattr(v,name,MethodType(getattr(PreviewRuntime,name),v))
    state=SimpleNamespace(position=[0]*6+[.027],header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:0.)))
    v.samples['state']=(state,100.)
    oldwait=v.stop.wait
    def wait(t):
        result=oldwait(t);v.samples['state']=(state,v.samples['state'][1]);return result
    v.stop.wait=wait
    return v,calls,published

def test_full_execution_loop_publishes_identical_committed_prefix(monkeypatch):
    v,calls,published=fixture(monkeypatch);v.run()
    assert len(published)==16
    np.testing.assert_allclose(calls[1],np.asarray(published)[[7,9,11,13,15]],atol=1e-7)
    assert published[0][6]==pytest.approx(.029)

def test_full_loop_hil_still_stops_mid_step(monkeypatch):
    v,calls,published=fixture(monkeypatch,takeover_after=1);v.run()
    assert len(published)==1 and v.gate.paused

def test_stale_feedback_cannot_create_a_reanchored_plan(monkeypatch):
    v,_,_=fixture(monkeypatch);v.samples['state']=(v.samples['state'][0],99.)
    with pytest.raises(ValueError,match='expired'):v.prepare_plan(np.zeros((8,7)),0,np.zeros(7))

def test_preview_rejects_wrong_version_or_training_identity():
    v=object.__new__(PreviewRuntime);v.args=SimpleNamespace(model_id='expo-plug-actor11-gripper-feedback')
    s=dict(deployment_id=v.args.model_id,actor_version=11,rtc_delay=5,edit_scale=.01,training_enabled=False,frozen_preview=True)
    v.validate_server_status(s)
    for key,bad in [('actor_version',0),('training_enabled',True),('edit_scale',.05)]:
        with pytest.raises(RuntimeError):v.validate_server_status({**s,key:bad})
