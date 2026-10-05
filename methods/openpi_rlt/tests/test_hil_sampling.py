import numpy as np
import pytest
from methods.openpi_rlt.tests.test_cobot_online_env import FakeTask2IO, _sample, _plan
from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


def env(io, sleeps, **kw):
    return CobotOnlineEnv(io,chunk_exec_horizon=kw.pop('horizon',1),
        control_frequency_hz=kw.pop('hz',20.),max_episode_steps=None,
        joint_step_limit=.03,gripper_step_limit=.004,sleep=sleeps.append,**kw)


def test_optin_hil_samples_feedback_after_logical_period_and_never_publishes(monkeypatch):
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING','logical20')
    io=FakeTask2IO([_sample(0),_sample(.01,'manual:right'),_sample(.06,'manual:right')])
    sleeps=[];e=env(io,sleeps);ob=e.reset()
    _,_,done,info=e.execute_chunk(ob,lambda *_:_plan(.9))
    assert sleeps==[.05] and io.published==[] and not done
    r=info['step_trace'][0]
    np.testing.assert_allclose(r['action'],.06)
    assert r['human_controlled'] and r['hil_sampling_mode']=='logical20'
    assert r['sample_received_monotonic']>=r['interval_start_sample_received_monotonic']


def test_release_at_end_of_hil_interval_requires_fresh_plan(monkeypatch):
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING','logical20')
    io=FakeTask2IO([_sample(0),_sample(.01,'manual:right'),_sample(.06,'policy'),
        _sample(.07,'policy'),_sample(.12,'policy',EpisodeOutcome.SUCCESS)])
    sleeps=[];e=env(io,sleeps,horizon=2);ob=e.reset();anchors=[]
    def planner(ob,*_):anchors.append(float(ob['state'][0]));return _plan(.9)
    _,rewards,done,info=e.execute_chunk(ob,planner)
    assert done and rewards==[0.,1.] and len(io.published)==1
    assert info['step_trace'][0]['human_controlled']
    assert not info['step_trace'][1]['human_controlled']
    np.testing.assert_allclose(anchors,[0.,.07])
    assert sleeps==[.05,.05]


def test_legacy_sampling_default_preserves_existing_feedback_stream(monkeypatch):
    monkeypatch.delenv('COBOT_RLT_HIL_SAMPLING',raising=False)
    io=FakeTask2IO([_sample(0),_sample(.01,'manual:right')]);sleeps=[]
    e=env(io,sleeps);ob=e.reset();_,_,_,info=e.execute_chunk(ob,lambda *_:_plan(.9))
    assert sleeps==[] and io.published==[]
    assert info['step_trace'][0]['hil_sampling_mode']=='legacy'


def test_invalid_sampling_mode_or_timebase_refused(monkeypatch):
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING','bad')
    with pytest.raises(ValueError,match='HIL sampling'):env(FakeTask2IO([]),[])
    monkeypatch.setenv('COBOT_RLT_HIL_SAMPLING','logical20')
    with pytest.raises(ValueError,match='Replay 20 Hz'):env(FakeTask2IO([]),[],hz=30.)
    io=FakeTask2IO([_sample(0)]);e=env(io,[]);ob=e.reset()
    with pytest.raises(ValueError,match='Publication Hz'):e.execute_chunk(ob,lambda *_:_plan(0),control_hz=30.)
