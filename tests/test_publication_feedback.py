"""Real ROS sampling methods, synthetic messages/clock, no ROS or hardware."""
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO
from test_async_execution import setup


def sampler(clock, decode=None):
    io = object.__new__(RosTask2IO)
    io._condition = threading.Condition(threading.RLock())
    io.ros = SimpleNamespace(Time=SimpleNamespace(now=lambda: SimpleNamespace(to_sec=lambda:100+clock())))
    io._images = {}; io._joints = {}
    io._mode='policy'; io._paused=False; io._outcome=None; io._prompt='insert'; io._max_sync_skew_sec=.15
    io.bridge=SimpleNamespace(imgmsg_to_cv2=decode or (lambda *args:np.zeros((3,4,3),np.uint8)))
    def refresh():
        for key in ('cam_high','cam_left_wrist','cam_right_wrist'):
            io._images[key]=SimpleNamespace(header=SimpleNamespace(stamp=100+clock()))
        for side in ('left','right'):
            io._joints[side]=SimpleNamespace(header=SimpleNamespace(stamp=100+clock()),position=np.zeros(7))
    refresh()
    return io,refresh


def test_publication_feedback_never_decodes_or_consumes_rgb_cache():
    decoded=[]
    io,refresh=sampler(lambda:0.,lambda *args:decoded.append(1) or np.zeros((3,4,3),np.uint8))
    feedback=io.sample_publication_control(right_arm_only=True)
    assert 'images' not in feedback.observation and decoded==[]
    assert not hasattr(io,'_control_image_key')
    first=io.sample_control(right_arm_only=True)
    assert len(decoded)==3
    io._joints['right'].position=np.arange(7)
    refresh()
    io._joints['right'].position=np.arange(7)
    second=io.sample_publication_control(right_arm_only=True)
    np.testing.assert_array_equal(second.observation['state'],np.arange(7))
    assert len(decoded)==3
    last=io.sample_control(right_arm_only=True)
    assert len(decoded)==6 and last.observation['images'] is not first.observation['images']


@pytest.mark.parametrize('fault',['missing_camera','stale_camera','future_camera','skew','bad_joint','shutdown'])
def test_light_feedback_preserves_input_guards(fault):
    io,_=sampler(lambda:0.)
    if fault=='missing_camera':io._images.pop('cam_high')
    if fault=='stale_camera':io._images['cam_high'].header.stamp=99.
    if fault=='future_camera':io._images['cam_high'].header.stamp=101.
    if fault=='skew':io._images['cam_high'].header.stamp=99.82
    if fault=='bad_joint':io._joints['right'].position=np.full(7,np.nan)
    if fault=='shutdown':io.ros.is_shutdown=lambda:True
    with pytest.raises(RuntimeError):io.sample_publication_control(right_arm_only=True)


def test_light_feedback_retains_pause_hil_and_one_shot_outcome():
    io,_=sampler(lambda:0.)
    io._mode='manual:right';io._paused=True;io._outcome='success'
    sample=io.sample_publication_control(right_arm_only=True)
    assert sample.mode=='manual:right' and sample.paused and sample.outcome=='success'
    assert io.sample_control(right_arm_only=True).outcome is None


@pytest.mark.parametrize('hz',[20,30,40,50])
def test_rgb_decode_cost_does_not_accumulate_in_physical_sends(monkeypatch,hz):
    engine,env,fake,clock=setup(monkeypatch,hz)
    calls=[]
    def decode(*args):
        calls.append(clock());clock.value+=.001
        return np.zeros((3,4,3),np.uint8)
    ros,refresh=sampler(clock,decode)
    def sync():
        refresh();ros._joints['right'].position=fake.state.copy()
    def full(**kwargs):sync();return ros.sample_control(**kwargs)
    def light(**kwargs):sync();return ros.sample_publication_control(**kwargs)
    fake.sample_control=full;fake.sample_publication_control=light
    try:
        for _ in range(12):
            _,rewards,done,_=env.execute_chunk(fake.sample().observation)
            assert not done and len(rewards)==10
        assert len(fake.records)==120 and len(fake.published)==hz*6
        assert np.diff([t for t,_ in fake.published]).min()>=1/hz-1e-9
        assert engine.stats['publication_sample_mode']=='feedback_only'
        assert engine.stats['max_control_sample_ms']==0
        assert engine.stats['max_observation_sample_ms']==pytest.approx(3.)
        assert engine.clock_window_shift_sec<.05
        assert calls
        for row in fake.records:
            assert 'images' in row['next_observation']
            np.testing.assert_array_equal(row['action'],row['publications'][-1]['action'])
        for obs,_ in engine.backend.requests:
            assert 'images' in obs
    finally:engine.close()


def test_light_feedback_stall_still_stops_before_publish(monkeypatch):
    engine,env,fake,clock=setup(monkeypatch,50)
    ros,refresh=sampler(clock)
    def light(**kwargs):
        refresh()
        if engine.last_wait_deadline is not None:clock.value+=.2;refresh()
        return ros.sample_publication_control(**kwargs)
    fake.sample_publication_control=light
    try:
        with pytest.raises(RuntimeError,match='control preparation exceeded budget'):
            env.execute_chunk(fake.sample().observation)
        assert not fake.published and fake.pauses[-1] and not fake.records
    finally:engine.close()


def test_background_health_check_is_not_on_publication_path(monkeypatch):
    engine,env,fake,clock=setup(monkeypatch,50)
    entered,release=threading.Event(),threading.Event()
    def check(**kwargs):entered.set();assert release.wait(2)
    fake.report_chunk=check
    try:
        engine.report_chunk(.14,3500);assert entered.wait(1)
        env.execute_chunk(fake.sample().observation)
        assert len(fake.published)==25 and len(fake.records)==10
        assert not release.is_set()
    finally:release.set();engine.close()


@pytest.mark.parametrize('mode',['policy','manual:right'])
def test_actual_light_probe_respects_pause_hil_and_latches_terminal(monkeypatch,mode):
    consumed=[]
    def state(t):
        if .10<=t<.20:return mode,True,None
        outcome='failure' if t>=.36 and not consumed else None
        if outcome:consumed.append(True)
        return 'policy',False,outcome
    engine,env,fake,clock=setup(monkeypatch,50,state_at=state)
    ros,refresh=sampler(clock)
    def light(**kwargs):
        refresh();sample=fake.sample()
        ros._mode=sample.mode;ros._paused=sample.paused;ros._outcome=sample.outcome
        ros._joints['right'].position=fake.state.copy()
        return ros.sample_publication_control(**kwargs)
    fake.sample_publication_control=light
    try:
        _,_,done,info=env.execute_chunk(fake.sample().observation)
        assert done and info['outcome']=='failure' and fake.records[-1]['done']
        assert not any(.10<=t<.20 or t>=.36 for t,_ in fake.published)
        assert 'last_error' not in engine.stats
    finally:engine.close()
