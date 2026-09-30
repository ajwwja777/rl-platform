"""No ROS, sockets, robot motion or production Replay writes."""
from types import SimpleNamespace
import time
import threading
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.async_execution import AsyncExecution
from methods.openpi_rlt.cobot_adapter.execution_profiles import ExecutionProfile, selected_profile
from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv

class Clock:
    def __init__(self): self.value=0.
    def __call__(self): return self.value
    def sleep(self, dt):
        self.value += dt
        time.sleep(.002)  # Give the real inference worker scheduling time.

class IO:
    shadow_mode=False
    def __init__(self, clock, state_at=None, reject=False):
        self.clock=clock; self.state_at=state_at; self.reject=reject
        self.state=np.zeros(7,np.float32); self.published=[]; self.records=[]; self.pauses=[]
    def sample(self):
        mode,paused,outcome=('policy',False,None) if self.state_at is None else self.state_at(self.clock())
        return SimpleNamespace(observation={'state':self.state.copy(),'images':{}},
                               mode=mode,paused=paused,outcome=outcome,timestamp=self.clock())
    def publish_policy_action(self, action):
        if self.reject: return False
        self.state=np.asarray(action).copy(); self.published.append((self.clock(),self.state.copy()))
        return True
    def set_chunk_ready(self, ready): pass
    def set_policy_paused(self, paused): self.pauses.append(paused)
    def record_raw_step(self, record): self.records.append(record)
    def report_chunk(self, **kwargs): pass
    def wait_armed(self): pass
    def wait_episode_ready(self): pass
    def mark_replay_finalized(self): pass

class Backend:
    episode_id=1
    def __init__(self): self.requests=[]
    def plan(self, observation, step):
        self.requests.append((observation,step))
        actions=np.full((10,7),.04 if len(self.requests)%2 else -.04,np.float32)
        actions[:,6]=.005
        features=SimpleNamespace(z_rl=np.zeros(2048,np.float32),
                                 proprio=observation['state'].copy(),ref_chunk=actions.copy())
        return SimpleNamespace(action_chunk=actions,ref_chunk=actions.copy(),
                               source=1,actor_param_version=3500,start_features=features)

def setup(monkeypatch, hz=40, state_at=None, reject=False):
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE','faithful20')
    clock=Clock();io=IO(clock,state_at,reject)
    env=RightArmCobotOnlineEnv(io,chunk_exec_horizon=10,control_frequency_hz=20,
        max_episode_steps=None,joint_step_limit=.03,gripper_step_limit=.004,sleep=clock.sleep)
    env._runtime.arm()
    engine=AsyncExecution(env,'test',ExecutionProfile(publish_hz=hz),clock)
    env._execution=engine;engine.backend=Backend()
    return engine,env,io,clock

@pytest.mark.parametrize('hz',[20,30,40,50])
def test_real_publication_grid_and_logical_replay(monkeypatch,hz):
    engine,env,io,clock=setup(monkeypatch,hz)
    try:
        _,rewards,done,info=env.execute_chunk(io.sample().observation)
        assert not done and len(rewards)==len(info['step_trace'])==10
        assert len(io.published)==hz//2
        np.testing.assert_allclose(np.diff([t for t,_ in io.published]),1/hz,atol=1e-10)
        assert clock()==pytest.approx(.5)
        for row in io.records:
            assert row['publications']
            np.testing.assert_array_equal(row['action'],row['publications'][-1]['action'])
        commands=np.stack([a for _,a in io.published])
        assert np.max(np.abs(np.diff(commands[:,:6],axis=0))) <= .6/hz+1e-7
        assert np.max(np.abs(np.diff(commands[:,6]))) <= .08/hz+1e-7
        rtc=[o['rtc'] for o,_ in engine.backend.requests if 'rtc' in o]
        assert rtc and rtc[0]['previous_actions'].shape==(5,7)
        assert rtc[0]['delay_steps']==4
    finally: engine.close()

def test_inference_overlaps_emission_and_stale_epoch_cannot_commit(monkeypatch):
    engine,env,io,clock=setup(monkeypatch)
    blocked=threading.Event(); release=threading.Event()
    real=engine.backend.plan
    def planner(obs,step):
        if step:
            blocked.set(); assert release.wait(2)
        return real(obs,step)
    engine.backend.plan=planner
    try:
        engine.fresh_plan(io.sample().observation)
        for _ in range(5): engine.queue.pop()
        env._episode_steps=5; engine.request_next(io.sample().observation)
        assert blocked.wait(1)
        # Waiting for a remote result does not own or lock the command queue.
        assert engine.queue.pop().shape==(16,)
        engine.invalidate(); release.set()
        engine.future.result(timeout=2); engine.accept_result()
        assert engine.plan is None and engine.queue.remaining_actions().size==0
        assert engine.stats['stale_results']==1
    finally: release.set(); engine.close()

def test_pause_hil_fresh_resume_and_terminal(monkeypatch):
    def state(t):
        if .10<=t<.15:return 'policy',True,None
        if .15<=t<.25:return 'manual:right',True,None
        return 'policy',False,'failure' if t>=.40 else None
    engine,env,io,clock=setup(monkeypatch,state_at=state)
    try:
        _,_,done,info=env.execute_chunk(io.sample().observation)
        assert done and info['outcome']=='failure'
        assert not any(.10<=t<.25 for t,_ in io.published)
        assert any(r['human_controlled'] for r in io.records)
        assert any(step>=2 and 'rtc' not in obs for obs,step in engine.backend.requests)
        assert io.records[-1]['done']
        assert engine.plan is None
    finally:engine.close()

def test_publish_rejected_cannot_fabricate_replay(monkeypatch):
    engine,env,io,clock=setup(monkeypatch,reject=True)
    try:
        with pytest.raises(RuntimeError,match='rejected'):
            env.execute_chunk(io.sample().observation)
        assert not io.records and not io.published and io.pauses[-1]
    finally:engine.close()

def test_delay_overrun_pauses_instead_of_replaying_old_chunk(monkeypatch):
    engine,env,io,clock=setup(monkeypatch)
    try:
        engine.fresh_plan(io.sample().observation)
        for _ in range(5):engine.queue.pop()
        engine.request_next(io.sample().observation)
        engine.future.result(timeout=2)
        for _ in range(5):engine.queue.pop()
        with pytest.raises(RuntimeError,match='delay exceeded'):engine.accept_result()
    finally:engine.close()

def test_late_clock_cannot_burst_commands(monkeypatch):
    engine,env,io,clock=setup(monkeypatch)
    try:
        clock.value=.1
        with pytest.raises(RuntimeError,match='deadline'):engine.wait_until(.05)
        assert not io.published
    finally:engine.close()

def test_registry_and_explicit_rollback():
    assert selected_profile(environ={'COBOT_RLT_EXECUTION_PROFILE':'faithful20'}) is None
    for hz in (20,30,40,50):
        _,profile=selected_profile(environ={'COBOT_RLT_EXECUTION_PROFILE':f'async_rtc{hz}'})
        assert profile.publish_hz==hz and profile.logical_hz==20
    with pytest.raises(ValueError):ExecutionProfile(publish_hz=60)

def test_actual_envdriver_replay_keeps_emitted_actions_and_rtc_anchors(monkeypatch):
    from rlt_online_rl.inference import EnvDriver, ActorResponse
    from rlt_online_rl.config import RLTOnlineRLConfig, EnvDriverConfig
    from methods.openpi_rlt.cobot_adapter.execution_runtime import install
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmPolicyRuntime
    engine,env,io,clock=setup(monkeypatch,state_at=lambda t:('policy',False,'success' if t>=.8 else None))
    # EnvDriver performs reset/arm itself.
    env._runtime=RightArmPolicyRuntime(joint_step_limit=.03,gripper_step_limit=.004)
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE','async_rtc40')
    for key in ('__init__','run_episode','_append_raw_chunk','close'):
        monkeypatch.setattr(EnvDriver,key,getattr(EnvDriver,key))
    monkeypatch.setattr(EnvDriver,'_cobot_execution_installed',False,raising=False)
    install()
    class Features:
        def __init__(self):self.seen=[]
        def get_features(self, obs):
            self.seen.append(obs)
            return dict(z_rl=np.zeros(2048,np.float32),proprio=obs['state'].copy(),
                        ref_chunk=np.full((10,7),.02,np.float32),rtc_used='rtc' in obs)
    class Actor:
        def infer(self, request):
            return ActorResponse(np.full((10,7),.04,np.float32),3500,request.request_id)
    class Replay:
        def __init__(self):self.rows=[]
        def add_transitions(self, rows):self.rows.extend(rows)
    features=Features();replay=Replay()
    driver=EnvDriver(env,features,Actor(),replay,RLTOnlineRLConfig(),
                     EnvDriverConfig(control_frequency_hz=20,step_trace_stride=1))
    monkeypatch.setattr(driver,'_persist_raw_episode',lambda *args,**kwargs:None)
    try:
        result=driver.run_episode(11)
        assert result['success']==1 and replay.rows
        assert any('rtc' in obs for obs in features.seen)
        for transition in replay.rows:
            start=transition.step_id
            expected=np.stack([x['action'] for x in io.records[start:start+10]])
            np.testing.assert_array_equal(transition.action_chunk,expected)
            assert transition.action_chunk.shape==(10,7)
            assert transition.episode_id==11
        # The planner requested step5 while step6+ was executing. That anchor
        # must retain the request's observation, never completion-time state.
        assert 5 not in engine.anchors  # consumed by raw chunk append
        # Verify that the unchanged native learner can train on the resulting
        # RTC-conditioned Replay schema, entirely in memory.
        import jax
        from rlt_online_rl import trainer
        cfg=RLTOnlineRLConfig(actor_hidden_dim=32,critic_hidden_dim=32)
        state,actor,critic=trainer.init_train_state(cfg,rng=jax.random.PRNGKey(42))
        rows=[transition.to_numpy() for transition in replay.rows[:2]]
        batch={key:np.stack([row[key] for row in rows]) for key in rows[0]}
        for _ in range(2):
            state,metrics=trainer.train_step(state,batch,actor=actor,critic=critic,rl_config=cfg)
        assert int(state.global_step)==2 and int(state.actor_version)==1
        assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())

    finally:driver.close()

def test_stage1_without_rtc_confirmation_cannot_run_candidate(monkeypatch):
    from methods.openpi_rlt.cobot_adapter.execution_runtime import PlannerBackend
    class Provider:
        def get_features(self,obs):return {'rtc_used':False}
    backend=PlannerBackend(SimpleNamespace(_feature_provider=Provider()))
    with pytest.raises(RuntimeError,match='did not confirm'):
        backend.plan({'rtc':{}},0)

def test_committed_prefix_retains_original_actor_version(monkeypatch):
    engine,env,io,clock=setup(monkeypatch)
    try:
        engine.fresh_plan(io.sample().observation)
        for _ in range(5):engine.queue.pop()
        original=engine.backend.plan
        def new_version(obs,step):
            result=original(obs,step);result.actor_param_version=3600;return result
        engine.backend.plan=new_version
        engine.request_next(io.sample().observation)
        engine.future.result(timeout=2);engine.accept_result()
        remaining=engine.queue.remaining_actions()
        assert (remaining[:4,14]==3500).all()
        assert (remaining[4:,14]==3600).all()
    finally:engine.close()

def test_control_peek_reuses_images_without_new_frames_and_rejects_stale(monkeypatch):
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import RosTask2IO
    import threading
    io=object.__new__(RosTask2IO)
    io._condition=threading.Condition(threading.RLock())
    io.ros=SimpleNamespace()
    converted=[]
    def image(message,encoding):
        converted.append(message);return np.zeros((224,224,3),np.uint8)
    io.bridge=SimpleNamespace(imgmsg_to_cv2=image)
    stamp=100.
    io._images={k:SimpleNamespace(header=SimpleNamespace(stamp=stamp)) for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
    io._joints={k:SimpleNamespace(header=SimpleNamespace(stamp=stamp),position=np.zeros(7)) for k in ('left','right')}
    io._mode='policy';io._paused=False;io._outcome=None;io._prompt='test';io._max_sync_skew_sec=.15
    monkeypatch.setattr('methods.openpi_rlt.cobot_adapter.cobot_ros1.time.time',lambda:100.01)
    first=io.sample_control(right_arm_only=True)
    io._joints['right'].position=np.ones(7)
    second=io.sample_control(right_arm_only=True)
    assert first.observation['state'].shape==(7,)
    assert (second.observation['state']==1).all()
    assert len(converted)==3
    assert first.observation['images'] is second.observation['images']
    monkeypatch.setattr('methods.openpi_rlt.cobot_adapter.cobot_ros1.time.time',lambda:100.3)
    with pytest.raises(RuntimeError,match='stale'):
        io.sample_control(right_arm_only=True)

def test_one_shot_terminal_seen_inside_publication_probe_is_not_lost(monkeypatch):
    consumed=[]
    def state(t):
        terminal='failure' if t>=.075 and not consumed else None
        if terminal:consumed.append(True)
        return 'policy',False,terminal
    engine,env,io,clock=setup(monkeypatch,state_at=state)
    try:
        _,_,done,info=env.execute_chunk(io.sample().observation)
        assert done and info['outcome']=='failure'
        assert io.records[-1]['done']
        assert not any(t>=.075 for t,_ in io.published)
    finally:engine.close()

def test_unrelated_left_teach_does_not_label_right_actions_as_human(monkeypatch):
    def state(t):
        if .10<=t<.25:return 'manual:left',True,None
        return 'policy',False,'failure' if t>=.40 else None
    engine,env,io,clock=setup(monkeypatch,state_at=state)
    try:
        env.execute_chunk(io.sample().observation)
        left_rows=[r for r in io.records if r['expert_mask']==[True,False]]
        assert left_rows
        assert all(not r['human_controlled'] for r in left_rows)
        assert not any(.10<=t<.25 for t,_ in io.published)
    finally:engine.close()

@pytest.mark.parametrize("mode", ["policy", "manual:right"])
def test_pause_or_hil_after_blocked_publication_cancels_old_deadline(monkeypatch, mode):
    # Reproduce an operator pause arriving during a blocked ROS publication.
    # The old 50Hz deadline is now late, but authority has already been revoked.
    def state(t):
        if .20 <= t < .30:
            return mode, True, None
        return 'policy', False, 'failure' if t >= .45 else None
    engine, env, io, clock = setup(monkeypatch, 50, state_at=state)
    original = io.publish_policy_action
    blocked = []
    def publish(action):
        result = original(action)
        if not blocked:
            blocked.append(True)
            clock.value += .20
        return result
    io.publish_policy_action = publish
    try:
        _, _, done, info = env.execute_chunk(io.sample().observation)
        assert done and info["outcome"] == "failure"
        assert not any(.20 <= t < .30 for t, _ in io.published)
        assert "last_error" not in engine.stats
        assert engine.stats["inference_requests"] >= 2
    finally:
        engine.close()

def test_active_blocked_publisher_still_fails_without_catchup(monkeypatch):
    engine, env, io, clock = setup(monkeypatch, 50)
    original = io.publish_policy_action
    def publish(action):
        result = original(action)
        clock.value += .20
        return result
    io.publish_policy_action = publish
    try:
        with pytest.raises(RuntimeError, match="late_ms="):
            env.execute_chunk(io.sample().observation)
        assert len(io.published) == 1 and io.pauses[-1]
    finally:
        engine.close()

def test_recorder_http_does_not_block_inference_or_queue_unbounded_checks(monkeypatch):
    engine, env, io, clock = setup(monkeypatch, 50)
    entered, release = threading.Event(), threading.Event()
    def blocked_report(**kwargs):
        entered.set()
        assert release.wait(2)
    io.report_chunk = blocked_report
    try:
        engine.report_chunk(.14, 3500)
        assert entered.wait(1)
        first = engine.health_future
        engine.report_chunk(.14, 3500)
        assert engine.health_future is first
        engine.health_started = time.monotonic() - 1.1
        engine.check_report()
        assert io.pauses[-1] is True
        assert engine.stats['recorder_warning'] == 'task5_recorder_health_timeout'
        assert engine.health_future is first
    finally:
        release.set()
        engine.close()

def test_pause_racing_between_precheck_and_clock_does_not_fault(monkeypatch):
    def state(t):
        return "policy", .20 <= t < .30, "failure" if t >= .45 else None
    engine, env, io, clock = setup(monkeypatch, 50, state_at=state)
    wait = engine.wait_until
    stalled = []
    def race(timestamp):
        if io.published and not stalled:
            stalled.append(True)
            clock.value += .20
        return wait(timestamp)
    engine.wait_until = race
    try:
        _, _, done, info = env.execute_chunk(io.sample().observation)
        assert done and info["outcome"] == "failure"
        assert "last_error" not in engine.stats
        assert not any(.20 <= t < .30 for t, _ in io.published)
    finally:
        engine.close()
