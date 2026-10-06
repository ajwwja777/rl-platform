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

def setup(monkeypatch, hz=40, state_at=None, reject=False, profile=None):
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE','faithful20')
    clock=Clock();io=IO(clock,state_at,reject)
    env=RightArmCobotOnlineEnv(io,chunk_exec_horizon=10,control_frequency_hz=20,
        max_episode_steps=None,joint_step_limit=.03,gripper_step_limit=.004,sleep=clock.sleep)
    env._runtime.arm()
    engine=AsyncExecution(env,'test',profile or ExecutionProfile(publish_hz=hz),clock)
    env._execution=engine;engine.backend=Backend()
    return engine,env,io,clock

@pytest.mark.parametrize("profile_name", ["async_rtc40", "async20_no_rtc_no_smoothing"])
def test_replay_features_match_full_rtc_input_not_only_proprio(monkeypatch, profile_name):
    from rlt_online_rl.inference import EnvDriver, ActorResponse
    from rlt_online_rl.config import RLTOnlineRLConfig, EnvDriverConfig
    from methods.openpi_rlt.cobot_adapter.execution_runtime import install
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmPolicyRuntime
    _, profile = selected_profile(environ={'COBOT_RLT_EXECUTION_PROFILE': profile_name})
    engine,env,io,clock=setup(monkeypatch,state_at=lambda t:('policy',False,'success' if t>=.8 else None), profile=profile)
    # EnvDriver performs reset/arm itself.
    env._runtime=RightArmPolicyRuntime(joint_step_limit=.03,gripper_step_limit=.004)
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE',profile_name)
    for key in ('__init__','run_episode','_append_raw_chunk','close'):
        monkeypatch.setattr(EnvDriver,key,getattr(EnvDriver,key))
    monkeypatch.setattr(EnvDriver,'_cobot_execution_installed',False,raising=False)
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT','trace')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT','off')
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    install()
    class Features:
        def __init__(self):self.seen=[]
        def get_features(self, obs):
            self.seen.append(obs)
            return dict(z_rl=np.full(2048,float(obs.get('rtc',{}).get('delay_steps',-1)),np.float32),proprio=obs['state'].copy(),
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
        assert any('rtc' in obs for obs in features.seen) == profile.rtc
        for transition in replay.rows:
            start=transition.step_id
            expected=np.stack([x['action'] for x in io.records[start:start+10]])
            np.testing.assert_array_equal(transition.action_chunk,expected)
            assert transition.action_chunk.shape==(10,7)
            assert transition.episode_id==11
            expected=float(io.records[start]['observation'].get('rtc',{}).get('delay_steps',-1))
            np.testing.assert_array_equal(transition.z_rl,np.full(2048,expected,np.float32))
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
