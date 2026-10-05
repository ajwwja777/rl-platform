"""Native episode regressions for the no-motion execution diagnostic tool."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.inference import ActorResponse
from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv
from methods.openpi_rlt.cobot_adapter.async_execution import AsyncExecution
from methods.openpi_rlt.cobot_adapter.execution_profiles import ExecutionProfile

spec=importlib.util.spec_from_file_location('execution_probe',Path(__file__).resolve().parents[1]/'scripts/validate_async_execution.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

class IO:
    shadow_mode=False
    def __init__(self):
        self.state=np.zeros(7,np.float32)
        self.records=[]
        self.commands=[]
    def sample(self):
        return SimpleNamespace(observation={'state':self.state.copy(),'images':{}},mode='policy',paused=False,outcome=None,timestamp=time.monotonic())
    def publish_policy_action(self,action):
        self.state=np.asarray(action).copy()
        self.commands.append(self.state.copy())
        return True
    def wait_armed(self):pass
    def mark_replay_finalized(self):pass
    def mark_terminal_pending(self,reason):pass
    def wait_terminal_outcome(self):return "failure"
    def set_chunk_ready(self,ready):pass
    def set_policy_paused(self,paused):pass
    def record_raw_step(self,record):self.records.append(record)
    def report_chunk(self,latency_sec,actor_version):pass

@pytest.mark.parametrize('asynchronous',[False,True])
def test_native_episode_stops_at30_and_never_uses_replay(monkeypatch,asynchronous):
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE','faithful20')
    io=IO()
    cfg=RLTOnlineRLConfig(action_dim=7,proprio_dim=7,z_dim=8,chunk_len=10)
    class Features:
        def get_features(self,ob):
            return {'z_rl':np.zeros(8,np.float32),'proprio':ob['state'].copy(),
                    'ref_chunk':np.tile(ob['state'],(10,1)),'rtc_used':'rtc' in ob}
    requests=[]
    class Actor:
        def infer(self,request):
            requests.append(request)
            return ActorResponse(np.asarray(request.ref_chunk,np.float32)+.001,2500,request.request_id)
    env=RightArmCobotOnlineEnv(io,chunk_exec_horizon=10,control_frequency_hz=20,max_episode_steps=30,collection_phase="online",
        joint_step_limit=.03,gripper_step_limit=.004,sleep=time.sleep if asynchronous else lambda seconds:None)
    engine=AsyncExecution(env,'async_rtc20',ExecutionProfile(publish_hz=20)) if asynchronous else None
    env._execution=engine
    if not asynchronous:
        from methods.openpi_rlt.cobot_adapter.execution_runtime import PlannerBackend
        def forbidden(*args,**kwargs):raise AssertionError('Faithful must use native planner, not async backend')
        monkeypatch.setattr(PlannerBackend,'plan',forbidden)
    try:
        result=probe.execute_probe_episode(env,Features(),Actor(),cfg)
        assert len(io.records)==30 and len(io.commands)==30
        assert result['eval_actor_only'] is True and result['transitions_written']==0
        assert result['fallback_count']==0 and result['intervention_count']==0
        assert result['actor_version_start']==result['actor_version_end']==2500
        assert result['actor_deterministic'] is True
        assert all(r.episode_id==1 and r.deterministic for r in requests)
        if not asynchronous:
            assert [r.step_id for r in requests]==[0,10,20]
    finally:
        if engine:engine.close()
