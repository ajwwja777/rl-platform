import json
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.cascade_filter import CascadedJointFilter
from methods.openpi_rlt.cobot_adapter.execution_profiles import ExecutionProfile,selected_profile,describe_execution,SMOOTH_MODEL_ID
from integrations.cobot_runtime import exploration_collection as selection
from test_async_execution import setup


def test_step_monotonic_no_overshoot_rate_bounded_and_gripper_untouched():
    f=CascadedJointFilter();f.reset(np.zeros(7));out=[]
    for i in range(300):
        q=f.apply(np.array([.1]*6+[.031],np.float32),np.zeros(7),.02);out.append(q)
    out=np.array(out)
    assert np.all(np.diff(out[:,:6],axis=0)>=0)
    assert np.max(out[:,:6])<=.100001
    assert np.max(np.diff(out[:,:6],axis=0))<=.012001
    np.testing.assert_allclose(out[-1,:6],.1,atol=2e-7)
    np.testing.assert_array_equal(out[:,6],np.full(300,np.float32(.031)))


def test_no_chunk_reset_and_reset_discards_both_stages():
    f=CascadedJointFilter();f.reset(np.zeros(7))
    for _ in range(15): f.apply(np.ones(7),np.zeros(7),.02)
    before=f.previous.copy();after=f.apply(-np.ones(7),before,.02)
    assert np.max(np.abs(after[:6]-before[:6]))<=.012001
    f.reset(np.full(7,.25))
    np.testing.assert_allclose(f.apply(np.full(7,.25),np.full(7,.25),.02),.25)
    f.reset();assert f.previous is None and f.stage is None

@pytest.mark.parametrize('tau,velocity',[(0,.6),(-1,.6),(.12,0),(float('nan'),.6)])
def test_invalid_settings(tau,velocity):
    with pytest.raises(ValueError):CascadedJointFilter(tau,velocity)


def test_invalid_input():
    f=CascadedJointFilter()
    for target,dt in [(np.full(7,np.nan),.02),(np.zeros(7),0),(np.zeros(7),float('nan')),(np.zeros(5),.02)]:
        with pytest.raises(ValueError):f.apply(target,np.zeros(7),dt)


def test_candidate_is_opt_in_and_storage_separate(monkeypatch):
    opts=dict(enabled=True,publish_hz=50,rtc=True,smoothing=True)
    env={'COBOT_DEPLOYMENT_MODEL_ID':SMOOTH_MODEL_ID,'COBOT_EXECUTION_OPTIONS':json.dumps(opts)}
    _,p=selected_profile(environ=env);assert p.smoothing_order==2 and p.smoothing_tau_sec==.12
    _,old=selected_profile(environ={**env,'COBOT_DEPLOYMENT_MODEL_ID':selection.MODEL_ID});assert old.smoothing_order==1 and old.smoothing_tau_sec==.08
    shown=describe_execution({'id':SMOOTH_MODEL_ID,'execution_options':opts});assert shown['smoothing_order']==2 and shown['smoothing_tau_sec']==.12
    for key,value in [('publish_hz',40),('rtc',False),('smoothing',False)]:
        with pytest.raises(ValueError):selected_profile(environ={**env,'COBOT_EXECUTION_OPTIONS':json.dumps({**opts,key:value})})
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID',selection.MODEL_ID);old=selection.locations()
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID',SMOOTH_MODEL_ID);new=selection.locations()
    assert new['run']!=old['run'] and new['recording']!=old['recording'] and selection.selected()


@pytest.mark.parametrize('hil',[False,True])
def test_async_real_loop_trace_and_hil_reset(monkeypatch,hil):
    def state(t):
        if hil and .15<=t<.25:return 'manual:right',True,None
        return 'policy',False,'failure' if t>=.8 else None
    engine,env,io,clock=setup(monkeypatch,profile=ExecutionProfile(publish_hz=50,smoothing_order=2,smoothing_tau_sec=.12),state_at=state)
    try:
        assert isinstance(engine.filter,CascadedJointFilter)
        while True:
            _,_,done,_=env.execute_chunk(io.sample().observation)
            if done:break
        assert io.records
        for row in io.records:
            assert row['execution_settings']['smoothing_order']==2
            if row['publications']:np.testing.assert_array_equal(row['action'],row['publications'][-1]['action'])
        assert not any(.15<=t<.25 for t,_ in io.published) if hil else True
        assert engine.filter.previous is None and engine.filter.stage is None
    finally:engine.close()


def test_smooth_entry_keeps_exact_input_cache(monkeypatch):
    from types import SimpleNamespace
    from methods.openpi_rlt.cobot_adapter.replay_feature_cache import enabled
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID',SMOOTH_MODEL_ID)
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT','trace')
    driver=SimpleNamespace(_env=SimpleNamespace(cobot_task2_contract=True),_env_config=SimpleNamespace(step_trace_stride=0))
    assert enabled(driver)
    monkeypatch.setenv('COBOT_RLT_REPLAY_EXACT_CACHE','0');assert not enabled(driver)


def test_smooth_launch_contract(monkeypatch):
    monkeypatch.setattr(selection,'verify_assets',lambda *a:None)
    env={'COBOT_DEPLOYMENT_MODEL_ID':SMOOTH_MODEL_ID,'COBOT_RLT_SHARED_MODEL':'1','COBOT_RLT_EXECUTION_PROFILE':'async_rtc50','COBOT_EVAL_SNAPSHOT':'/fake/actor_snapshot.pkl'}
    selection.validate_launch('frozen',env)
    with pytest.raises(ValueError):selection.validate_launch('online',env)
    root=Path(__file__).resolve().parents[1];models=json.loads((root/'configs/deployment_models.json').read_text())['models']
    old=next(m for m in models if m['id']==selection.MODEL_ID);new=next(m for m in models if m['id']==SMOOTH_MODEL_ID)
    for key in ('checkpoint','base_checkpoint','normalization','actor_version','capabilities','deterministic'):
        assert old.get(key)==new.get(key)
    assert new['mode']=='frozen' and not new['capabilities']['train']
