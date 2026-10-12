import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.exploration import ChunkExploration,ExplorationConfig
from test_exploration_collection import enable
from test_async_execution import setup


def make():return ChunkExploration(ExplorationConfig(seed=42,mode='episode_constant'),np.zeros(7),np.ones(7)*.02)


def test_constant_after_ramp_and_same_timeline_overlap():
    e=make();a=np.zeros((50,7),np.float32)
    first,r=e.apply(a,context=(1,1),time_sec=0.)
    nxt,_=e.apply(a,context=(1,1),time_sec=.25)
    np.testing.assert_array_equal(first[5:],nxt[:45])
    np.testing.assert_array_equal(first[0],a[0])
    np.testing.assert_array_equal(first[:,6],a[:,6])
    assert np.all(np.abs(first[:,:6])<=2*e.sigma+1e-9)
    for t in [1.,2.,4.]:
        x,_=e.apply(a,context=(1,1),time_sec=t)
        np.testing.assert_array_equal(x[0],first[20])
    x,s=e.apply(a,context=(1,2),time_sec=7.)
    np.testing.assert_array_equal(x[0],a[0])
    np.testing.assert_array_equal(s['sampled_episode_offset_rad'],r['sampled_episode_offset_rad'])
    _,s=e.apply(a,context=(2,1),time_sec=8.)
    assert s['sampled_episode_offset_rad']!=r['sampled_episode_offset_rad']


def test_clock_and_configuration_fail_closed():
    e=make();a=np.zeros((10,7),np.float32)
    with pytest.raises(ValueError):e.apply(a,context=(1,1))
    with pytest.raises(ValueError):e.apply(a,context=(None,1),time_sec=0)
    with pytest.raises(ValueError):ExplorationConfig(mode='episode_constant',ramp_sec=.1)


def test_row_provenance_and_rtc_prefix_are_preserved(monkeypatch):
    engine,env,io,clock=setup(monkeypatch,hz=50);enable(engine);engine.explorer=make()
    try:
        plan=engine.diagnostic_plan(io.sample().observation,0)
        bundle=engine.plan_bundle(plan)
        np.testing.assert_array_equal(bundle[:,17:23],plan.exploration['offset_chunk_rad'])
        engine.fresh_plan(io.sample().observation)
        for _ in range(5):engine.queue.pop()
        old=engine.queue.remaining_actions().copy()
        engine.request_next(io.sample().observation);engine.future.result(timeout=2);engine.accept_result()
        np.testing.assert_array_equal(engine.queue.remaining_actions()[:4],old[:4])
    finally:engine.close()


def test_constant_replay_action_and_publication_provenance(monkeypatch):
    engine,env,io,clock=setup(monkeypatch,hz=50);enable(engine);engine.explorer=make()
    try:
        env.execute_chunk(io.sample().observation)
        assert io.records
        for row in io.records:
            np.testing.assert_array_equal(row['action'],row['publications'][-1]['action'])
            np.testing.assert_array_equal(row['exploration']['offset_rad'],row['publications'][-1]['exploration']['offset_rad'])
    finally:engine.close()


def test_constant_entry_is_isolated_and_fixed(monkeypatch):
    from integrations.cobot_runtime import exploration_collection as c
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID',c.CONSTANT_MODEL_ID)
    assert c.selected()
    assert c.locations()['run'].name=='hil8-7281-constant-v1'
    monkeypatch.setattr(c,'verify_assets',lambda *a:None)
    env=dict(COBOT_DEPLOYMENT_MODEL_ID=c.CONSTANT_MODEL_ID,COBOT_RLT_SHARED_MODEL='1',COBOT_RLT_EXECUTION_PROFILE='async_rtc50',COBOT_EVAL_SNAPSHOT='/fake/actor.pkl')
    c.validate_launch('frozen',env)
    with pytest.raises(ValueError):c.validate_launch('online',env)
