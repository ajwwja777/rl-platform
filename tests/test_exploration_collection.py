"""CPU-only frozen collection, provenance, RTC/HIL and private storage contracts."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.exploration import ChunkExploration, ExplorationConfig
from integrations.cobot_runtime import exploration_collection as selection
from test_async_execution import setup


def explorer(seed=19):
    return ChunkExploration(ExplorationConfig(seed=seed), np.zeros(7), np.full(7, .02))


def enable(engine):
    engine.explorer = explorer()
    engine.exploration_collecting = lambda: True
    engine.exploration_identity = {"learner_enabled": False}
    original = engine.backend.plan
    def plan(obs, step):
        result = original(obs, step)
        result.actor_param_version = 3640
        return result
    engine.backend.plan = plan


def test_offsets_bounded_correlated_and_gripper_bitwise_unchanged():
    e = explorer()
    action = np.arange(350, dtype=np.float32).reshape(50, 7) / 1000
    offsets = []
    for _ in range(3000):
        output, receipt = e.apply(action, context=(1, 1))
        offset = np.array(receipt['offset_rad'])
        assert np.all(np.abs(offset) <= 2*e.sigma+1e-9)
        np.testing.assert_array_equal(output[:, 6], action[:, 6])
        np.testing.assert_allclose(output[:, :6]-action[:, :6], np.tile(offset,(50,1)),atol=3e-8)
        offsets.append(offset)
    assert .35 < np.corrcoef(np.array(offsets)[:-1,1], np.array(offsets)[1:,1])[0,1] < .65
    _, receipt = e.apply(action, context=(1, 2))
    assert receipt['plan_index'] == 0
    with pytest.raises(ValueError): e.apply(np.full((10,7),np.nan), context=(2,1))


def test_rtc_prefix_preserves_its_original_offset_and_request(monkeypatch):
    engine, env, io, clock = setup(monkeypatch, hz=50)
    enable(engine)
    try:
        engine.fresh_plan(io.sample().observation)
        for _ in range(5): engine.queue.pop()
        old = engine.queue.remaining_actions().copy()
        engine.request_next(io.sample().observation)
        engine.future.result(timeout=2)
        engine.accept_result()
        new = engine.queue.remaining_actions()
        np.testing.assert_array_equal(new[:4],old[:4])
        assert new[4,16] != old[0,16]
        assert engine.exploration_receipt(new[0])['request_id'] == int(old[0,16])
        assert engine.exploration_receipt(new[4])['request_id'] == int(new[4,16])
    finally: engine.close()


def test_trace_is_actual_command_not_unfiltered_proposal(monkeypatch):
    engine, env, io, clock = setup(monkeypatch, hz=50)
    enable(engine)
    try:
        env.execute_chunk(io.sample().observation)
        assert all(row['exploration'] for row in io.records)
        for row in io.records:
            np.testing.assert_array_equal(row['action'], row['publications'][-1]['action'])
            assert row['exploration']['request_id'] == row['publications'][-1]['exploration']['request_id']
        assert any(not np.array_equal(row['action'],row['planned_action']) for row in io.records)
        events = [event for row in io.records for event in row['inference_events']]
        assert events and all('actor_target_before_exploration' in event['exploration'] for event in events)
    finally: engine.close()


def test_hil_and_evaluation_do_not_inherit_exploration(monkeypatch):
    def state(t):
        if .15<=t<.25:return 'manual:right',True,None
        return 'policy',False,'failure' if t>=.4 else None
    engine, env, io, clock = setup(monkeypatch, hz=50, state_at=state)
    enable(engine)
    try:
        env.execute_chunk(io.sample().observation)
        hil = [r for r in io.records if r['human_controlled'] and not r['publications']]
        assert hil and all(r['exploration'] is None for r in hil)
        assert not any(.15<=t<.25 for t,_ in io.published)
    finally: engine.close()
    engine, env, io, clock = setup(monkeypatch, hz=50)
    enable(engine); engine.exploration_collecting=lambda:False
    try:
        plan=engine.diagnostic_plan(io.sample().observation,0)
        assert plan.exploration is None
        np.testing.assert_array_equal(plan.action_chunk,plan.ref_chunk)
        assert engine.explorer.index == 0
    finally: engine.close()


def test_changed_actor_or_fallback_fails_closed(monkeypatch):
    engine, env, io, clock = setup(monkeypatch)
    engine.explorer=explorer(); engine.exploration_collecting=lambda:True
    try:
        with pytest.raises(RuntimeError,match='pinned'): engine.diagnostic_plan(io.sample().observation,0)
        assert not io.published
    finally: engine.close()


def test_private_config_and_learner_guard(monkeypatch,tmp_path):
    import yaml
    from integrations.cobot_runtime import shared_model_env, paths
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(paths,'RLT',root)
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID',selection.MODEL_ID)
    private=dict(run=tmp_path/'private',recording=tmp_path/'recordings')
    monkeypatch.setattr(selection,'locations',lambda:private)
    snapshot=tmp_path/'assets/actor_snapshot/actor_snapshot.pkl';snapshot.parent.mkdir(parents=True);snapshot.write_bytes(b'fake')
    (snapshot.parent.parent/'action_norm_stats.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(selection,'verify_assets',lambda *a:calls.append(a))
    target=tmp_path/'config.yaml'
    shared_model_env.prepare_config('frozen',str(snapshot),target)
    config=yaml.safe_load(target.read_text())
    rt=config['runtime']; assert calls
    assert rt['actor_service']['snapshot_path']==str(snapshot)
    assert rt['learner_service']['actor_snapshot_path'] != str(snapshot)
    assert rt['replay']['journal_path']==str(private['run']/'replay/collection.pkl')
    assert rt['env_driver']['actor_deterministic'] is True
    env=dict(COBOT_DEPLOYMENT_MODEL_ID=selection.MODEL_ID,RLT_DISABLE_LEARNER='1',COBOT_RLT_DISABLE_PHASE_CONTROLLER='1',
             COBOT_RLT_HOLD_RIGHT_GRIPPER='1',COBOT_RLT_REPLAY_ACTION_PRECISION='float32',COBOT_RLT_HIL_SAMPLING='logical20',
             COBOT_RLT_INPUT_AUDIT='strict',COBOT_RLT_RAW_OBSERVATION_CONTRACT='trace',
             RLT_OUTPUT_DIR=str(private['run']/'online'),COBOT_RLT_TRACE_DIR=str(private['run']/'traces'))
    selection.validate_runtime(target,env)
    for key in ('RLT_DISABLE_LEARNER','COBOT_RLT_DISABLE_PHASE_CONTROLLER','COBOT_RLT_INPUT_AUDIT','RLT_OUTPUT_DIR'):
        with pytest.raises(ValueError): selection.validate_runtime(target,{**env,key:'wrong'})
    with pytest.raises(ValueError): selection.configure(config,'online')
    monkeypatch.delenv('COBOT_DEPLOYMENT_MODEL_ID')
    original=copy.deepcopy(config)
    assert selection.configure(config,'online') == original
    selection.validate_runtime('does-not-exist',{})


def test_asset_identity_rejects_changed_snapshot(tmp_path):
    fake=tmp_path/'fake';fake.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):selection.verify_assets(fake,fake)


def test_new_entry_is_opt_in_without_training_seed():
    root=Path(__file__).resolve().parents[1]
    new=json.loads((root/'configs/deployment_models.json').read_text())
    row=next(x for x in new['models'] if x['id']==selection.MODEL_ID)
    assert row['mode']=='frozen' and row['deterministic'] and not row['capabilities']['train']
    assert not row['online_seed_available'] and 'collection_adapter' not in row


def test_attachment_requires_collection_lifecycle_and_keeps_evaluation_quiet(monkeypatch):
    from types import SimpleNamespace
    from rlt_online_rl import inference
    from methods.openpi_rlt.cobot_adapter.exploration import attach
    engine, env, io, clock=setup(monkeypatch)
    lifecycle=SimpleNamespace(collecting=False)
    io._session_application=SimpleNamespace(_task5=lifecycle)
    driver=SimpleNamespace(_env=env,_env_config=SimpleNamespace(actor_deterministic=True),_rl_config=None)
    monkeypatch.setattr(inference,'ActionRepresentationAdapter',SimpleNamespace(from_config=lambda _:SimpleNamespace(stats=SimpleNamespace(q01=np.zeros(7),q99=np.full(7,.02)))))
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID',selection.MODEL_ID)
    monkeypatch.setenv('RLT_DISABLE_LEARNER','1')
    try:
        attach(engine,driver)
        assert engine.explorer is not None and not engine.exploration_collecting()
        lifecycle.collecting=True
        assert engine.exploration_collecting()
        assert engine.stats['learner_enabled'] is False
        monkeypatch.setenv('RLT_DISABLE_LEARNER','0')
        with pytest.raises(ValueError,match='disabled'):attach(engine,driver)
    finally:engine.close()


def test_launch_rejects_training_or_changed_execution(monkeypatch):
    monkeypatch.setattr(selection,'verify_assets',lambda *args:None)
    env=dict(COBOT_DEPLOYMENT_MODEL_ID=selection.MODEL_ID,COBOT_RLT_SHARED_MODEL='1',
             COBOT_RLT_EXECUTION_PROFILE='async_rtc50',COBOT_EVAL_SNAPSHOT='/fake/actor_snapshot.pkl')
    selection.validate_launch('frozen',env)
    with pytest.raises(ValueError): selection.validate_launch('online',env)
    with pytest.raises(ValueError): selection.validate_launch('frozen',{**env,'COBOT_RLT_ONLINE_SEED':'/seed'})
    with pytest.raises(ValueError): selection.validate_launch('frozen',{**env,'COBOT_RLT_EXECUTION_PROFILE':'faithful20'})


def test_disk_trace_retains_exploration(monkeypatch,tmp_path):
    from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter
    engine,env,io,clock=setup(monkeypatch,hz=50)
    enable(engine)
    try:
        env.execute_chunk(io.sample().observation)
        writer=AtomicEpisodeTraceWriter(tmp_path/'traces')
        for row in io.records:writer.append(row)
        writer.finalize('success',identity={'episode_uuid':'mock-episode'})
        lines=[json.loads(x) for p in (tmp_path/'traces').glob('*success.jsonl') for x in p.read_text().splitlines()]
        assert len(lines)==len(io.records) and lines[-1]['episode_uuid']=='mock-episode'
        for row in lines:
            assert row['exploration']['request_id']==row['publications'][-1]['exploration']['request_id']
    finally:engine.close()
