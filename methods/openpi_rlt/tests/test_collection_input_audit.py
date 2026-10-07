import json
from types import SimpleNamespace
import numpy as np
import pytest
from integrations.cobot_runtime.shared_model_env import CollectionTrace
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter
from methods.openpi_rlt.tests.test_raw_observation_contract import native_driver


def episode(tmp_path, monkeypatch, source, success, mode, snapshots):
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE', 'faithful20')
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'trace')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT', mode)
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_COUNT', str(snapshots))
    monkeypatch.setenv('COBOT_RLT_INPUT_SNAPSHOT_ROOT', str(tmp_path/'snapshots'))
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.replay import RawEpisodeTrace, RLTTransition
    from methods.openpi_rlt.cobot_adapter.replay_precision import install_action_precision_patch
    monkeypatch.setattr(RLTTransition, 'to_numpy', RLTTransition.to_numpy)
    install_action_precision_patch(enabled=True)
    trace = AtomicEpisodeTraceWriter(tmp_path/'trace')
    lifecycle = SimpleNamespace(collecting=True)
    wrapped = CollectionTrace(trace, lifecycle)
    d = native_driver(tmp_path)
    d._env._io._trace_writer = wrapped
    d._env.replay_commit_allowed = lambda: lifecycle.collecting
    d._rl_config = RLTOnlineRLConfig(action_dim=7, proprio_dim=7, z_dim=2048, chunk_len=10)
    def obs(i):
        return {'state':np.asarray([.001*i]*6+[.005],np.float32),
                'prompt':'Insert the held plug into the socket.',
                'images':{'base_0_rgb':np.full((2,2,3),i,np.uint8)}}
    def features(o):
        return dict(z_rl=np.full(2048,o['state'][0],np.float32),proprio=o['state'].copy(),
                    ref_chunk=np.full((10,7),.0123,np.float32))
    d._feature_provider = SimpleNamespace(get_features=features)
    observations=[obs(i) for i in range(21)]
    records=[dict(observation=observations[i],next_observation=observations[i+1],
                  action=np.asarray([.002+i*.0001]*6+[.005],np.float32),
                  ref_action=np.full(7,.0123,np.float32),reward=float(i==19 and success),
                  done=i==19,outcome=('success' if success else 'failure') if i==19 else None,
                  human_controlled=source==2,source=source,
                  actor_param_version=-1 if source==2 else 3500) for i in range(20)]
    raw=RawEpisodeTrace(31,10,[observations[0]],[],[])
    index=0
    wrapped.start_episode()
    for chunk in range(2):
        rows=records[chunk*10:(chunk+1)*10]
        native_rows=d._build_trace_records(rows,episode_id=31,start_env_step_id=chunk*10,
                                          chunk_success=int(success),collection_phase='online')
        index=d._append_raw_chunk(raw,observation_idx=index,trace_records=native_rows,
            chunk_step_id=chunk,chunk_source=source,collection_phase='online',done=chunk==1,
            success=int(success),drop_transition=False,start_features=None,
            policy_anchor_offsets=[],policy_anchor_features=[])
        for row in rows:wrapped.append(row)
    return d,raw,records,wrapped,lifecycle


@pytest.mark.parametrize('mode',['record','strict'])
@pytest.mark.parametrize('source',[1,2])
@pytest.mark.parametrize('success',[False,True])
@pytest.mark.parametrize('snapshots',[0,1])
def test_real_collection_wrapper_c10_episode_reaches_native_journal_and_batch(
        tmp_path,monkeypatch,mode,source,success,snapshots):
    d,raw,records,wrapped,_=episode(tmp_path,monkeypatch,source,success,mode,snapshots)
    transitions,stats=d._build_episode_replay(raw)
    assert stats['replay_transition_count']==len(transitions)==2
    assert [t.step_id for t in transitions]==[0,10]
    for t in transitions:
        np.testing.assert_array_equal(t.action_chunk,np.stack([r['action'] for r in records[t.step_id:t.step_id+10]]))
        assert t.episode_id==31 and bool(t.intervention_flag)==(source==2)
        assert t.source==source
    assert not transitions[0].done and transitions[1].done
    assert transitions[1].rewards[-1]==float(success)
    receipts=[json.loads(line)for p in (tmp_path/'trace/replay_inputs').glob('*.jsonl')for line in p.read_text().splitlines()]
    assert len(receipts)==2 and all(r['checks_passed'] for r in receipts)
    assert receipts[0]['raw_step_indices']==list(range(10))
    assert receipts[1]['raw_step_indices']==list(range(10,20))
    from rlt_online_rl.replay import ReplayManager
    journal=tmp_path/'journal.pkl'
    manager=ReplayManager(8,journal_path=str(journal));manager.add_transitions(transitions)
    batch=manager.sample_batch(2)
    assert batch['action_chunk'].shape==(2,10,7) and np.isfinite(batch['action_chunk']).all()
    restored=ReplayManager(8,journal_path=str(journal))
    assert restored.sample_batch(2)['action_chunk'].dtype==np.float32
    if snapshots:
        from methods.openpi_rlt.cobot_adapter.input_snapshots import load_and_verify
        payload,_=load_and_verify(receipts[0]['lossless_snapshot']['path'])
        np.testing.assert_array_equal(payload['current_input']['images']['base_0_rgb'],raw.observations[0]['images']['base_0_rgb'])
        assert receipts[1]['lossless_snapshot']['reason']=='snapshot_count_limit'
    wrapped.finalize('success' if success else 'failure')
    assert len(list((tmp_path/'trace').glob('*_success.jsonl'if success else'*_failure.jsonl')))==1


def test_real_wrapper_strict_mismatch_keeps_receipt_and_prevents_episode_submission(tmp_path,monkeypatch):
    d,raw,_,wrapped,_=episode(tmp_path,monkeypatch,1,True,'strict',0)
    get=d._feature_payload_for_observation
    def wrong(raw, index, cache, stats):
        value=dict(get(raw,index,cache,stats));value['proprio']=np.full(7,99,np.float32);return value
    d._feature_payload_for_observation=wrong
    with pytest.raises(ValueError,match='Replay input audit failed'):d._build_episode_replay(raw)
    receipt=json.loads(next((tmp_path/'trace/replay_inputs').glob('*.jsonl')).read_text())
    assert not receipt['checks_passed'] and not (tmp_path/'journal.pkl').exists()


def test_evaluation_gate_cannot_write_audit_into_previous_collection_root(tmp_path,monkeypatch):
    d,raw,_,wrapped,lifecycle=episode(tmp_path,monkeypatch,1,True,'strict',0)
    lifecycle.collecting=False
    transitions,stats=d._build_episode_replay(raw)
    assert transitions==[] and stats['replay_skipped_reason']=='cobot_episode_not_eligible'
    assert not (tmp_path/'trace/replay_inputs').exists()
    with pytest.raises(RuntimeError,match='collection episode'):root=wrapped._root
