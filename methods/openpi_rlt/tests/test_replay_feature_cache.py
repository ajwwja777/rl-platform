import copy
from types import SimpleNamespace
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.replay_feature_cache import ExactFeatureCache, build, enabled
from methods.openpi_rlt.tests.test_raw_observation_contract import observation, native_driver, raw_trace, append


def test_exact_duplicates_reuse_and_outputs_do_not_alias():
    calls=[]
    def infer(obs):
        calls.append(copy.deepcopy(obs));return {'ref_chunk':obs['state'].copy()}
    c=ExactFeatureCache(SimpleNamespace(get_features=infer));obs=observation(1)
    a=c.get_features(obs);a['ref_chunk'][0]=99
    b=c.get_features(copy.deepcopy(obs));assert b['ref_chunk'][0]==1
    assert len(calls)==1 and c.hits==1


@pytest.mark.parametrize('field',['state','image','prompt','rtc','extra','dtype','shape'])
def test_any_conditional_input_change_is_a_miss(field):
    c=ExactFeatureCache(SimpleNamespace(get_features=lambda o:{'x':1}));a=observation(1);b=copy.deepcopy(a)
    if field=='state': b['state'][0]+=1e-5
    elif field=='image': b['images']['base_0_rgb'][0,0,0]+=1
    elif field=='prompt': b['prompt']='changed'
    elif field=='rtc': b['rtc']={'delay_steps':1,'previous_actions':np.zeros((2,7),np.float32)}
    elif field=='extra': b['future_condition']=True
    elif field=='dtype': b['state']=b['state'].astype(np.float64)
    elif field=='shape': b['state']=b['state'][None]
    c.get_features(a);c.get_features(b);assert c.requests==2 and c.hits==0


def test_real_native_replay_arrays_identical_strict_audit_and_fewer_calls(tmp_path,monkeypatch):
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT','trace')
    monkeypatch.setenv('COBOT_RLT_INPUT_AUDIT','strict')
    d=native_driver(tmp_path);raw=raw_trace(observation(0));idx=0
    for i in range(3):
        idx=append(d,raw,observation(i),observation(i+1),observation_idx=idx,chunk_id=i)
        raw.steps[-1].done=i==2;raw.chunks[-1].start_z_rl=None;raw.chunks[-1].start_proprio=None;raw.chunks[-1].start_ref_chunk=None
    old_provider=d._feature_provider;calls=[]
    def infer(obs):calls.append(1);return old_provider.get_features(obs)
    provider=SimpleNamespace(get_features=infer);d._feature_provider=provider
    monkeypatch.delenv('COBOT_DEPLOYMENT_MODEL_ID',raising=False)
    baseline,_=d._build_episode_replay(raw);baseline_calls=len(calls);calls.clear()
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID','plug-v3-hil8-explore')
    actual,stats=d._build_episode_replay(raw)
    assert baseline_calls==6 and len(calls)==4 and stats['exact_cache_hits']==2
    assert len(actual)==len(baseline)==3
    for a,b in zip(actual,baseline):
        for k,v in a.to_numpy().items():np.testing.assert_array_equal(v,b.to_numpy()[k],err_msg=k)
    assert d._feature_provider is provider
    calls.clear();d._build_episode_replay(raw);assert len(calls)==4 # never cross Episodes/builds


def test_provider_restored_on_failure_and_optout(monkeypatch):
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID','plug-v3-hil8-explore')
    monkeypatch.setenv('COBOT_RLT_RAW_OBSERVATION_CONTRACT','trace')
    provider=object();d=SimpleNamespace(_env=SimpleNamespace(cobot_task2_contract=True),_env_config=SimpleNamespace(step_trace_stride=0),_feature_provider=provider,_feature_payload_for_observation=lambda *a: None)
    old_lookup=d._feature_payload_for_observation
    def fail(*args):raise RuntimeError('strict audit failure')
    with pytest.raises(RuntimeError,match='strict audit failure'):build(d,SimpleNamespace(episode_id=1),fail)
    assert d._feature_provider is provider and d._feature_payload_for_observation is old_lookup
    monkeypatch.setenv('COBOT_RLT_REPLAY_EXACT_CACHE','0');assert not enabled(d)
    monkeypatch.setenv('COBOT_RLT_REPLAY_EXACT_CACHE','1');d._env_config.step_trace_stride=2;assert not enabled(d)
    d._env_config.step_trace_stride=0;monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID','other');assert not enabled(d)


@pytest.mark.parametrize('source',[1,2])
@pytest.mark.parametrize('success',[False,True])
def test_c10_real_collection_wrapper_duplicate_indices_preserve_all_arrays(tmp_path,monkeypatch,source,success):
    from methods.openpi_rlt.tests.test_collection_input_audit import episode
    d,raw,_,wrapped,_=episode(tmp_path,monkeypatch,source,success,'strict',0)
    old_index=raw.steps[10].observation_idx
    raw.steps[10].observation_idx=len(raw.observations)
    raw.observations.append(copy.deepcopy(raw.observations[old_index]))
    original=d._feature_provider;calls=[]
    def infer(o):calls.append(1);return original.get_features(o)
    d._feature_provider=SimpleNamespace(get_features=infer)
    monkeypatch.delenv('COBOT_DEPLOYMENT_MODEL_ID',raising=False)
    before,_=d._build_episode_replay(raw);assert len(calls)==4;calls.clear()
    monkeypatch.setenv('COBOT_DEPLOYMENT_MODEL_ID','plug-v3-hil8-explore')
    after,stats=d._build_episode_replay(raw);assert len(calls)==3 and stats['exact_cache_hits']==1
    assert len(before)==len(after)==2
    for a,b in zip(before,after):
        for k,v in a.to_numpy().items():np.testing.assert_array_equal(v,b.to_numpy()[k],err_msg=k)
    wrapped.finalize('success' if success else 'failure')
