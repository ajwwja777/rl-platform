import importlib.util
import numpy as np
import pytest
MODULE="methods.openpi_rlt.cobot_adapter.offline_warmup"

def api():
    assert importlib.util.find_spec(MODULE) is not None, "offline warmup data module is missing"
    return __import__(MODULE,fromlist=["x"])

def records():
    out=[]
    for group in ["expert","success","failure"]:
        for ep in range(6):
            for step in range(2+ep):
                out.append(dict(group=group,episode_id=100*["expert","success","failure"].index(group)+ep,
                    step_id=step,success=int(group!="failure" and step==1+ep),done=step==1+ep,
                    rewards=np.array([int(group!="failure" and step==1+ep)],np.float32),
                    action_chunk=np.zeros((10,14)),source_chunk=np.zeros(10,np.uint8)))
    return out

def test_split_keeps_whole_episodes_and_outcome_from_terminal():
    a=api();data=records();train,val=a.split_online_episodes(data,seed=42)
    assert not {x["episode_id"] for x in train}&{x["episode_id"] for x in val}
    assert set(x["group"] for x in train)=={"expert","success","failure"}
    assert len({x["episode_id"] for x in val})==3
    for group in ["expert","success","failure"]:
        assert any(x["group"]==group for x in val)

def test_sampler_exact_counts_and_no_length_bias():
    a=api();s=a.EpisodeStratifiedSampler(records(),seed=42)
    totals={g:0 for g in ["expert","success","failure"]}
    eps={}
    for _ in range(200):
        ids=s.sample_indices(128)
        assert len(ids)==128
        groups=[s.records[i]["group"] for i in ids]
        assert [groups.count(g) for g in totals]==[38,51,39]
        for i in ids:
            e=s.records[i]["episode_id"];eps[e]=eps.get(e,0)+1
    for group in range(3):
        counts=[eps[100*group+i] for i in range(6)]
        assert max(counts)/min(counts)<1.3

def test_expert_chunk_full_length_single_terminal_and_real_next_state():
    a=api();n=24
    states=np.ones((n,14),np.float32)
    actions=states+np.arange(n)[:,None]*.001
    features={i:dict(z_rl=np.ones(2048)*i,ref_chunk=np.ones((10,14)),proprio=states[i]) for i in a.feature_indices(n,10)}
    rec=a.expert_transitions(states,actions,features,episode_id=1001)
    assert sum(bool(x["done"]) for x in rec)==1
    assert sum(float(np.sum(x["rewards"])) for x in rec)==1
    assert all(x["action_chunk"].shape==(10,14) for x in rec)
    assert np.array_equal(rec[-1]["next_z_rl"],features[n-1]["z_rl"])
    assert np.all(rec[-1]["source_chunk"]==2)
    assert rec[-1]["step_id"]==14

def test_expert_rejects_missing_features_or_nan():
    a=api()
    with pytest.raises(ValueError):a.feature_indices(8,10)
    with pytest.raises((KeyError,ValueError)):
        a.expert_transitions(np.ones((12,14)),np.ones((12,14)),{},episode_id=1)

def test_empty_stratum_rejected():
    a=api()
    with pytest.raises(ValueError):
        a.EpisodeStratifiedSampler([r for r in records() if r["group"]!="failure"])

def test_metrics_detect_first_step_and_chunk_boundary_jump():
    a=api()
    records=[dict(episode_id=1,step_id=i,proprio=np.ones(14),ref_chunk=np.ones((10,14)),action_chunk=np.ones((10,14)),source_chunk=np.full(10,2)) for i in [0,10]]
    pred=np.ones((2,10,14));pred[1,:,7:13]+=.1
    m=a.action_metrics(pred,records)
    assert m["boundary_jump_p95_rad"]==pytest.approx(.1)
    assert m["first_delta_p95_rad"]==pytest.approx(.1)
    assert m["right_rmse_rad"]>0
