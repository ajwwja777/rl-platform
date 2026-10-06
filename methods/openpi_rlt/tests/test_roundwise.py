import copy
import numpy as np
import pytest
from methods.openpi_rlt.experiments.roundwise import (annotate, episode_key, heldout,
    RoundSampler, SamplingProfile, derive_batch)


def episode(e, sources=None, success=True, phase="online", uid=None):
    sources = sources or [[1]*10, [1]*10, [1]*10]
    rows=[]
    for j, source in enumerate(sources):
        rewards=np.zeros(10,np.float32)
        if j==len(sources)-1 and success: rewards[-1]=1.
        rows.append(dict(episode_id=e,step_id=j*10,source_chunk=np.array(source),
            collection_phase=phase,done=j==len(sources)-1,success=int(success),rewards=rewards,
            **({"episode_uuid":uid} if uid else {})))
    return rows


def test_logical_mc_accounts_for_reward_slot_and_not_window_rank():
    rows=episode(1); meta=annotate(rows,heldout_fraction=0)
    assert meta[0]["mc_return"]==pytest.approx(.99**29)
    assert meta[-1]["mc_return"]==pytest.approx(.99**9)
    assert meta[-1]["mc_return"] != 1.
    assert all(m["outcome"]=="auto_success" for m in meta)


def test_hil_cut_is_separate_from_original_success_and_post_hil_is_not_autonomous():
    rows=episode(2,[[1]*10,[2]*10,[1]*10]); original=copy.deepcopy(rows)
    meta=annotate(rows,heldout_fraction=0)
    assert meta[0]["autonomy_cut"] and meta[0]["mc_return"]==0
    assert meta[1]["pool"]=="human_correction"
    assert meta[2]["pool"]=="failure" and meta[2]["outcome"]=="assisted_success"
    assert all(a["success"]==b["success"] for a,b in zip(rows,original))
    assert not rows[0]["done"]
    raw={k:np.stack([r[k] for r in rows]) for k in ["done","rewards","source_chunk"]}
    derived=derive_batch(raw,meta,np.array([0,1,2]))
    assert derived["done"][0] and not raw["done"][0]
    assert not derived["autonomous_success"].any()


@pytest.mark.parametrize("corrupt",["no_terminal","duplicate","gap","reward","prefix","unknown_source"])
def test_ambiguous_or_incomplete_episode_excluded(corrupt):
    rows=episode(1)
    if corrupt=="no_terminal": rows[-1]["done"]=False
    if corrupt=="duplicate": rows.append(copy.deepcopy(rows[0]))
    if corrupt=="gap": rows[1]["step_id"]=40
    if corrupt=="reward": rows[0]["rewards"][2]=1
    if corrupt=="prefix": rows=rows[1:]
    if corrupt=="unknown_source": rows[0]["source_chunk"][0]=9
    assert not any(m["eligible"] for m in annotate(rows))


def test_uuid_and_phase_prevent_cross_session_numeric_id_leakage():
    rows=episode(1,uid="a")+episode(1,uid="b")+episode(1,phase="warmup")
    meta=annotate(rows)
    assert len({m["episode_key"] for m in meta})==3
    assert len({episode_key(r) for r in rows})==3
    for key in {m["episode_key"] for m in meta}:
        assert len({m["split"] for m in meta if m["episode_key"]==key})==1


def test_tail_proxy_uses_steps_and_mixed_windows_are_excluded():
    rows=episode(1,[[1]*10,[1]*5+[2]*5,[2]*10])
    meta=annotate(rows,heldout_fraction=0,tail_steps=10,takeover_before=0,takeover_after=0)
    assert not meta[0]["eligible"] and meta[0]["reason"]=="outside_tail_time_proxy"
    assert not meta[1]["eligible"] and meta[1]["reason"]=="mixed_source_window"
    assert meta[2]["eligible"]


def test_balanced_sampler_exact_size_disjoint_pools_and_no_dev():
    rows=episode(-1,[[2]*10]*3)+episode(2,[[2]*10]*3)+episode(3)+episode(4,success=False)
    meta=annotate(rows,heldout_fraction=0)
    for m in meta: m["round_id"]="old"
    for m in meta[6:9]: m["round_id"]="new"
    sampler=RoundSampler(meta,profile=SamplingProfile(recent_round="new"))
    ids=sampler.sample(np.random.default_rng(5),128)
    assert len(ids)==128
    receipt=sampler.receipt()
    assert sum(receipt["pool_draws"].values())==128
    assert receipt["pool_draws"]=={"expert":29,"human_correction":28,"auto_success":28,"failure":43}
    assert receipt["recent_ratio"]==pytest.approx(28/128)
    assert all(meta[i]["split"]=="train" for i in ids)
    meta[0]["split"]="development"
    sampler=RoundSampler(meta)
    assert 0 not in sampler.sample(np.random.default_rng(5),1000)


def test_episode_first_sampling_avoids_long_episode_overweight():
    # One failure Episode has3 windows, the other30; their probability is equal.
    rows=episode(1,success=False)+episode(2,[[1]*10]*30,success=False)
    meta=annotate(rows,heldout_fraction=0,tail_steps=1000)
    sampler=RoundSampler(meta)
    ids=sampler.sample(np.random.default_rng(0),10000)
    short=sum(i<3 for i in ids)
    assert 4800<short<5200


def test_empty_pool_redistribution_keeps_batch_full():
    sampler=RoundSampler(annotate(episode(1),heldout_fraction=0))
    assert len(sampler.sample(np.random.default_rng(0),128))==128
    assert sampler.receipt()["pool_draws"]["auto_success"]==128


def test_explicit_recency_and_deterministic_holdout():
    key="dataset:online:uuid:known"
    assert heldout(key)==heldout(key)
    sampler=RoundSampler(annotate(episode(1),heldout_fraction=0))
    sampler.sample(np.random.default_rng(0),16)
    assert not sampler.receipt()["recent_identity_available"]
    assert SamplingProfile().recent_round is None


def test_archive_expert_convention_is_explicit_and_not_applied_to_online():
    rows=episode(100007,[[2]*10]*3,phase="warmup")
    native=annotate(rows,heldout_fraction=0)
    archive=annotate(rows,heldout_fraction=0,legacy_expert_id_base=100000)
    assert native[0]["pool"]=="human_correction"
    assert archive[0]["pool"]=="expert" and archive[0]["outcome"]=="expert_success"
    online=annotate(episode(100007,[[2]*10]*3),heldout_fraction=0,legacy_expert_id_base=100000)
    assert online[0]["pool"]=="human_correction"
