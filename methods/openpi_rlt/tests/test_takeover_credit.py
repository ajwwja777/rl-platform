import numpy as np
import pytest
from methods.openpi_rlt.experiments.takeover_credit import takeover_masks, replace_quota

def episode(sources, ep=1):
    starts=np.arange(len(sources)-1);n=len(starts)
    rewards=np.zeros(len(sources));rewards[-1]=1.
    return dict(collection_phase_id=np.full(n,2),episode_id=np.full(n,ep),step_id=starts,
        source_chunk=np.array([sources[i:i+2]for i in starts]),rewards=np.array([rewards[i:i+2]for i in starts]),
        done=starts==starts[-1],success=(starts==starts[-1]).astype(int))

def test_multi_takeover_mixed_and_human_starts():
    r=episode([1,1,2,2,1,1,2,2]);before={k:v.copy()for k,v in r.items()}
    m=takeover_masks(r)
    assert m['future_takeover'].tolist()==[True,True,False,False,True,True,False]
    assert m['bootstrap_cut'].tolist()==[True,True,False,False,True,True,False]
    for k in r:np.testing.assert_array_equal(r[k],before[k])

def test_far_before_takeover_bootstraps_until_boundary():
    m=takeover_masks(episode([1,1,1,1,2,2]))
    assert m['future_takeover'].tolist()==[True,True,True,True,False]
    assert m['bootstrap_cut'].tolist()==[False,False,True,True,False]

def test_experts_and_no_intervention_not_censored():
    assert not takeover_masks(episode([2,2,2,2],-1))['future_takeover'].any()
    assert not takeover_masks(episode([1,1,1,1]))['future_takeover'].any()

@pytest.mark.parametrize('kind',['unknown','conflict','reward','prefix'])
def test_malformed_evidence_rejected(kind):
    r=episode([1,1,2,2])
    if kind=='unknown':r['source_chunk'][0,0]=9
    if kind=='conflict':r['source_chunk'][0,1]=2
    if kind=='reward':r['rewards'][0,0]=1
    if kind=='prefix':r={k:v[1:]for k,v in r.items()}
    with pytest.raises(ValueError):takeover_masks(r)

def test_quota_exact_and_empty_pool_identity():
    base=np.arange(128);r=replace_quota(base,[200,201],8,np.random.default_rng(1))
    assert np.isin(r,[200,201]).sum()==8
    np.testing.assert_array_equal(base,np.arange(128))
    np.testing.assert_array_equal(replace_quota(base,[],8,np.random.default_rng(1)),base)
    with pytest.raises(ValueError):replace_quota(base,[200],129,np.random.default_rng(1))
