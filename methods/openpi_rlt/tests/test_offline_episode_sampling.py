from pathlib import Path
import importlib.util
import numpy as np

spec=importlib.util.spec_from_file_location("episode_sampling",Path(__file__).resolve().parents[3]/"scripts/offline_episode_sampling.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_non_recent_slots_and_available_support_are_identical():
    ids=np.zeros(200,int);ids[10:100]=1
    out=m.paired_recent_indices(np.arange(100),np.arange(100,150),np.arange(150,180),np.arange(200),ids,np.random.default_rng(42))
    a,b=out.values();assert len(a)==len(b)==128
    # Some uniform slots may also hit recent; each non-recent draw remains paired.
    np.testing.assert_array_equal(a[a>=100],b[b>=100])
    assert (a>=0).all()and(a<200).all()and(b>=0).all()and(b<200).all()

def test_single_episode_has_exact_same_draws():
    out=m.paired_recent_indices(np.arange(20),np.arange(20,30),np.arange(30,40),np.arange(40),np.zeros(40,int),np.random.default_rng(41))
    np.testing.assert_array_equal(out['window_recent'],out['episode_recent'])

def test_empty_recent_fallback_has_exact_same_draws():
    out=m.paired_recent_indices([],[],[],np.arange(40),np.arange(40),np.random.default_rng(43))
    np.testing.assert_array_equal(out['window_recent'],out['episode_recent'])

def test_short_episode_is_not_weighted_by_trajectory_length():
    ids=np.ones(300,int);ids[0]=0;rng=np.random.default_rng(42);counts=np.zeros(2,int)
    # Other pools use only outside indices, so the short Episode can enter via recent only.
    for _ in range(200):
        out=m.paired_recent_indices(np.arange(100),np.arange(100,200),np.arange(200,300),np.arange(100,300),ids,rng)
        counts += [(out[k]==0).sum()for k in ['window_recent','episode_recent']]
    assert .005 < counts[0]/10200 < .02
    assert .47 < counts[1]/10200 < .53
