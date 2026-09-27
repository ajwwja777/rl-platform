import numpy as np
from methods.openpi_rlt.plug_v2.balanced_replay import select_balanced_indices

def storage():
    n=100
    human=np.zeros(n,dtype=bool); human[:30]=True
    source=np.where(human,2,1).astype(np.uint8)
    source_chunk=np.repeat(source[:,None],10,axis=1)
    success=np.zeros(n,dtype=np.int8)
    success[:15]=1
    success[30:65]=1
    phase=np.full(n,2,dtype=np.uint8)
    episode=np.arange(n,dtype=np.int32)
    return dict(intervention_flag=human,source=source,source_chunk=source_chunk,
                success=success,collection_phase_id=phase,episode_id=episode)

def test_human_ratio_is_exact_and_pools_do_not_overlap():
    s=storage(); idx=select_balanced_indices(s,100,50,np.random.default_rng(4),human_ratio=.2)
    assert len(idx)==50
    assert int(s['intervention_flag'][idx].sum())==10

def test_policy_outcomes_are_balanced_when_both_exist():
    s=storage(); idx=select_balanced_indices(s,100,50,np.random.default_rng(5),human_ratio=.2)
    policy=idx[~s['intervention_flag'][idx]]
    assert int(s['success'][policy].sum())==20
    assert int((s['success'][policy]==0).sum())==20

def test_missing_pool_falls_back_without_changing_batch_size():
    s=storage(); s['intervention_flag'][:]=False; s['source'][:]=1; s['source_chunk'][:]=1
    idx=select_balanced_indices(s,100,64,np.random.default_rng(6),human_ratio=.2)
    assert len(idx)==64
    assert np.all((idx>=0)&(idx<100))


def test_extra_actor_metadata_round_trips_through_sampling():
    from methods.openpi_rlt.plug_v2.balanced_replay import BalancedReplayBuffer
    from rlt_online_rl.replay import RLTTransition
    b=BalancedReplayBuffer(capacity=2,seed=1,sample_strategy='stratified')
    zero=lambda shape:np.zeros(shape,np.float32)
    r=RLTTransition(z_rl=zero(3),proprio=zero(2),ref_chunk=zero((10,7)),action_chunk=zero((10,7)),
      rewards=zero(16),done=True,next_z_rl=zero(3),next_proprio=zero(2),next_ref_chunk=zero((10,7)),
      source=1,source_chunk=np.ones(10,np.uint8),collection_phase='online',success=1,
      intervention_flag=False,episode_id=0,step_id=0)
    b.add_with_metadata(r,original_done=True,episode_position=.75)
    sample=b.sample(1)
    assert bool(sample['original_done'][0]) is True
    assert float(sample['episode_position'][0])==.75
