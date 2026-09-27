"""Exclusive HIL and outcome-balanced replay sampling for plug_v2."""
from __future__ import annotations
import numpy as np
from rlt_online_rl.replay import ReplayBuffer

def _sample(rng, pool, count):
    pool=np.asarray(pool,dtype=np.int64)
    if count<=0 or pool.size==0:return np.empty((0,),dtype=np.int64)
    return rng.choice(pool,size=int(count),replace=True)

def select_balanced_indices(storage,size,batch_size,rng,*,human_ratio=.2):
    size=int(size); batch_size=int(batch_size)
    all_indices=np.arange(size,dtype=np.int64)
    intervention=np.asarray(storage['intervention_flag'][:size],dtype=bool)
    source=np.asarray(storage['source'][:size])
    source_chunk=np.asarray(storage['source_chunk'][:size])
    human=intervention|np.isin(source,(2,3))|np.any(np.isin(source_chunk,(2,3)),axis=1)
    human_pool=all_indices[human]; policy_pool=all_indices[~human]
    if human_pool.size and policy_pool.size:
        n_human=min(batch_size,max(0,int(round(batch_size*float(human_ratio)))))
    elif human_pool.size:
        n_human=batch_size
    else:
        n_human=0
    n_policy=batch_size-n_human
    success=np.asarray(storage['success'][:size]).astype(bool)
    policy_success=policy_pool[success[policy_pool]]
    policy_failure=policy_pool[~success[policy_pool]]
    if n_policy and policy_success.size and policy_failure.size:
        n_success=n_policy//2
        policy_parts=[_sample(rng,policy_success,n_success),_sample(rng,policy_failure,n_policy-n_success)]
    else:
        policy_parts=[_sample(rng,policy_pool,n_policy)]
    result=np.concatenate([_sample(rng,human_pool,n_human),*policy_parts])
    if result.size<batch_size:
        result=np.concatenate([result,_sample(rng,all_indices,batch_size-result.size)])
    rng.shuffle(result)
    return result[:batch_size]

class BalancedReplayBuffer(ReplayBuffer):
    """Replay buffer whose 20% HIL quota is exclusive and auditable."""

    def add_with_metadata(self,record,*,original_done,episode_position):
        index=self._position
        super().add(record)
        with self._lock:
            assert self._storage is not None
            if 'original_done' not in self._storage:
                self._storage['original_done']=np.zeros((self.capacity,),dtype=np.bool_)
                self._storage['episode_position']=np.zeros((self.capacity,),dtype=np.float32)
            self._storage['original_done'][index]=bool(original_done)
            self._storage['episode_position'][index]=float(episode_position)

    def _sample_stratified_indices(self,batch_size):
        assert self._storage is not None
        return select_balanced_indices(
            self._storage,self._size,batch_size,self._rng,
            human_ratio=self._human_intervention_ratio,
        )
