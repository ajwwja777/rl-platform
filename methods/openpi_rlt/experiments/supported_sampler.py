"""Journal-backed reproducible sampler for explicitly selected HIL8 candidate.

Own both base and Actor-replacement RNG states. Field labels and journal are
immutable inputs; no RPC/robot writes. Recent windows still use the pinned20-ID
rule, and only real Online HIL >= candidate ID floor receives extra Actor slots.
"""
from pathlib import Path
import pickle
import json
import hashlib
import numpy as np

FIELDS=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl',
        'next_proprio','next_ref_chunk','source_chunk','collection_phase_id',
        'episode_id','step_id','success','source','intervention_flag']

class SupportedSampler:
    def __init__(self,index,adapter,*,state,actor_hil_slots=8,floor=10000):
        if actor_hil_slots!=8:raise ValueError("Only validated HIL8 profile accepted")
        self.index=index;self.adapter=adapter;self.slots=actor_hil_slots;self.floor=floor
        self.base=np.random.default_rng();self.extra=np.random.default_rng()
        self.base.bit_generator.state=state['base_rng'];self.extra.bit_generator.state=state['actor_rng']
        self.step=int(state['global_step']);self.signature=None;self.arrays=None;self.last_receipt=None
    def state(self):
        return dict(base_rng=self.base.bit_generator.state,actor_rng=self.extra.bit_generator.state,global_step=self.step)
    def refresh(self):
        self.index.refresh()
        if self.signature==self.index.signature:return
        rows=[]
        with self.index.path.open('rb')as f:
            while f.tell()<self.index.signature[2]:
                try:row=pickle.load(f)
                except (EOFError,pickle.UnpicklingError):break
                if f.tell()>self.index.signature[2]:break
                key=tuple(int(row[k])for k in ['collection_phase_id','episode_id','step_id'])
                if self.index.lookup.get(key,(0,False,False))[1]:rows.append(row)
        st=self.index.path.stat();signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if signature!=self.index.signature:raise RuntimeError("Journal changed during sampler refresh; retry after complete admission")
        if not rows:raise ValueError("No complete eligible Episode")
        self.arrays={k:np.stack([row[k]for row in rows])for k in FIELDS}
        self.signature=signature
    def sample_batch(self,size):
        if size!=128:raise ValueError("Validated batch128 required")
        self.refresh();a=self.arrays;all_pool=np.arange(len(a['episode_id']))
        ep=a['episode_id'];phase=a['collection_phase_id'];recent=all_pool[(phase==2)&(ep>=int(ep.max())-19)]
        warmup=all_pool[phase==1];human=np.isin(a['source_chunk'],[2,3]).any(axis=1)
        hp=all_pool[a['intervention_flag']|np.isin(a['source'],[2,3])|human]
        pool=recent if len(recent)else all_pool
        # Matches the preregistered paired study, including all RNG consumption.
        first=pool[(self.base.random(51)*len(pool)).astype(np.int64)]
        other=np.concatenate([self.base.choice(p if len(p)else all_pool,n,replace=True)
            for p,n in [(warmup,38),(hp,26),(all_pool,13)]])
        ci=np.concatenate([first,other])[self.base.permutation(128)];ai=ci.copy()
        eligible=all_pool[(phase==2)&(ep>=int(ep.max())-19)&(ep>=self.floor)&human]
        if len(eligible):
            positions=self.extra.choice(128,self.slots,replace=False)
            ai[positions]=self.extra.choice(eligible,self.slots,replace=True)
        cb=self.index.attach({k:v[ci]for k,v in a.items()});ab=self.index.attach({k:v[ai]for k,v in a.items()})
        ab=self.adapter.prepare_training_batch(ab)
        packed=dict(cb,**{'actor__'+k:v for k,v in ab.items()})
        self.step+=1
        self.last_receipt=dict(global_step=self.step,actor_update=self.step%2==0,
            critic_identities=[[int(a[k][i])for k in ['collection_phase_id','episode_id','step_id']]for i in ci],
            actor_identities=[[int(a[k][i])for k in ['collection_phase_id','episode_id','step_id']]for i in ai],
            available_complete_windows=len(all_pool),recent_hil_pool_windows=len(eligible),extra_actor_slots=8 if len(eligible)else 0,
            train_only=True,unit='draws, not independent samples')
        return packed

def unpack_actor_batch(batch):
    actor={k[len('actor__'):]:v for k,v in batch.items()if k.startswith('actor__')}
    if not actor:raise ValueError("Missing Actor sampling contract")
    critic={k:v for k,v in batch.items()if not k.startswith('actor__')}
    return dict(critic,actor_batch=actor)
