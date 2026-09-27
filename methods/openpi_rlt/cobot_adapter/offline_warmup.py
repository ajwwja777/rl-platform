"""Offline-only episode splitting, sampling and expert chunk construction."""
from collections import defaultdict
import numpy as np

GROUPS = ("expert", "success", "failure")

def split_online_episodes(records, *, seed=42):
    by_group = defaultdict(set)
    for r in records:
        by_group[r["group"]].add(int(r["episode_id"]))
    rng=np.random.default_rng(seed)
    held=set()
    for group in sorted(by_group):
        ids=sorted(by_group[group])
        if len(ids)<2: raise ValueError("each stratum needs at least two episodes")
        rng.shuffle(ids)
        held.update(ids[:max(1,round(len(ids)*.2))])
    return ([r for r in records if int(r["episode_id"]) not in held],
            [r for r in records if int(r["episode_id"]) in held])

class EpisodeStratifiedSampler:
    """Choose stratum, then uniform episode, then uniform window."""
    def __init__(self, records, *, seed=42):
        self.records=list(records)
        self.rng=np.random.default_rng(seed)
        self.pools={g:defaultdict(list) for g in GROUPS}
        for i,r in enumerate(records): self.pools[r["group"]][int(r["episode_id"])].append(i)
        if any(not p for p in self.pools.values()): raise ValueError("empty replay stratum")
    def sample_indices(self,n):
        counts=[round(n*.3),round(n*.4)]
        counts.append(n-sum(counts))
        result=[]
        for g,count in zip(GROUPS,counts):
            pool=self.pools[g];ids=sorted(pool)
            for ep in self.rng.choice(ids,size=count):result.append(self.rng.choice(pool[int(ep)]))
        self.rng.shuffle(result)
        return np.asarray(result,dtype=np.int64)

def chunk_starts(n,chunk_len=10):
    if n<chunk_len: raise ValueError("episode shorter than one full chunk")
    return sorted(set(range(0,n-chunk_len+1,chunk_len))|{n-chunk_len})

def feature_indices(n,chunk_len=10):
    return sorted(set(chunk_starts(n,chunk_len))|{min(i+chunk_len,n-1) for i in chunk_starts(n,chunk_len)})

def expert_transitions(states,actions,features,*,episode_id,chunk_len=10):
    states=np.asarray(states,np.float32);actions=np.asarray(actions,np.float32)
    if states.shape!=actions.shape or states.ndim!=2 or states.shape[1]!=14:
        raise ValueError("expert state/action must both be [N,14]")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("nonfinite expert")
    out=[]
    for start in chunk_starts(len(states),chunk_len):
        end=start+chunk_len;terminal=end==len(states)
        cur=features[start];nxt=features[min(end,len(states)-1)]
        for f in [cur,nxt]:
            if np.asarray(f["z_rl"]).shape!=(2048,) or np.asarray(f["ref_chunk"]).shape!=(chunk_len,14):
                raise ValueError("feature shape mismatch")
            if not all(np.isfinite(f[k]).all() for k in ["z_rl","ref_chunk"]):raise ValueError("nonfinite feature")
        rewards=np.zeros(chunk_len,np.float32)
        if terminal: rewards[-1]=1
        out.append(dict(z_rl=cur["z_rl"],proprio=states[start],ref_chunk=cur["ref_chunk"],
            action_chunk=actions[start:end],rewards=rewards,done=terminal,
            next_z_rl=nxt["z_rl"],next_proprio=states[min(end,len(states)-1)],next_ref_chunk=nxt["ref_chunk"],
            source=2,source_chunk=np.full(chunk_len,2,np.uint8),collection_phase="warmup",success=int(terminal),
            intervention_flag=True,episode_id=episode_id,step_id=start,group="expert"))
    return out

def action_metrics(pred,records):
    pred=np.asarray(pred,np.float64)
    ref=np.stack([r["ref_chunk"] for r in records]).astype(np.float64)
    actual=np.stack([r["action_chunk"] for r in records]).astype(np.float64)
    src=np.stack([r["source_chunk"] for r in records])
    target=np.where(np.isin(src,[2,3])[...,None],actual,ref)
    state=np.stack([r["proprio"] for r in records])
    if pred.shape!=target.shape or not np.isfinite(pred).all():raise ValueError("invalid actor prediction")
    right=pred[:,:,7:13]
    lookup={(int(r["episode_id"]),int(r["step_id"])):i for i,r in enumerate(records)}
    jumps=[]
    for i,r in enumerate(records):
        j=lookup.get((int(r["episode_id"]),int(r["step_id"])+10))
        if j is not None:jumps.extend(np.abs(pred[j,0,7:13]-pred[i,-1,7:13]).tolist())
    def rms(v):return float(np.sqrt(np.mean(np.square(v))))
    return dict(right_rmse_rad=rms(right-target[:,:,7:13]),
        ref_correction_rms_rad=rms(right-ref[:,:,7:13]),
        first_delta_p95_rad=float(np.percentile(np.abs(right[:,0]-state[:,7:13]),95)),
        velocity_p95_rad_per_sample=float(np.percentile(np.abs(np.diff(right,axis=1)),95)),
        acceleration_p95_rad_per_sample2=float(np.percentile(np.abs(np.diff(right,n=2,axis=1)),95)),
        boundary_jump_p95_rad=float(np.percentile(jumps,95)) if jumps else None,
        boundary_joint_values=len(jumps),
        gripper_rmse=rms(pred[:,:,[6,13]]-target[:,:,[6,13]]))
