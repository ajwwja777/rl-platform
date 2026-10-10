"""Target-only intervention censoring for explicitly isolated experiments.

This changes the return definition; it does not relabel human assistance as failure.
Only validated complete sparse terminal-reward Episodes are accepted. Human-start
windows keep their observed return. Autonomous starts stop before the next human
segment, including mixed windows crossing that boundary. Stored Replay is untouched.
"""
from collections import defaultdict
import numpy as np

def takeover_masks(raw):
    from .target_attribution import reconstruct_observed_returns
    observed, _ = reconstruct_observed_returns(raw, .99)
    if not np.isfinite(observed).all():
        raise ValueError("Incomplete or inconsistent sparse-reward Episode")
    groups=defaultdict(list)
    for i,(phase,ep) in enumerate(zip(raw['collection_phase_id'],raw['episode_id'])):
        groups[(int(phase),int(ep))].append(i)
    future=np.zeros(len(observed),bool);cut=future.copy()
    for key,ids in groups.items():
        timeline={}
        for i in ids:
            start=int(raw['step_id'][i])
            for off,source in enumerate(raw['source_chunk'][i]):
                if int(source) not in (0,1,2,3):raise ValueError("Unknown control source")
                t=start+off;human=int(source) in (2,3)
                if t in timeline and timeline[t]!=human:raise ValueError("Conflicting overlapping control identity")
                timeline[t]=human
        if min(timeline)!=0 or len(timeline)!=max(timeline)+1:raise ValueError("Incomplete control timeline")
        for i in ids:
            start=int(raw['step_id'][i]);horizon=len(raw['source_chunk'][i])
            if timeline[start] or key[1]<0:continue
            nxt=next((t for t in range(start+1,max(timeline)+1) if timeline[t]),None)
            if nxt is not None:
                future[i]=True;cut[i]=nxt<=start+horizon
                if cut[i] and np.any(raw['rewards'][i]):
                    # Reward belongs to human continuation; excluded by this objective.
                    pass
    return dict(future_takeover=future,bootstrap_cut=cut)

def replace_quota(selected, pool, count, rng):
    """Replace exactly count random slots; empty eligible pool preserves baseline."""
    result=np.asarray(selected,dtype=np.int64).copy();pool=np.asarray(pool,dtype=np.int64)
    if count<0 or count>len(result):raise ValueError("Invalid quota")
    if not count or not len(pool):return result
    positions=rng.choice(len(result),count,replace=False)
    result[positions]=rng.choice(pool,count,replace=True)
    return result
