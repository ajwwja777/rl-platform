"""Critic target protocols for complete factual episodes."""
from __future__ import annotations
import numpy as np

def actor_q_mask(original_done,source_chunk):
    done=np.asarray(original_done,dtype=bool)
    source=np.asarray(source_chunk)
    return done|np.any(np.isin(source,(2,3)),axis=1)

def training_target(values,*,success,mode):
    out=dict(values)
    out['rewards']=np.asarray(values['rewards'],dtype=np.float32).copy()
    if mode=='td':return out
    if mode!='mc_success':raise ValueError('unknown critic target '+str(mode))
    out['rewards'].fill(0.)
    out['rewards'][0]=1. if bool(success) else 0.
    out['done']=True
    out['duration']=1
    out['td_valid']=True
    return out
