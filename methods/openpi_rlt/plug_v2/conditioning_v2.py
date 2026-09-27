# Matched differentiable fixed-candidate conditioning; offline candidate only.
import numpy as np
import torch
from .rtc_queue import PIPER_LOWER, PIPER_UPPER

def conditioned(raw, context):
    if raw.ndim != 3 or raw.shape[1:] != (10,14) or context.shape != (len(raw),99):
        raise ValueError('invalid conditioning shape')
    state=context[:,:14]
    prefix=context[:,14:98].reshape(-1,6,14)+state[:,None,:]
    d=context[:,-1]
    if not bool(torch.all((d==0)|(d==1))):
        raise ValueError('conditioning requires d0 or d6')
    active=d.bool()
    anchor=torch.where(active[:,None],prefix[:,-1],state)
    prev=torch.where(active[:,None],prefix[:,-1,7:13]-prefix[:,-2,7:13],torch.zeros_like(state[:,7:13]))
    low=torch.as_tensor(PIPER_LOWER.copy(),device=raw.device,dtype=raw.dtype)
    high=torch.as_tensor(PIPER_UPPER.copy(),device=raw.device,dtype=raw.dtype)
    alpha=1-np.exp(-1/(30*(-.05/np.log(.65))))
    outputs=[]
    for t in range(10):
        target=raw[:,t,7:13].clamp(min=low,max=high)
        delta=(alpha*(target-anchor[:,7:13])).clamp(-.1/30,.1/30)
        delta=torch.minimum(torch.maximum(delta,prev-.9/900),prev+.9/900)
        joints=(anchor[:,7:13]+delta).clamp(min=low,max=high)
        command=torch.cat((state[:,:7],joints,state[:,13:14]),dim=1)
        prev=joints-anchor[:,7:13]
        anchor=command
        outputs.append(command)
    return torch.stack(outputs,dim=1)
