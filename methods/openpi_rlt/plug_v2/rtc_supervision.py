"""Causal reference-queue augmentation for BC only; never factual TD replay.
Images/state are recorded teacher observations. Committed prefixes come only
from earlier reference predictions, never from future demonstration actions.
"""
import numpy as np
from .rtc_queue import RTCQueue,CommandFilter

def build_segment(state,actions,images,start,end,rpc):
    if end-start<7:return []
    queue=RTCQueue();queue.resume();filter=CommandFilter();rows=[]
    def infer(t):
        req=queue.request(state[t]);prefix=np.zeros((50,14),np.float32)
        prefix[:req.prefix_length]=req.prefix[:req.prefix_length]
        result=rpc.infer({'state':state[t].copy(),'images':images[t],
                         'action_prefix':prefix,'prefix_length':req.prefix_length})
        if int(result['prefix_length'])!=req.prefix_length:raise ValueError('RPC prefix contract')
        raw=np.asarray(result['ref_chunk'],np.float32)
        if raw.shape!=(50,14):raise ValueError('full reference horizon required')
        filtered=filter.plan(raw,state[t],prefix,req.prefix_length)
        if not queue.complete(req,filtered):raise ValueError('causal queue rejected plan')
        return req,result,raw
    # Runtime cold-primes the queue and immediately replans with d6 at tick0.
    infer(start)
    for t in range(start,end-6,10):
        req,result,raw=infer(t);d=req.prefix_length
        if d!=6:raise ValueError('causal deployment context must be d6')
        count=min(10,end-(t+d));mask=np.arange(10)<count
        c=np.r_[state[t],(req.prefix[:6]-state[t]).reshape(-1),d/6].astype(np.float32)
        if c.shape!=(99,):raise ValueError('RTC context must contain six committed commands only')
        target=actions[np.minimum(t+d+np.arange(10),end-1)].copy()
        rows.append({'z':np.asarray(result['z_rl'],np.float32),'c':c,
                     'ref':raw[d:d+10].copy(),'target':target,'mask':mask,
                     'frame':t,'prefix_origin_tick':max(0,req.start_tick-10),
                     'prefix_max_state_delta':float(np.max(abs(req.prefix[:6,7:13]-state[t,7:13]))),
                     'factual_RL_transition':False})
        # Consume the same ten queue ticks as deployment; state is a recorded
        # observation, not a simulated consequence of these synthetic commands.
        if t+10<end-6:
            for _ in range(10):queue.pop()
    return rows
