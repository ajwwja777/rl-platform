"""Optional deployment-only low-frequency correction projection.
Least-squares constant/linear fit of actor-reference correction across one C10
chunk. This is not an RLT paper component or a learner/network change.
"""
import numpy as np

def project_correction(actor,reference,degree=1):
    a=np.asarray(actor,np.float32);r=np.asarray(reference,np.float32)
    if a.shape!=r.shape or a.shape[-2:]!=(10,7) or not np.isfinite(a).all() or not np.isfinite(r).all():raise ValueError('expected finite (...,10,7) chunks')
    if degree not in (0,1):raise ValueError('projection degree must be 0 or 1')
    delta=a[...,:6]-r[...,:6];mean=delta.mean(axis=-2,keepdims=True)
    fitted=np.broadcast_to(mean,delta.shape).copy()
    if degree:
        t=np.arange(10,dtype=np.float32)-4.5
        slope=(delta*t[:,None]).sum(axis=-2,keepdims=True)/(t*t).sum()
        fitted+=slope*t[:,None]
    result=r.copy();result[...,:6]+=fitted
    return result
