"""RLT-only opt-in two-pole command filter. No model/ROS or shared VLA mutation.

Tau is the equivalent one-pole LOW-FREQUENCY delay parameter. Each pole alpha
is chosen so the sum of discrete DC group delays equals that of the one-pole
filter at this tau/dt. This is not a hard acceleration/jerk bound.
"""
import math
import numpy as np

class CascadedJointFilter:
    def __init__(self, tau_sec=.12, max_velocity=.6, joint_dimensions=6):
        if not np.isfinite([tau_sec,max_velocity]).all() or tau_sec<=0 or max_velocity<=0:
            raise ValueError('Invalid cascade configuration')
        self.tau,self.velocity,self.joints=tau_sec,max_velocity,joint_dimensions
        self.previous=None
        self.stage=None

    def reset(self, measured=None):
        self.previous=None if measured is None else np.asarray(measured,np.float32).copy()
        self.stage=None if self.previous is None else self.previous.copy()

    def apply(self,target,measured,dt):
        target,measured=np.asarray(target,np.float32),np.asarray(measured,np.float32)
        if target.ndim!=1 or target.shape!=measured.shape or target.size<self.joints or not np.isfinite(dt) or dt<=0:
            raise ValueError('Invalid filter tick')
        if not np.isfinite(target).all() or not np.isfinite(measured).all():raise ValueError('Nonfinite command')
        if self.previous is None:self.reset(measured)
        a=-math.expm1(-dt/self.tau)
        pole_alpha=2*a/(1+a)
        self.stage[:self.joints]+=pole_alpha*(target[:self.joints]-self.stage[:self.joints])
        output=target.copy()
        delta=pole_alpha*(self.stage[:self.joints]-self.previous[:self.joints])
        output[:self.joints]=self.previous[:self.joints]+np.clip(delta,-self.velocity*dt,self.velocity*dt)
        self.previous=output.copy()
        return output
