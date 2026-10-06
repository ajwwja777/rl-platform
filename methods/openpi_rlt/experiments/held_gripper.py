"""Opt-in Critic action projection for the fixed-gripper right-arm task.

The network and serialized parameter layout stay native. Current and target Q
see the held physical gripper in the same normalized action representation.
This is a control-contract adapter, not proof of autonomous action values.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class HeldGripperCritic:
    native: object
    q01: float
    q99: float

    def __post_init__(self):
        if not math.isfinite(self.q01) or not math.isfinite(self.q99) or self.q99<=self.q01:
            raise ValueError('Finite nondegenerate gripper normalization required')

    def project(self, action, proprio):
        import jax.numpy as jnp
        if action.shape[-1]!=7 or proprio.shape[-1]!=7:
            raise ValueError('Held-gripper Critic only supports right-arm 7D')
        held=(proprio[...,6]-self.q01)/(self.q99-self.q01+1e-6)*2.-1.
        return action.at[...,6].set(jnp.broadcast_to(held[...,None],action.shape[:-1]))

    def q_values(self, params, z_rl, proprio, action):
        return self.native.q_values(params,z_rl,proprio,self.project(action,proprio))

    def init_params(self, key):
        return self.native.init_params(key)
