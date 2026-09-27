"""Fixed-trial corrective actor, explicit training/inference command contract."""
import math,torch
from .learning import Actor,JOINTS
class CorrectiveActor(Actor):
 def __init__(self,config):
  super().__init__();self.config=config;self.radius=float(config['raw_residual_radius']);self.beta=float(config['brake_beta']);self.gate={k:float(v) for k,v in config['prefix_gate'].items()}
  if not all(math.isfinite(v) for v in [self.radius,self.beta,*self.gate.values()]):raise ValueError('nonfinite policy config')
  if not 0<self.radius<=.2 or not 0<=self.beta<=1 or not 0<=self.gate['low']<=self.gate['high']<=1 or not 0<self.gate['temperature']<=.1 or not 0<=self.gate['threshold']<=.1:raise ValueError('unsupported policy config')
 def components(self,c,ref):
  distance=c[:,84:98][:,JOINTS].abs().max(1).values;g=self.gate;gain=(g['low']+(g['high']-g['low'])*torch.sigmoid((g['threshold']-distance)/g['temperature']))[:,None,None];prior=ref.clone();prior[...,JOINTS]+=(self.beta*(c[:,None,JOINTS]-ref[...,JOINTS])).clamp(-.05,.05);return gain,prior
 def forward(self,z,c,ref,dropout=0.):
  base=super().forward(z,c,ref,dropout);core=ref+(self.radius/.05)*(base-ref);gain,prior=self.components(c,ref);result=gain*core+(1-gain)*prior
  return torch.where((c[:,-1]>.5)[:,None,None],result,ref) if self.config.get('apply_only_with_committed_prefix') else result
 def policy_bounds(self,c,ref):
  gain,prior=self.components(c,ref);low=ref.clone();high=ref.clone();low[...,JOINTS]-=self.radius;high[...,JOINTS]+=self.radius;return gain*low+(1-gain)*prior,gain*high+(1-gain)*prior
