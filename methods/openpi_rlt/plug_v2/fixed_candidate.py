"""Immutable fixed-only candidate. CPU core service + explicit deterministic profile.
No ROS/CAN imports. The core service intentionally returns the legacy .05 residual;
the wrapper reconstructs the exact CorrectiveActor forward validated in training.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .actor_service import select_actor
ROOT=Path(__file__).resolve().parents[3]
BUNDLE=ROOT/'runs/plug_v2/learning/candidates/rtc-corrective-r1'
def manifest(require_ready=True):
    m=json.loads((BUNDLE/'manifest.json').read_text())
    if require_ready and m['status']!='ready_for_fixed_onsite_trial':raise ValueError('fixed candidate blocked: status='+str(m['status']))
    path=Path(m['path']).resolve(strict=True)
    if path!=BUNDLE/'actor.pt' or hashlib.sha256(path.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('candidate fingerprint mismatch')
    current=json.loads((ROOT/'deployments/plug_v2/manifest.json').read_text())
    if m['stage1_checkpoint']!=current['checkpoint']:raise ValueError('candidate frozen Stage-1 mismatch')
    return m
class FixedCorrectiveClient:
    def __init__(self,core,config):self.core=core;self.config=config
    def act(self,z,context,ref):
        if self.config.get('apply_only_with_committed_prefix') and context[-1]<.5:return ref.copy()
        base=self.core.act(z,context,ref);p=self.config;g=p['prefix_gate']
        distance=np.abs(context[84:98][7:13]).max()
        gain=g['low']+(g['high']-g['low'])/(1+np.exp(np.clip((distance-g['threshold'])/g['temperature'],-80,80)))
        prior=ref.copy();prior[:,7:13]+=np.clip(p['brake_beta']*(context[None,7:13]-ref[:,7:13]),-.05,.05)
        corrected=ref.copy();corrected[:,7:13]=ref[:,7:13]+(p['raw_residual_radius']/.05)*(base[:,7:13]-ref[:,7:13])
        result=gain*corrected+(1-gain)*prior
        if not np.isfinite(result).all():raise ValueError('nonfinite fixed corrective action')
        return result.astype(np.float32)
def select_fixed_candidate(require_ready=True):
    m=manifest(require_ready);core,version=select_actor(m['path'])
    if core.key!=m['sha256'] or version!=m['version']:raise ValueError('core service loaded a different immutable candidate')
    return FixedCorrectiveClient(core,m['inference_config']),version
