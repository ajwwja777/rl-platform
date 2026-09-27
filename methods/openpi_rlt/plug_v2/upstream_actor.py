"""CPU inference adapter for pinned direct RLT actor; no residual/gain/clipping."""
import json,pickle,hashlib,sys
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/openpi-rlt/rlt_online_rl/src'))
import jax,jax.numpy as jnp
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import ChunkActor,TwinCritic
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from .rtc_upstream_core import rtc_proprio

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def diagnostic_delta(actor_plan,reference_plan):
    actor=np.asarray(actor_plan,np.float32);reference=np.asarray(reference_plan,np.float32)
    if actor.shape!=(10,14) or reference.shape!=(10,14):raise ValueError('invalid diagnostic plan')
    delta=actor[:,7:13]-reference[:,7:13]
    return {'horizon_rms':np.sqrt(np.mean(delta**2,axis=1)).tolist(),
            'joint_rms':np.sqrt(np.mean(delta**2,axis=0)).tolist(),
            'overall_rms':float(np.sqrt(np.mean(delta**2)))}
class UpstreamActor:
    def __init__(self,release):
        self.release=dict(release)
        path=Path(release['checkpoint']).resolve(strict=True);path.relative_to((ROOT/'runs/plug_v2/learning').resolve())
        stats=Path(release['norm_stats']).resolve(strict=True);stats.relative_to((ROOT/'runs/plug_v2/learning').resolve())
        if digest(path)!=release['checkpoint_sha256'] or digest(stats)!=release['norm_stats_sha256']:raise ValueError('immutable actor artifact changed')
        cfg=dict(release['rl_config']);cfg['action_norm_stats_path']=str(stats);cfg=RLTOnlineRLConfig(**cfg)
        if (cfg.proprio_dim,cfg.action_dim,cfg.chunk_len,cfg.z_dim)!=(99,7,10,2048):raise ValueError('actor RTC contract mismatch')
        state=pickle.load(path.open('rb'))['state'];self.learner_step=int(state['global_step']);self.actor_updates=int(state['actor_version']);self.version=self.actor_updates
        if self.learner_step!=release['global_step'] or self.actor_updates!=release['actor_updates']:raise ValueError('checkpoint version mismatch')
        self.adapter=ActionRepresentationAdapter.from_config(cfg)
        actor=ChunkActor(cfg.z_dim,cfg.proprio_dim,cfg.chunk_len,cfg.action_dim,cfg.actor_hidden_dim,cfg.actor_num_layers,cfg.fixed_std)
        params=jax.tree_util.tree_map(jnp.asarray,state['actor_params'])
        self._mean=jax.jit(lambda z,p,r:actor.actor_mean(params,z,p,r))
        critic=TwinCritic(cfg.z_dim,cfg.proprio_dim,cfg.chunk_len,cfg.action_dim,cfg.critic_hidden_dim,cfg.critic_num_layers)
        critic_params=jax.tree_util.tree_map(jnp.asarray,state['critic_params'])
        self._q=jax.jit(lambda z,p,a:critic.q_values(critic_params,z,p,a)[0])
        self.key=hashlib.sha256(json.dumps(release,sort_keys=True).encode()).hexdigest()
        # Compile without any device/robot observation dependency.
        zeros=(jnp.zeros((1,2048)),jnp.zeros((1,99)),jnp.zeros((1,10,7)));self._mean(*zeros).block_until_ready();self._q(*zeros).block_until_ready()
    def act(self,z,context,ref):
        z=np.asarray(z,np.float32);c=np.asarray(context,np.float32);ref=np.asarray(ref,np.float32)
        if z.shape!=(2048,) or c.shape!=(99,) or ref.shape!=(10,14) or not all(np.isfinite(x).all() for x in (z,c,ref)):raise ValueError('invalid actor input')
        if c[-1] not in (0.,1.):raise ValueError('untrained RTC delay')
        p=rtc_proprio(c)[None];r=self.adapter.normalize_ref_chunk(ref[None,:,7:14],p)
        mean=self._mean(jnp.asarray(z[None]),jnp.asarray(p),jnp.asarray(r))
        action=self.adapter.denormalize_to_abs_chunk(np.asarray(mean),p)[0]
        if not np.isfinite(action).all():raise ValueError('nonfinite upstream actor result')
        out=ref.copy();out[:,7:13]=action[:,:6]
        return out
    def diagnose(self,z,context,ref,actor_plan):
        z=np.asarray(z,np.float32);c=np.asarray(context,np.float32);ref=np.asarray(ref,np.float32);plan=np.asarray(actor_plan,np.float32)
        p=rtc_proprio(c)[None]
        ref_norm=self.adapter.normalize_ref_chunk(ref[None,:,7:14],p)
        actor_norm=self.adapter.normalize_chunk(plan[None,:,7:14],p)
        actor_q=float(np.asarray(self._q(jnp.asarray(z[None]),jnp.asarray(p),jnp.asarray(actor_norm)))[0])
        reference_q=float(np.asarray(self._q(jnp.asarray(z[None]),jnp.asarray(p),jnp.asarray(ref_norm)))[0])
        return {'actor_q':actor_q,'reference_q':reference_q,**diagnostic_delta(plan,ref)}
