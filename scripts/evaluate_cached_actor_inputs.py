"""CPU worker for immutable cached-input Actor/Critic checks; no live services."""
def main():
    import sys,io,base64,json,pickle,hashlib,time
    from pathlib import Path
    root=Path('/home/agilex/jiaan/project/rl-platform');sys.path[:0]=[str(root),str(root/'third_party/openpi-rlt/rlt_online_rl/src')]
    import numpy as np
    import jax,jax.numpy as jnp
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.experiments.supported_runtime import load_profile,build_components
    run=Path('/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_online_20261006_runtime_v4');old=run.parent/'supported_online_20261006_v4'
    assert all(d.platform=='cpu'for d in jax.devices())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=[run/'checkpoints/latest.pkl',old/'checkpoints/latest.pkl',run/'actor_snapshot/actor_snapshot.pkl',run/'action_norm_stats.json',run/'profile.json']
    protected={str(p):sha(p)for p in files};started=time.monotonic()
    cp=pickle.loads(files[0].read_bytes());initial=pickle.loads(files[1].read_bytes());cfgd=dict(cp['rl_config']);cfgd['action_norm_stats_path']=str(files[3]);cfg=RLTOnlineRLConfig(**cfgd)
    profile=load_profile(run/'profile.json');actor,critic,_=build_components(cfg,profile);adapter=ActionRepresentationAdapter.from_config(cfg)
    raw=dict(np.load(io.BytesIO(base64.b64decode(sys.stdin.read()))));norm=np.stack([adapter.normalize_ref_chunk(ref,pro)for ref,pro in zip(raw['ref_chunk'],raw['proprio'])]);actions=np.stack([adapter.normalize_ref_chunk(act,pro)for act,pro in zip(raw['action_chunk'],raw['proprio'])])
    actfun=jax.jit(actor.actor_mean);qfun=jax.jit(critic.q_values);out={}
    for name,payload in [('initial',initial),('pending',cp)]:
     params=jax.tree_util.tree_map(jnp.asarray,payload['state']['actor_params']);qparams=jax.tree_util.tree_map(jnp.asarray,payload['state']['critic_params']);pred=[];qs=[]
     for i in range(0,len(norm),128):
      sl=slice(i,i+128);z=jnp.asarray(raw['z_rl'][sl]);pro=jnp.asarray(raw['proprio'][sl]);ref=jnp.asarray(norm[sl]);act=jnp.asarray(actions[sl]);aa=actfun(params,z,pro,ref);pred.append(np.asarray(aa));q=qfun(qparams,z,pro,act);qs.append(np.stack([np.asarray(x)for x in q],axis=1))
     out[name+'_actor']=np.concatenate(pred);out[name+'_Q_behavior']=np.concatenate(qs)
    for name,payload in [('initial',initial),('pending',cp)]:
     qparams=jax.tree_util.tree_map(jnp.asarray,payload['state']['critic_params'])
     for alternative,aa in [('initial_actor',out['initial_actor']),('pending_actor',out['pending_actor']),('reference',norm)]:
      qs=[]
      for i in range(0,len(norm),128):
       sl=slice(i,i+128);q=qfun(qparams,jnp.asarray(raw['z_rl'][sl]),jnp.asarray(raw['proprio'][sl]),jnp.asarray(aa[sl]));qs.append(np.stack([np.asarray(x)for x in q],axis=1))
      out[name+'_Q_'+alternative]=np.concatenate(qs)
    assert all(np.isfinite(x).all()for x in out.values())
    assert protected=={str(p):sha(p)for p in files}
    buffer=io.BytesIO();np.savez_compressed(buffer,**out)
    print(json.dumps(dict(elapsed_sec=time.monotonic()-started,protected_sha256=protected,states=len(norm),all_finite=True,input_fields=list(raw),result_npz_base64=base64.b64encode(buffer.getvalue()).decode(),boundary='Repeated DEV cached-input CPU inference, no Stage1/GPU/HTTP/robot/assets writes. No independent TEST.')))

if __name__=='__main__':main()
