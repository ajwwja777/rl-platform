#!/usr/bin/env python3
"""Actual candidate model/native Learner resume and export checks on CPU.

Uses archived training batches and a synthetic adds_total counter. It does not
claim new real experience, autonomous gain, Stage1 forward or field execution.
"""
import argparse,dataclasses,hashlib,json,os,pickle,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import jax,jax.numpy as jnp,optax
from rlt_online_rl import trainer
from rlt_online_rl.config import load_system_config_yaml,LearnerServiceConfig
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.replay import ReplayBuffer
from methods.openpi_rlt.experiments.supported_runtime import load_profile,build_components,install_supported_learner,SupportedIndex,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    o=a.output;o.mkdir(exist_ok=False)
    package=a.candidate.resolve();profile_path=package/'profile.json';profile=load_profile(profile_path)
    protected={str(p):sha256(p)for p in package.rglob('*')if p.is_file()}
    system=load_system_config_yaml(str(package/'online.yaml'));cfg=system.rl
    data=[]
    with Path(system.replay.journal_path).open('rb')as f:
        while True:
            try:data.append(pickle.load(f))
            except EOFError:break
    fields=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source','source_chunk','collection_phase_id','success','intervention_flag','episode_id','step_id']
    arrays={k:np.stack([r[k]for r in data])for k in fields}
    sampler=ReplayBuffer(len(data),seed=42,sample_strategy='stratified',recent_episode_window=20,recent_online_ratio=.4,warmup_demo_ratio=.3,human_intervention_ratio=.2)
    sampler._size=len(data);sampler._storage=arrays
    indices=np.stack([sampler._sample_stratified_indices(128)for _ in range(8)]);np.save(o/'batch_indices.npy',indices)
    index=SupportedIndex(system.replay.journal_path,cfg.gamma,profile['retention_episodes'])
    batches=[index.attach({k:v[ids]for k,v in arrays.items()})for ids in indices]
    adapter=ActionRepresentationAdapter.from_config(cfg)
    actor,critic,step=build_components(cfg,profile)
    payload=pickle.loads((package/'checkpoints/latest.pkl').read_bytes())
    initial=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v)for k,v in payload['state'].items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
    package_export=pickle.loads((package/'actor_snapshot/actor_snapshot.pkl').read_bytes())
    assert package_export['version']==int(initial.actor_version)
    for x,y in zip(jax.tree.leaves(package_export['actor_params']),jax.tree.leaves(initial.actor_params)):np.testing.assert_array_equal(x,y)
    direct=initial
    for raw in batches:
        b={k:jnp.asarray(v)for k,v in adapter.prepare_training_batch(raw).items()}
        direct,metrics=step(direct,b,actor=actor,critic=critic,rl_config=cfg,bc_weight=cfg.online_bc_weight,q_weight=cfg.online_q_weight,delta_weight=cfg.delta_weight,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
        assert all(np.isfinite(float(v))for v in metrics.values())
    install_supported_learner(profile_path,system.replay.journal_path)
    class Source:
        def __init__(self,cursor=0):self.cursor=cursor
        def stats(self):return dict(size=len(data),adds_total=len(data)+8,max_episode_id=int(arrays['episode_id'].max()),recent_episode_window=20)
        def sample_batch(self,size):
            assert size==128
            b=batches[self.cursor];self.cursor+=1;return b
    def service(name,source):
        run=o/'candidates'/name;cp=run/'checkpoints';cp.mkdir(parents=True,exist_ok=True)
        if not(cp/'latest.pkl').exists():shutil.copyfile(package/'checkpoints/latest.pkl',cp/'latest.pkl')
        sc=LearnerServiceConfig(checkpoint_dir=str(cp),actor_snapshot_path=str(run/'pending_actor/actor_snapshot.pkl'),sample_batch_size=128,push_actor_interval_steps=50,checkpoint_interval_steps=50)
        return trainer.LearnerService(cfg,sc,source,metrics_path=str(run/'metrics/learner.jsonl'))
    continuous=service('continuous',Source())
    for _ in range(8):assert continuous.train_once()is not None
    split=service('split',Source())
    for _ in range(4):assert split.train_once()is not None
    split.flush_artifacts();split=service('split',Source(4))
    for _ in range(4):assert split.train_once()is not None
    split.flush_artifacts()
    for state in [continuous.state,split.state]:
        for x,y in zip(jax.tree.leaves(direct),jax.tree.leaves(state)):np.testing.assert_array_equal(x,y)
    assert int(split.state.global_step)==int(initial.global_step)+8
    pending=o/'candidates/split/pending_actor/actor_snapshot.pkl'
    exported=pickle.loads(pending.read_bytes())
    for x,y in zip(jax.tree.leaves(exported['actor_params']),jax.tree.leaves(split.state.actor_params)):np.testing.assert_array_equal(x,y)
    from rlt_online_rl.inference import ActorService,ActorRequest
    from rlt_online_rl.config import ActorServiceConfig
    native=ActorService(cfg,ActorServiceConfig(snapshot_path=str(pending)))
    deadline=time.monotonic()+30
    while native.actor_param_version != int(split.state.actor_version):
        if time.monotonic()>deadline: raise RuntimeError('Actor snapshot did not become ready')
        time.sleep(.01)
    max_error=0.
    for i in range(32):
        request=ActorRequest(z_rl=arrays['z_rl'][i],proprio=arrays['proprio'][i],ref_chunk=arrays['ref_chunk'][i],deterministic=True,request_id=str(i),episode_id=int(arrays['episode_id'][i]),step_id=int(arrays['step_id'][i]))
        result=native.infer(request)
        assert result.actor_param_version == int(split.state.actor_version)
        assert result.source == 1
        norm=adapter.normalize_ref_chunk(request.ref_chunk,request.proprio)
        expected=actor.sample_action(split.state.actor_params,jax.random.PRNGKey(0),jnp.asarray(request.z_rl,dtype=jnp.float32)[None],jnp.asarray(request.proprio,dtype=jnp.float32)[None],jnp.asarray(norm)[None],deterministic=True)
        expected=adapter.denormalize_to_abs_chunk(np.asarray(expected)[0],request.proprio)
        max_error=max(max_error,float(np.max(np.abs(result.refined_chunk-expected))))
    native._stop_event.set()
    native._poll_thread.join(timeout=2)
    print('native_actor_max_error',max_error,flush=True)
    assert max_error<1e-6
    from integrations.cobot_runtime.evaluation_env import prepare_config
    frozen_config=o/'frozen-config.yaml'
    prepare_config('frozen',package/'actor_snapshot/actor_snapshot.pkl',frozen_config)
    frozen_system=load_system_config_yaml(str(frozen_config))
    frozen=ActorService(frozen_system.rl,frozen_system.actor_service)
    deadline=time.monotonic()+30
    while frozen.actor_param_version!=int(initial.actor_version):
        if time.monotonic()>deadline:raise RuntimeError('Frozen Actor did not become ready')
        time.sleep(.01)
    frozen_error=0.
    try:
        for i in range(32):
            request=ActorRequest(z_rl=arrays['z_rl'][i],proprio=arrays['proprio'][i],ref_chunk=arrays['ref_chunk'][i],deterministic=True,request_id=str(i),episode_id=int(arrays['episode_id'][i]),step_id=int(arrays['step_id'][i]))
            response=frozen.infer(request)
            assert response.actor_param_version==int(initial.actor_version)and response.source==1
            norm=adapter.normalize_ref_chunk(request.ref_chunk,request.proprio)
            expected=actor.sample_action(initial.actor_params,jax.random.PRNGKey(0),jnp.asarray(request.z_rl,dtype=jnp.float32)[None],jnp.asarray(request.proprio,dtype=jnp.float32)[None],jnp.asarray(norm)[None],deterministic=True)
            physical=adapter.denormalize_to_abs_chunk(np.asarray(expected)[0],request.proprio)
            frozen_error=max(frozen_error,float(np.max(np.abs(response.refined_chunk-physical))))
        assert frozen_error<1e-6
    finally:
        frozen._stop_event.set();frozen._poll_thread.join(timeout=2)
    assert protected=={str(p):sha256(Path(p))for p in protected}
    result=dict(status='verified',devices=[str(d)for d in jax.devices()],updates=8,new_actor_updates=int(split.state.actor_version-initial.actor_version),step=int(split.state.global_step),actor_version=int(split.state.actor_version),direct_native_service_and_four_plus_four_resume_bitwise=True,export_all_leaves_equal=True,package_export_all_leaves_equal=True,frozen_actual_candidate32_max_error=frozen_error,native_actor32_max_absolute_error=max_error,source_package_unchanged=True,profile_sha256=sha256(profile_path),batch_indices_sha256=sha256(o/'batch_indices.npy'),boundary='Actual candidate weights/native networks/optimizer and archived batches; synthetic new-arrival counter. Exact learner resume with the same recorded batches; external live Replay RNG restoration, GPU Stage1, field timing and autonomous capability are not certified.')
    (o/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
