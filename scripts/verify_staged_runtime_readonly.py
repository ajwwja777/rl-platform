"""CPU-only, read-only staged-runtime admission check. Prints a JSON report.

No HTTP server/client, process start/stop, publication, asset writes or Stage1.
Future updates below exist only in memory and use archived TRAIN batches.
"""
import argparse,hashlib,json,os,pickle,sys,threading,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--code-root',type=Path,required=True);parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--budget-only',action='store_true');args=parser.parse_args()
    root=args.code_root.resolve();run=args.runtime.resolve()
    sys.path[:0]=[str(root),str(root/'third_party/openpi-rlt/rlt_online_rl/src'),str(root/'third_party/openpi-rlt/packages/openpi-client/src')]
    import numpy as np
    import jax,jax.numpy as jnp,optax
    assert all(d.platform=='cpu'for d in jax.devices()),'This admission probe is CPU only'
    from collections import defaultdict
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from rlt_online_rl.inference import ActorService,ActorRequest,RLTPolicyInferenceWrapper
    from methods.openpi_rlt.experiments.supported_runtime import load_profile,build_components,SupportedIndex
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    names=['profile.json','teacher.pkl','action_norm_stats.json','checkpoints/latest.pkl','actor_snapshot/actor_snapshot.pkl','pending_actor/actor_snapshot.pkl','replay/replay_journal.pkl']
    protected={name:sha(run/name)for name in names};started=time.monotonic()
    profile=load_profile(run/'profile.json');payload=pickle.loads((run/'checkpoints/latest.pkl').read_bytes())
    assert payload['candidate_profile_sha256']==protected['profile.json']
    cfgd=dict(payload['rl_config']);cfgd['action_norm_stats_path']=str(run/'action_norm_stats.json');cfg=RLTOnlineRLConfig(**cfgd)
    actor,critic,step=build_components(cfg,profile);adapter=ActionRepresentationAdapter.from_config(cfg)
    def state_from(data):return trainer.RLTTrainState(**{k:trainer._tree_to_jax(v)for k,v in data.items()},actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
    initial=state_from(payload['state']);pending=pickle.loads((run/'pending_actor/actor_snapshot.pkl').read_bytes());served=pickle.loads((run/'actor_snapshot/actor_snapshot.pkl').read_bytes())
    assert pending['version']==int(initial.actor_version)
    for x,y in zip(jax.tree.leaves(pending['actor_params']),jax.tree.leaves(initial.actor_params)):np.testing.assert_array_equal(x,y)
    assert served['version']<pending['version'],'Expected current staged snapshot before promotion'
    rows=[]
    with(run/'replay/replay_journal.pkl').open('rb')as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    native_learner=object.__new__(trainer.LearnerService)
    native_learner._checkpoint_dir=str(run/'checkpoints');native_learner._rl_config=cfg
    native_learner._state=native_learner._load_latest_checkpoint()
    native_learner._write_status=lambda progress: None  # CPU probe has no status-file writes.
    budgets={}
    for added in [0,1,18,39]:
        progress=native_learner._refresh_progress(dict(size=len(rows)+added,adds_total=len(rows)+added))
        budgets[str(added)]=progress['pending_update_budget']
        assert progress['pending_update_budget']==added*cfg.grad_updates_per_cycle
    for x,y in zip(jax.tree.leaves(initial),jax.tree.leaves(native_learner.state)):np.testing.assert_array_equal(x,y)
    if args.budget_only:
        assert protected=={name:sha(run/name)for name in names}
        print(json.dumps(dict(native_load_all_state_equal=True,native_pending_update_budgets=budgets,checkpoint_progress=payload.get('progress'),rows=len(rows),served_version=served['version'],pending_version=pending['version'],all_production_assets_unchanged=True,boundary='Native checkpoint-load/progress methods, status writer replaced by no-op, no train_once or constructor/service invoked. Synthetic future counts only.')))
        return
    fields=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source','source_chunk','collection_phase_id','success','intervention_flag','episode_id','step_id']
    arrays={k:np.stack([r[k]for r in rows])for k in fields}
    index=SupportedIndex(run/'replay/replay_journal.pkl',cfg.gamma,profile['retention_episodes']);raw=index.attach(arrays);data=adapter.prepare_training_batch(raw)
    selected=np.random.default_rng(7281).integers(0,len(rows),(8,128))
    batches=[{k:jnp.asarray(v[ix])for k,v in data.items()}for ix in selected]
    def advance(state,batches):
        for batch in batches:
            state,metrics=step(state,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=cfg.online_bc_weight,q_weight=cfg.online_q_weight,delta_weight=cfg.delta_weight,use_action_adapter=True,action_q01=jnp.asarray(adapter.stats.q01),action_q99=jnp.asarray(adapter.stats.q99))
            assert all(np.isfinite(float(v))for v in metrics.values())
        return state
    direct=advance(initial,batches);half=advance(initial,batches[:4])
    serial={k:trainer._tree_to_numpy(getattr(half,k))for k in payload['state']}
    restored=state_from(pickle.loads(pickle.dumps(serial)));split=advance(restored,batches[4:])
    for x,y in zip(jax.tree.leaves(direct),jax.tree.leaves(split)):np.testing.assert_array_equal(x,y)
    def native_snapshot(path):
        # Invoke the actual native load/infer methods without starting its poll
        # thread or HTTP listener. Only immutable existing files are read.
        obj=object.__new__(ActorService);obj._rl_config=cfg;obj._action_adapter=adapter
        obj._wrapper=RLTPolicyInferenceWrapper(cfg);obj._lock=threading.Lock()
        obj._actor_params=None;obj._actor_version=-1;obj._rng=jax.random.PRNGKey(0)
        obj._snapshot_path=str(path);obj._logged_missing_params=False
        obj._try_reload_snapshot();return obj
    native=native_snapshot(run/'pending_actor/actor_snapshot.pkl');max_error=0.
    probes=np.linspace(0,len(rows)-1,64,dtype=int)
    for i in probes:
        request=ActorRequest(z_rl=arrays['z_rl'][i],proprio=arrays['proprio'][i],ref_chunk=arrays['ref_chunk'][i],deterministic=True,request_id=str(i),episode_id=int(arrays['episode_id'][i]),step_id=int(arrays['step_id'][i]))
        answer=native.infer(request);assert answer.actor_param_version==int(initial.actor_version)
        norm=adapter.normalize_ref_chunk(request.ref_chunk,request.proprio)
        pred=actor.sample_action(initial.actor_params,jax.random.PRNGKey(0),jnp.asarray(request.z_rl)[None],jnp.asarray(request.proprio)[None],jnp.asarray(norm)[None],deterministic=True)
        expected=adapter.denormalize_to_abs_chunk(np.asarray(pred)[0],request.proprio)
        max_error=max(max_error,float(np.max(np.abs(answer.refined_chunk-expected))))
    assert max_error<1e-6
    native._snapshot_path=str(run/'actor_snapshot/actor_snapshot.pkl');native._try_reload_snapshot()
    assert native.actor_param_version==pending['version'],'Native monotonic reload behavior changed'
    fresh=native_snapshot(run/'actor_snapshot/actor_snapshot.pkl');assert fresh.actor_param_version==served['version']
    predictions={};qvalues={};amean=jax.jit(actor.actor_mean);qvalue=jax.jit(critic.q_values)
    for label,params in [('served',trainer._tree_to_jax(served['actor_params'])),('pending',initial.actor_params)]:
        pp=[]
        for i in range(0,len(rows),128):
            sl=slice(i,i+128);pp.append(np.asarray(amean(params,jnp.asarray(data['z_rl'][sl]),jnp.asarray(data['proprio'][sl]),jnp.asarray(data['ref_chunk'][sl]))))
        predictions[label]=np.concatenate(pp);assert np.isfinite(predictions[label]).all()
    human=np.isin(arrays['source_chunk'],[2,3]);bc=np.where(human[...,None],data['action_chunk'],data['ref_chunk']);scale=(adapter.stats.q99[:6]-adapter.stats.q01[:6]+1e-6)/2
    episodes=defaultdict(list)
    for i,r in enumerate(rows):episodes[(int(r['collection_phase_id']),int(r['episode_id']))].append(i)
    per_episode=[]
    for ep,ix in episodes.items():
        row={'phase':ep[0],'episode':ep[1],'windows':len(ix),'assisted':bool(human[ix].any()),'success':bool(arrays['success'][ix].any()),'split':'TRAIN'}
        for label,pred in predictions.items():
            err=(pred[ix,...,:6]-bc[ix,...,:6])*scale
            row[label]={'BC6_RMSE_mrad':float(np.sqrt((err**2).mean())*1000),'BC6_abs_p95_mrad':float(np.quantile(np.abs(err),.95)*1000),'BC6_joint_RMSE_mrad':(np.sqrt((err**2).mean(axis=(0,1)))*1000).tolist()}
        delta=(predictions['pending'][ix,...,:6]-predictions['served'][ix,...,:6])*scale
        row['actor_change_abs_p99_mrad']=float(np.quantile(np.abs(delta),.99)*1000);per_episode.append(row)
    assert protected=={name:sha(run/name)for name in names},'Runtime assets changed during probe'
    result=dict(native_pending_update_budgets=budgets,status='verified_software_only',elapsed_sec=time.monotonic()-started,CPU_devices=[str(d)for d in jax.devices()],asset_sha256=protected,learner_step=int(initial.global_step),served_version=served['version'],pending_version=pending['version'],pending_all_actor_leaves_match_checkpoint=True,in_memory_eight_vs_four_plus_four_all_state_bitwise=True,in_memory_final_step=int(split.global_step),native_64_state_max_abs_error=max_error,live_lower_version_reload_ignored=True,fresh_instance_loads_old_version=True,all_production_assets_unchanged=True,train_episode_metrics=per_episode,boundary='Existing real Actor/Critic/state and TRAIN arrays on CPU only. Future8 updates are in memory, not committed or published. No Stage1/HTTP/robot/services, no independent TEST; native lower-version restore requires fresh Actor process. No claim of exact external Replay RNG restore.')
    print(json.dumps(result))
if __name__=='__main__':main()
