#!/usr/bin/env python3
"""Isolated offline warmup. Never writes to the live session or serves an actor."""
import argparse,dataclasses,hashlib,importlib.util,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--live-run",type=Path,required=True);ap.add_argument("--runtime-overlay",type=Path,required=True);ap.add_argument("--upstream",type=Path,required=True)
    args=ap.parse_args();out=args.run.resolve();live=args.live_run.resolve()
    if out==live or live in out.parents:raise ValueError("offline output must be isolated from live session")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["COBOT_RLT_R2_SMOOTHNESS"]="1"
    os.environ["COBOT_RLT_ACTIVE_ARM"]="right"
    os.environ["TMPDIR"]=str(out)
    sys.path[:0]=[str(args.runtime_overlay),str(args.upstream/"rlt_online_rl/src")]
    spec=importlib.util.spec_from_file_location("offline_warmup",Path(__file__).resolve().parents[1]/"cobot_adapter/offline_warmup.py")
    data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
    import jax,jax.numpy as jnp
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    from methods.openpi_rlt.cobot_adapter.status_io import install_status_write_throttle
    install_status_write_throttle()
    from rlt_online_rl.config import load_system_config_yaml,save_system_config_yaml
    from rlt_online_rl.replay import RLTTransition
    from rlt_online_rl.trainer import LearnerService
    assert jax.devices()[0].platform=="gpu"
    if (out/"training/checkpoints/latest.pkl").exists():raise FileExistsError("run already trained; explicit audited resume required")
    protect=[live/"replay/replay_journal.pkl",live/"checkpoints/latest.pkl",live/"actor_snapshot/actor_snapshot.pkl",live/"metrics/learner_status.json"]
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protect if p.exists()}
    before=hashes()
    online=[]
    with protect[0].open("rb") as f:
        while True:
            try:online.append(pickle.load(f))
            except EOFError:break
    by_ep={}
    for r in online:by_ep.setdefault(int(r["episode_id"]),[]).append(r)
    for ep,records in by_ep.items():
        assert sum(bool(r["done"]) for r in records)==1
        reward=sum(float(np.sum(r["rewards"])) for r in records)
        assert reward in [0.,1.]
        success=reward==1
        hil=any(bool(r["intervention_flag"]) for r in records)
        group=("success_hil" if hil else "success_auto") if success else "failure"
        for r in records:r["group"]=group
    online_train,online_val=data.split_online_episodes(online,seed=42)
    for r in online:r["group"]="success" if r["group"].startswith("success") else "failure"
    with (out/"expert_replay.pkl").open("rb") as f:expert=pickle.load(f)
    train=online_train+[r for r in expert if r["split"]=="train"]
    val=online_val+[r for r in expert if r["split"]=="val"]
    assert not {int(r["episode_id"]) for r in train}&{int(r["episode_id"]) for r in val}
    def summarize(rs):
        return {g:dict(episodes=len({int(r["episode_id"]) for r in rs if r["group"]==g}),transitions=sum(r["group"]==g for r in rs),episode_ids=sorted({int(r["episode_id"]) for r in rs if r["group"]==g})) for g in data.GROUPS}
    frozen=dict(train=train,validation=val)
    with (out/"training_dataset.pkl").open("xb") as f:pickle.dump(frozen,f)
    manifest=dict(seed=42,train=summarize(train),validation=summarize(val),ratios=[.3,.4,.3],batch_counts=[38,51,39],live_hashes=before,expert_sha256=hashlib.sha256((out/"expert_replay.pkl").read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256((out/"training_dataset.pkl").read_bytes()).hexdigest(),steps=[100,500,2000,5000,10000,20000],bc_weight=10.,q_weight=.1,delta_weight=10.)
    (out/"training_manifest.json").write_text(json.dumps(manifest,indent=2))
    sampler=data.EpisodeStratifiedSampler(train,seed=42)
    converted=[RLTTransition.from_mapping(r).to_numpy() for r in train]
    arrays={k:np.stack([r[k] for r in converted]) for k in converted[0]}
    assert all(np.isfinite(v).all() for v in arrays.values())
    class Source:
        def stats(self):return dict(size=len(train),adds_total=len(train),max_episode_id=max(int(r["episode_id"]) for r in train),recent_episode_window=20)
        def sample_batch(self,n):
            ix=sampler.sample_indices(n);return {k:v[ix] for k,v in arrays.items()}
    cfg=load_system_config_yaml(str(live/"resolved_online.yaml"))
    rl=dataclasses.replace(cfg.rl,warmup_post_collect_updates=20000,freeze_after_warmup=True,warmup_bc_weight=10.,warmup_q_weight=.1)
    svc=dataclasses.replace(cfg.learner_service,checkpoint_dir=str(out/"training/checkpoints"),actor_snapshot_path=str(out/"training/actor/actor.pkl"),checkpoint_interval_steps=1000000,push_actor_interval_steps=1000000)
    cfg=dataclasses.replace(cfg,rl=rl,learner_service=svc)
    save_system_config_yaml(cfg,str(out/"training_config.yaml"))
    learner=LearnerService(rl,svc,Source(),rng=jax.random.PRNGKey(42),metrics_path=str(out/"training/metrics/learner.jsonl"))
    adapter=learner._action_adapter
    @jax.jit
    def infer(params,z,prop,ref):return learner._actor.actor_mean(params,z,prop,ref)
    def predictions(rs):
        chunks=[]
        for start in range(0,len(rs),128):
            part=rs[start:start+128];n=len(part)
            part=part+[part[-1]]*(128-n)
            props=np.stack([r["proprio"] for r in part]);ref=np.stack([r["ref_chunk"] for r in part])
            normalized=adapter.normalize_ref_chunk(ref,props)
            pred=np.array(infer(learner.state.actor_params,jnp.asarray(np.stack([r["z_rl"] for r in part]),dtype=jnp.float32),jnp.asarray(props),jnp.asarray(normalized)))
            chunks.extend(adapter.denormalize_to_abs_chunk(pred,props)[:n])
        return np.asarray(chunks)
    reference={g:data.action_metrics(np.stack([r["ref_chunk"] for r in val if r["group"]==g]),[r for r in val if r["group"]==g]) for g in data.GROUPS}
    (out/"reference_validation.json").write_text(json.dumps(reference,indent=2))
    evaluations=[];start=time.perf_counter();best=float("inf");bad=0
    def evaluate(step):
        pred=predictions(val)
        result=dict(step=step,elapsed_sec=time.perf_counter()-start,groups={})
        for group in data.GROUPS:
            ids=[i for i,r in enumerate(val) if r["group"]==group]
            result["groups"][group]=data.action_metrics(pred[ids],[val[i] for i in ids])
        result["score"]=float(np.mean([result["groups"][g]["right_rmse_rad"] for g in ["expert","success"]]))
        result["smoothness_gate"]=all(
            result["groups"][g]["velocity_p95_rad_per_sample"]<=max(.01,reference[g]["velocity_p95_rad_per_sample"]*2)
            and result["groups"][g]["first_delta_p95_rad"]<=max(.04,reference[g]["first_delta_p95_rad"]*1.5)
            for g in data.GROUPS)
        evaluations.append(result)
        (out/"evaluations.json").write_text(json.dumps(evaluations,indent=2))
        np.savez_compressed(out/f"validation_predictions_step_{step}.npz",actions=pred)
        print("EVALUATION "+json.dumps(result),flush=True)
        return result
    evaluate(0)
    stopped=None
    for i in range(1,20001):
        metrics=learner.train_once()
        if metrics is None or not all(np.isfinite(v) for v in metrics.values()):
            stopped="nonfinite_or_missing_update";break
        if i in manifest["steps"]:
            learner.save_checkpoint();learner.export_actor_snapshot(force=True)
            ev=evaluate(i)
            if i>=2000:
                if ev["score"]>best*1.5:bad+=1
                else:bad=0
                best=min(best,ev["score"])
                if bad>=2:stopped="validation_regression_two_checkpoints";break
        if i%1000==0:print(json.dumps(dict(step=i,elapsed_sec=time.perf_counter()-start)),flush=True)
    candidates=[x for x in evaluations if x["step"]>=500 and x["smoothness_gate"]]
    recommended=min(candidates,key=lambda x:x["score"])["step"] if candidates else None
    unchanged=before==hashes()
    assert unchanged,"live artifacts changed during offline training"
    result=dict(status="completed" if i==20000 and stopped is None else "stopped",last_step=int(learner.state.global_step),actor_version=int(learner.state.actor_version),elapsed_sec=time.perf_counter()-start,stop_reason=stopped,recommended_step=recommended,selection="minimum expert/current-success validation RMSE among smoothness-gated checkpoints; offline proxy only",formal_artifacts_unchanged=unchanged,not_deployed=True)
    (out/"training_result.json").write_text(json.dumps(result,indent=2))
    print("RESULT "+json.dumps(result),flush=True)
if __name__=="__main__":main()
