#!/usr/bin/env python3
"""Transactional 32-step updates at committed episode boundaries, no robot I/O."""
import argparse,dataclasses,fcntl,hashlib,json,os,pickle,shutil,signal,sys,time,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.openpi_rlt.cobot_adapter.online_cycle import atomic_json,load_journal,complete_episodes,pending_episodes,accept_candidate,read_optional_json,project_training_action
SEED_HASH="ea4770d1609c380a3029bcceebe37689d019f10a7f3831e2ef3bcfbf9dc58ccd"
def publish(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(".tmp")
    with source.open("rb") as src,tmp.open("wb") as dst:
        shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
    os.replace(tmp,target)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,required=True);ap.add_argument("--warmup",type=Path,required=True)
    ap.add_argument("--upstream",type=Path,required=True);ap.add_argument("--freeze-updates",action="store_true")
    ap.add_argument("--active-arm",choices=["left","right","both"],default="right")
    ap.add_argument("--rollback",action="store_true");ap.add_argument("--once",action="store_true")
    args=ap.parse_args();run=args.run.resolve();warm=args.warmup.resolve()
    if run==warm or warm in run.parents:raise ValueError("online run must be isolated from warmup")
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/"worker.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pointer=run/"current.json";actor_path=run/"actor_snapshot/actor_snapshot.pkl";status_path=run/"metrics/learner_status.json"
    if args.rollback:
        current=json.loads(pointer.read_text())
        previous=current.get("previous") or current["seed"]
        new={**current,**previous,"previous":None,"rollback":True}
        atomic_json(pointer,new);publish(Path(new["actor"]),actor_path)
        atomic_json(status_path,dict(phase="stopped",ready_for_online=False,actor_version=new["actor_version"],global_step=new["global_step"],last_decision="manual_rollback"))
        print(json.dumps(new),flush=True);return
    import numpy as np
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false";os.environ["COBOT_RLT_R2_SMOOTHNESS"]="1";os.environ["TMPDIR"]=str(run)
    sys.path.insert(0,str(args.upstream/"rlt_online_rl/src"))
    import jax,jax.numpy as jnp
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    from methods.openpi_rlt.cobot_adapter.offline_warmup import EpisodeStratifiedSampler,action_metrics
    from rlt_online_rl.config import load_system_config_yaml
    from rlt_online_rl.replay import RLTTransition
    from rlt_online_rl.trainer import LearnerService
    seed_ck=warm/"training/checkpoints/step_10000.pkl";seed_actor=warm/"training/actor/history/actor_v005000.pkl"
    if hashlib.sha256(seed_ck.read_bytes()).hexdigest()!=SEED_HASH:raise ValueError("warmup checkpoint hash mismatch")
    with (warm/"training_dataset.pkl").open("rb") as f:frozen=pickle.load(f)
    expected=json.loads((warm/"training_manifest.json").read_text())["dataset_sha256"]
    if hashlib.sha256((warm/"training_dataset.pkl").read_bytes()).hexdigest()!=expected:raise ValueError("frozen dataset hash mismatch")
    seed=dict(checkpoint=str(seed_ck),actor=str(seed_actor),global_step=10000,actor_version=5000)
    if not pointer.exists():atomic_json(pointer,{**seed,"seed":seed,"processed":[],"transactions":0,"previous":None})
    current=json.loads(pointer.read_text())
    publish(Path(current["actor"]),actor_path)
    cfg=load_system_config_yaml(str(warm/"training_config.yaml"))
    rl=dataclasses.replace(cfg.rl,freeze_after_warmup=False,warmup_post_collect_updates=10000,online_bc_weight=10.,online_q_weight=.1)
    work=run/"worker";(work/"checkpoints").mkdir(parents=True,exist_ok=True)
    publish(Path(current["checkpoint"]),work/"checkpoints/latest.pkl")
    class Source:
        def set_records(self,records):
            records=[project_training_action(r,active_arm=args.active_arm,hold_grippers=True) for r in records]
            self.records=records;self.sampler=EpisodeStratifiedSampler(records,seed=42+int(current["transactions"]))
            values=[RLTTransition.from_mapping(r).to_numpy() for r in records]
            self.arrays={k:np.stack([v[k] for v in values]) for k in values[0]}
        def stats(self):return dict(size=len(self.records),adds_total=len(self.records))
        def sample_batch(self,n):
            ix=self.sampler.sample_indices(n);return {k:v[ix] for k,v in self.arrays.items()}
    source=Source();source.set_records(frozen["train"])
    service=dataclasses.replace(cfg.learner_service,checkpoint_dir=str(work/"checkpoints"),actor_snapshot_path=str(work/"actor.pkl"),checkpoint_interval_steps=1000000,push_actor_interval_steps=1000000)
    # A private instance has no free-running budget and no public artifact writes.
    learner=LearnerService(rl,service,source,metrics_path=str(work/"metrics/learner.jsonl"))
    learner._cycle_target=int(learner.state.global_step)
    learner._desired_total_updates=types.MethodType(lambda self,size,adds:self._cycle_target,learner)
    learner._write_status=types.MethodType(lambda self,progress:None,learner)
    if not args.freeze_updates:
        compile_started=time.perf_counter()
        saved_state=learner._state
        saved_metrics=learner._metrics_path
        learner._metrics_path=None
        learner._cycle_target=int(saved_state.global_step)+2
        for _ in range(2):
            compiled=learner.train_once()
            if compiled is None or not all(np.isfinite(v) for v in compiled.values()):
                raise ValueError("precompile failed")
        learner._state=saved_state
        learner._cycle_target=int(saved_state.global_step)
        learner._metrics_path=saved_metrics
        source.set_records(frozen["train"])
        print("ONLINE_PRECOMPILE_READY "+str(time.perf_counter()-compile_started),flush=True)
    adapter=learner._action_adapter
    @jax.jit
    def infer(params,z,prop,ref):return learner._actor.actor_mean(params,z,prop,ref)
    def evaluate():
        rs=frozen["validation"];pred=[]
        for start in range(0,len(rs),128):
            part=rs[start:start+128];n=len(part);part=part+[part[-1]]*(128-n)
            props=np.stack([r["proprio"] for r in part]);refs=np.stack([r["ref_chunk"] for r in part])
            y=infer(learner.state.actor_params,jnp.asarray(np.stack([r["z_rl"] for r in part]),dtype=jnp.float32),jnp.asarray(props),jnp.asarray(adapter.normalize_ref_chunk(refs,props)))
            pred.extend(adapter.denormalize_to_abs_chunk(np.array(y),props)[:n])
        pred=np.asarray(pred);groups={}
        for g in ["expert","success","failure"]:
            ids=[i for i,r in enumerate(rs) if r["group"]==g];groups[g]=action_metrics(pred[ids],[rs[i] for i in ids])
        summary=dict(score=float(np.mean([groups[g]["right_rmse_rad"] for g in ["expert","success"]])),
            velocity=max(x["velocity_p95_rad_per_sample"] for x in groups.values()),
            acceleration=max(x["acceleration_p95_rad_per_sample2"] for x in groups.values()),
            first=max(x["first_delta_p95_rad"] for x in groups.values()),
            boundary=max(x["boundary_jump_p95_rad"] or 0 for x in groups.values()))
        return summary
    baseline=evaluate()
    seed_eval=next(x for x in json.loads((warm/"evaluations.json").read_text()) if x["step"]==10000)
    sg=seed_eval["groups"]
    initial_baseline=dict(score=seed_eval["score"],velocity=max(x["velocity_p95_rad_per_sample"] for x in sg.values()),acceleration=max(x["acceleration_p95_rad_per_sample2"] for x in sg.values()),first=max(x["first_delta_p95_rad"] for x in sg.values()),boundary=max(x["boundary_jump_p95_rad"] or 0 for x in sg.values()))
    stop=False
    def stop_requested(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    last_request=None;last_decision="seed_loaded";last_updates=0;last_elapsed=0.
    def status(phase,request_id=None,**extra):
        atomic_json(status_path,dict(worker_pid=os.getpid(),phase=phase,ready_for_online=phase=="ready",actor_version=current["actor_version"],
            global_step=current["global_step"],pending_update_budget=32 if phase=="updating" else 0,
            last_decision=last_decision,last_update_steps=last_updates,last_update_sec=last_elapsed,
            updates_per_episode=32,episodes_per_update=1,processed_episodes=len(current["processed"]),
            request_id=request_id,training_enabled=not args.freeze_updates,collection_only=False,
            actor_source="warmup_step10000_then_episode_updates",training_action_contract="controlled_arm_reference_passive_v1",active_arm=args.active_arm,hold_grippers=True,timestamp=time.time(),**extra))
    status("ready")
    print("ONLINE_READY "+json.dumps(current),flush=True)
    try:
        while not stop:
            p=run/"update_request.json"
            request=read_optional_json(p)
            rid=request.get("request_id")
            if rid and rid!=last_request:
                groups=complete_episodes(load_journal(run/"replay/replay_journal.pkl"))
                pending=pending_episodes(groups,current["processed"])
                if args.freeze_updates:
                    if pending:
                        current={**current,"processed":sorted(set(current["processed"])|set(pending))}
                        atomic_json(pointer,current)
                    last_decision="frozen_actor";last_updates=0;last_elapsed=0.
                    pending=[]
                for ep in pending:
                    if stop:break
                    started=time.perf_counter();status("updating",rid,updating_episode=ep)
                    records=list(frozen["train"])
                    for eid,rs in groups.items():
                        if eid>ep:continue
                        records.extend([{**r,"episode_id":1000000+eid} for r in rs])
                    source.set_records(records)
                    previous={k:current[k] for k in ["checkpoint","actor","global_step","actor_version"]}
                    learner._cycle_target=int(learner.state.global_step)+32
                    for n in range(32):
                        if stop:break
                        m=learner.train_once()
                        if m is None or not all(np.isfinite(v) for v in m.values()):raise ValueError("nonfinite/missing update")
                    if stop:break
                    candidate=evaluate()
                    accepted=accept_candidate(baseline,candidate) and accept_candidate(initial_baseline,candidate)
                    number=int(current["transactions"])+1
                    txn=run/"updates"/f"batch_{number:06d}"
                    # A unique attempt directory allows safe retry after an interrupted pre-commit write.
                    attempt=txn.with_name(txn.name+"_"+str(time.time_ns()));attempt.mkdir(parents=True)
                    saved=Path(learner.save_checkpoint());publish(saved,attempt/"latest.pkl")
                    learner.export_actor_snapshot(force=True);publish(work/"actor.pkl",attempt/"actor.pkl")
                    last_updates=32;last_elapsed=time.perf_counter()-started
                    last_decision="accepted" if accepted else "rejected_kept_previous"
                    report=dict(episode=ep,updates=32,before=baseline,after=candidate,accepted=accepted,seconds=last_elapsed)
                    atomic_json(attempt/"report.json",report)
                    next_pointer={**current,"processed":sorted(set(current["processed"])|{ep}),"transactions":number,"last_report":str(attempt/"report.json")}
                    if accepted:
                        next_pointer.update(checkpoint=str(attempt/"latest.pkl"),actor=str(attempt/"actor.pkl"),
                            global_step=int(learner.state.global_step),actor_version=int(learner.state.actor_version),previous=previous)
                        baseline=candidate
                    # Commit marker is authoritative; restart republishes from it after any interrupted publication.
                    atomic_json(pointer,next_pointer);current=next_pointer
                    publish(Path(current["actor"]),actor_path)
                    if not accepted:
                        publish(Path(current["checkpoint"]),work/"checkpoints/latest.pkl")
                        learner._state=learner._load_latest_checkpoint()
                    print("ONLINE_UPDATE "+json.dumps(report),flush=True)
                if not stop:
                    last_request=rid;status("ready",rid)
                if args.once:break
            time.sleep(.1)
    except Exception as error:
        status("error",last_request,error=str(error));raise
    finally:
        if stop:status("stopped",last_request)
if __name__=="__main__":main()
