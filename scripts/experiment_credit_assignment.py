#!/usr/bin/env python3
"""Registered credit/sampling ablations on an immutable recorded Replay snapshot."""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"third_party/openpi-rlt/rlt_online_rl/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np
from diagnose_online_learning import auc, episode_split
from integrations.cobot_runtime.replay_audit import metadata,annotate,atomic_json,composition
from methods.openpi_rlt.experiments.credit import Profile,episode_credit,sample_indices,make_train_step

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/rlt/plug_v3_yyshadow/online_rl.yaml")
    p.add_argument("--registry",type=Path,default=ROOT/"configs/experiments/credit_ablation.json")
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--updates",type=int,default=2000)
    p.add_argument("--seeds",default="41,42,43")
    p.add_argument("--save-candidates",type=Path)
    args=p.parse_args()
    import yaml,jax,jax.numpy as jnp,optax
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl import trainer
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    config=yaml.safe_load(args.config.read_text())
    cfg=RLTOnlineRLConfig(**config["experiment"]["rl"])
    adapter=ActionRepresentationAdapter.from_config(cfg)
    journal=Path(config["runtime"]["replay"]["journal_path"])
    size=journal.stat().st_size; rows=[]
    with journal.open("rb") as f:
        while f.tell()<size:rows.append(pickle.load(f))
    if journal.stat().st_size!=size:raise RuntimeError("Replay changed; rerun with a stable snapshot")
    meta=annotate([metadata(r) for r in rows])
    cohort=json.loads(args.cohort.read_text())
    train,val,held=episode_split(meta,held_episodes=cohort["holdout_online_episodes"])
    if len(val)!=cohort["holdout_transitions"]:raise ValueError("Cohort changed")
    raw={k:np.stack([r[k] for r in rows]) for k in rows[0] if k!="collection_phase"}
    batch=adapter.prepare_training_batch(raw)
    returns,valid=episode_credit(rows,cfg.gamma)
    batch.update(mc_return=returns,mc_valid=valid)
    b={k:jnp.asarray(v) for k,v in batch.items()}
    model=Path(config["runtime"]["learner_service"]["checkpoint_dir"]).parent.parent
    initial_path=model/"warmup_5000/checkpoints/latest.pkl"
    initial_payload=pickle.load(initial_path.open("rb"))
    initial=trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in initial_payload["state"].items()},
        actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))
    if int(initial.global_step)!=5000:raise ValueError("Expected original warmup5000")
    actor,critic=trainer._make_networks(cfg)
    q01,q99=jnp.asarray(adapter.stats.q01),jnp.asarray(adapter.stats.q99)

    @jax.jit
    def predictions(ap,cp,z,proprio,ref,executed):
        pred=actor.sample_action(ap,jax.random.PRNGKey(0),z,proprio,ref,deterministic=True)
        q1,q2=critic.q_values(cp,z,proprio,executed)
        return pred,jnp.minimum(q1,q2)

    def evaluate(state, include_traces=False):
        pred,q=predictions(state.actor_params,state.critic_params,b["z_rl"][val],
            b["proprio"][val],b["ref_chunk"][val],b["action_chunk"][val])
        pred,q=np.asarray(pred),np.asarray(q)
        native=adapter.denormalize_to_abs_chunk(pred,raw["proprio"][val])
        ref=raw["ref_chunk"][val];actual=raw["action_chunk"][val]
        human=np.isin(raw["source_chunk"][val],[2,3])
        error=np.abs(native[...,:6]-actual[...,:6])
        ref_error=np.abs(ref[...,:6]-actual[...,:6])
        correction=native[...,:6]-ref[...,:6]
        human_direction=actual[...,:6]-ref[...,:6]
        norm=np.linalg.norm(correction,axis=-1)*np.linalg.norm(human_direction,axis=-1)
        eligible=human&(norm>1e-10)
        alignment=np.sum(correction*human_direction,axis=-1)/np.maximum(norm,1e-10)
        groups={}
        for j,i in enumerate(val):
            groups.setdefault((meta[i]["phase"],meta[i]["episode_id"]),[]).append((j,i))
        portions={}
        for portion in ["all","early","middle","late"]:
            scores,labels=[],[]
            for members in groups.values():
                js=[j for j,i in members if portion=="all" or meta[i]["portion"]==portion]
                if not js:continue
                scores.append(float(q[js].mean()))
                labels.append(int(meta[members[0][1]]["outcome"]=="success"))
            portions[portion]={"auc":auc(scores,labels),"episodes":len(scores)}
        result={"human_mae_rad":float(error[human].mean()),
            "reference_human_mae_rad":float(ref_error[human].mean()),
            "hil_steps_improved_ratio":float((error.mean(-1)<ref_error.mean(-1))[human].mean()),
            "correction_human_cosine":float(alignment[eligible].mean()),
            "actor_reference_rms_rad":float(np.sqrt(np.mean(correction**2))),
            "step_delta_p95_rad":float(np.quantile(np.abs(np.diff(native[...,:6],axis=1)),.95)),
            "q_by_portion":portions,"q_above_one_ratio":float((q>1).mean()),
            "mc_coverage":float(valid[val].mean()),
            "mc_rmse":float(np.sqrt(np.mean((q[valid[val]]-returns[val][valid[val]])**2)))}
        if include_traces:
            # One human-controlled window per episode, readable at joint level.
            traces=[]
            for members in groups.values():
                match=next(((j,i) for j,i in members if human[j].any()),None)
                if match is None:continue
                j,i=match
                traces.append({"episode":int(meta[i]["episode_id"]),"step":int(meta[i]["step_id"]),
                    "outcome":meta[i]["outcome"],"human_mask":human[j].tolist(),
                    "reference":ref[j].tolist(),"actor":native[j].tolist(),
                    "executed":actual[j].tolist()})
            result["traces"]=traces
        return result

    registry=json.loads(args.registry.read_text())
    profiles={name:Profile(**options) for name,options in registry["profiles"].items()}
    report={"schema":1,"generated_at":time.time(),"updates":args.updates,
        "source_journal":str(journal),"source_sha256":hashlib.sha256(journal.read_bytes()).hexdigest(),
        "initial_checkpoint":str(initial_path),"initial_sha256":hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "holdout_online_episodes":[list(k) for k in held],"holdout_transitions":len(val),
        "train_transitions":len(train),"mc_valid_transitions":int(valid.sum()),
        "validation_semantics":"Fixed 19-episode development cohort, repeatedly examined. Not an untouched test set or robot success rate.",
        "registry":registry,"algorithm_config":config["experiment"]["rl"],
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "baseline":evaluate(initial,True),"experiments":[]}
    atomic_json(args.output,report)
    for name,profile in profiles.items():
        update=make_train_step(profile.mc_weight)
        for seed in map(int,args.seeds.split(",")):
            state=initial.replace(rng=jax.random.PRNGKey(seed));rng=np.random.default_rng(seed)
            counts=np.zeros(len(rows),np.int64);started=time.time()
            for step in range(args.updates):
                ids=sample_indices(rng,train,meta,profile,128)
                np.add.at(counts,ids,1)
                state,metrics=update(state,{k:v[ids] for k,v in b.items()},actor=actor,critic=critic,
                    rl_config=cfg,bc_weight=cfg.online_bc_weight,q_weight=cfg.online_q_weight,
                    delta_weight=cfg.delta_weight,use_action_adapter=True,action_q01=q01,action_q99=q99)
            result={"variant":name,"seed":seed,"elapsed_sec":time.time()-started,
                "learner_step":int(state.global_step),"actor_version":int(state.actor_version),
                "sample_outcome":{label:float(sum(counts[i] for i,r in enumerate(meta) if r["outcome"]==label)/counts.sum())
                                  for label in ["success","failure","unknown"]},
                "sample_portion":{part:float(sum(counts[i] for i,r in enumerate(meta) if r["portion"]==part)/counts.sum())
                                  for part in ["early","middle","late"]},
                "unique_sampled":int(np.count_nonzero(counts)),
                "validation":evaluate(state,seed==42)}
            if args.save_candidates and seed==42:
                # Research snapshots only. No production latest pointer is touched.
                folder=args.save_candidates/name
                folder.mkdir(parents=True,exist_ok=True)
                fields=initial_payload["state"].keys()
                payload={"state":{k:trainer._tree_to_numpy(getattr(state,k)) for k in fields},
                         "research_only":True,"profile":registry["profiles"][name],
                         "source_sha256":report["source_sha256"],"excluded_episodes":report["holdout_online_episodes"]}
                path=folder/"research_state.pkl"
                with path.open("wb") as f:pickle.dump(payload,f)
                result["research_state"]=str(path)
            report["experiments"].append(result)
            atomic_json(args.output,report)
            print(json.dumps({k:v for k,v in result.items() if k!="validation"}),flush=True)
    report["finished_at"]=time.time();atomic_json(args.output,report)

if __name__=="__main__":main()
