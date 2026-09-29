#!/usr/bin/env python3
"""Offline RLT sampling ablation and archived Actor audit; no ROS, RPC or weight writes."""
import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/openpi-rlt/rlt_online_rl/src"),
               str(ROOT / "third_party/openpi-rlt/packages/openpi-client/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from integrations.cobot_runtime.replay_audit import metadata, annotate, build_report, atomic_json

def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    a, b = scores[labels == 1], scores[labels == 0]
    if not len(a) or not len(b): return None
    d = a[:, None] - b[None, :]
    return float(np.mean((d > 0) + .5 * (d == 0)))

def episode_split(meta, seed=42):
    """Split ONLINE episodes only. Warmup init has never trained on these online rows."""
    groups = {}
    for row in meta:
        if row["phase"] == "online" and row["outcome"] != "unknown":
            groups[(row["phase"], row["episode_id"])] = (row["outcome"], row["episode_hil"])
    strata = {}
    for key, label in groups.items(): strata.setdefault(label, []).append(key)
    rng = np.random.default_rng(seed)
    held = set()
    for values in strata.values():
        values = sorted(values)
        rng.shuffle(values)
        held.update(values[:max(1, round(len(values) * .25))] if len(values) > 1 else [])
    train = np.array([i for i,r in enumerate(meta) if (r["phase"],r["episode_id"]) not in held])
    val = np.array([i for i,r in enumerate(meta) if (r["phase"],r["episode_id"]) in held])
    return train, val, sorted(held)

def choose(rng, indices, success, ratio, size=128):
    if ratio is None: return rng.choice(indices, size, replace=True)
    pos, neg = indices[success[indices]], indices[~success[indices]]
    if not len(pos) or not len(neg): raise ValueError("Both outcome pools required")
    count = round(size * ratio)
    ids = np.r_[rng.choice(pos, count, replace=True), rng.choice(neg, size-count, replace=True)]
    rng.shuffle(ids)
    return ids

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"configs/rlt/plug_v3_yyshadow/online_rl.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--seeds", default="41,42,43")
    args = p.parse_args()
    import yaml, jax, jax.numpy as jnp, optax
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl import trainer
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    config = yaml.safe_load(args.config.read_text())
    journal = Path(config["runtime"]["replay"]["journal_path"])
    model = Path(config["runtime"]["learner_service"]["checkpoint_dir"]).parent.parent
    cfg = RLTOnlineRLConfig(**config["experiment"]["rl"])
    adapter = ActionRepresentationAdapter.from_config(cfg)
    if adapter is None: raise ValueError("Registered action normalization required")
    rows = []
    size = journal.stat().st_size
    with journal.open("rb") as f:
        while f.tell() < size: rows.append(pickle.load(f))
    if f.closed and sum(not isinstance(x,dict) for x in rows): raise ValueError("Invalid journal")
    meta = annotate([metadata(r) for r in rows])
    train, val, held = episode_split(meta)
    if not len(val): raise ValueError("No online episode holdout")
    numeric = [k for k in rows[0] if k != "collection_phase"]
    raw = {k: np.stack([r[k] for r in rows]) for k in numeric}
    batch = adapter.prepare_training_batch(raw)
    b = {k:jnp.asarray(v) for k,v in batch.items()}
    actor, critic = trainer._make_networks(cfg)
    success = np.array([r["outcome"] == "success" for r in meta])
    q01, q99 = jnp.asarray(adapter.stats.q01), jnp.asarray(adapter.stats.q99)

    def load_state(path):
        d = pickle.load(path.open("rb"))["state"]
        return trainer.RLTTrainState(**{k:trainer._tree_to_jax(v) for k,v in d.items()},
             actor_tx=optax.adam(cfg.actor_lr),critic_tx=optax.adam(cfg.critic_lr))

    @jax.jit
    def predict(params, z, proprio, ref):
        return actor.sample_action(params,jax.random.PRNGKey(0),z,proprio,ref,deterministic=True)

    @jax.jit
    def values(params,z,proprio,action):
        a,b=critic.q_values(params,z,proprio,action)
        return jnp.minimum(a,b)

    def evaluate(ap, cp, ids):
        pred = np.asarray(predict(ap,b["z_rl"][ids],b["proprio"][ids],b["ref_chunk"][ids]))
        native = adapter.denormalize_to_abs_chunk(pred,raw["proprio"][ids])
        hil = np.isin(raw["source_chunk"][ids],[2,3])
        error = np.abs(native[...,:6]-raw["action_chunk"][ids,...,:6])
        referror=np.abs(raw["ref_chunk"][ids,...,:6]-raw["action_chunk"][ids,...,:6])
        result={"transitions":len(ids), "hil_control_steps":int(hil.sum()),
            "human_mae_rad":float(error[hil].mean()) if hil.any() else None,
            "reference_human_mae_rad":float(referror[hil].mean()) if hil.any() else None,
            "actor_reference_rms_rad":float(np.sqrt(np.mean((native[...,:6]-raw["ref_chunk"][ids,...,:6])**2))),
            "step_delta_p95_rad":float(np.quantile(np.abs(np.diff(native[...,:6],axis=1)),.95))}
        if cp is not None:
            q=np.asarray(values(cp,b["z_rl"][ids],b["proprio"][ids],b["action_chunk"][ids]))
            groups={}
            for j,i in enumerate(ids):
                key=(meta[i]["phase"],meta[i]["episode_id"])
                groups.setdefault(key,[]).append((j,i))
            byportion={}
            for portion in ["all","early","middle","late"]:
                scores,labels=[],[]
                for members in groups.values():
                    selected=[j for j,i in members if portion=="all" or meta[i]["portion"]==portion]
                    if not selected:continue
                    label=meta[members[0][1]]["outcome"]
                    if label=="unknown":continue
                    scores.append(float(q[selected].mean()));labels.append(int(label=="success"))
                byportion[portion]=dict(auc=auc(scores,labels),episodes=len(scores),
                    success_q=float(np.mean(np.array(scores)[np.array(labels)==1])) if 1 in labels else None,
                    failure_q=float(np.mean(np.array(scores)[np.array(labels)==0])) if 0 in labels else None)
            result.update(q_by_portion=byportion,q_above_one_ratio=float(np.mean(q>1)))
        return result

    initpath=model/"warmup_5000/checkpoints/latest.pkl"
    initial=load_state(initpath)
    if int(initial.global_step)!=5000: raise ValueError("Expected the original 5000-step warmup")
    report={"schema":1,"generated_at":time.time(),"journal":str(journal),"journal_sha256":hashlib.sha256(journal.read_bytes()).hexdigest(),
        "warmup_checkpoint":str(initpath),"warmup_sha256":hashlib.sha256(initpath.read_bytes()).hexdigest(),
        "holdout_online_episodes":[list(x) for x in held],"train_transitions":len(train),"holdout_transitions":len(val),
        "split_seed":42,"updates":args.updates,"seeds":[int(s) for s in args.seeds.split(",")],
        "validation_semantics":"New online episode holdout for experiments starting at warmup5000. Archived online Actors may have seen these episodes; their scores are retrospective audits, NOT independent validation.",
        "baseline":evaluate(initial.actor_params,initial.critic_params,val),"versions":[],"experiments":[]}
    # All retained published versions, plus the current final flush snapshot if distinct.
    snapshots=list((model/"online/actor_snapshot/history").glob("actor_v*.pkl"))
    snapshots.append(model/"online/actor_snapshot/actor_snapshot.pkl")
    seen=set()
    for path in sorted(snapshots):
        d=pickle.load(path.open("rb"));version=int(d["version"])
        if version in seen:continue
        seen.add(version)
        row=dict(actor_version=version,learner_step=int(d.get("global_step",-1)),
            path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evaluation="training-seen retrospective audit")
        row.update(evaluate(trainer._tree_to_jax(d["actor_params"]),None,val));report["versions"].append(row)
    atomic_json(args.output, report)
    for seed in report["seeds"]:
        for name,ratio in [("uniform",None),("success_50",.5),("success_70",.7),("success_90",.9)]:
            state=initial.replace(rng=jax.random.PRNGKey(seed))
            rng=np.random.default_rng(seed)
            sampled=[]
            started=time.time()
            for step in range(args.updates):
                ids=choose(rng,train,success,ratio)
                sampled.append(float(success[ids].mean()))
                state,metrics=trainer.train_step(state,{k:v[ids] for k,v in b.items()},
                    actor=actor,critic=critic,rl_config=cfg,
                    bc_weight=cfg.online_bc_weight,q_weight=cfg.online_q_weight,delta_weight=cfg.delta_weight,
                    use_action_adapter=True,action_q01=q01,action_q99=q99)
            result=dict(variant=name,seed=seed,mean_sample_success=float(np.mean(sampled)),
                elapsed_sec=time.time()-started,validation=evaluate(state.actor_params,state.critic_params,val))
            report["experiments"].append(result)
            atomic_json(args.output,report)
            print(json.dumps(result),flush=True)
    report["finished_at"]=time.time()
    atomic_json(args.output,report)
    print(args.output,flush=True)

if __name__=="__main__":main()
