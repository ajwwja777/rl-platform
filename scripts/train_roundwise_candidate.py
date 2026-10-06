#!/usr/bin/env python3
"""Closed-round CPU candidate training; never drives robots or publishes weights.

Original replay is read-only. Outputs must be in a fresh candidates directory.
Critic resets each round; Actor starts from supplied weights with fresh Adam.
The resumable round state is distinct from upstream latest.pkl.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/openpi-rlt/rlt_online_rl/src")]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_journal(path):
    rows = []
    with Path(path).open("rb") as f:
        while True:
            try: rows.append(pickle.load(f))
            except EOFError: return rows


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["journal", "init-checkpoint", "norm-stats", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, default=ROOT / "third_party/openpi-rlt")
    p.add_argument("--development-journal", type=Path)
    p.add_argument("--rounds", type=Path, help="JSON mapping canonical Episode key to explicit round ID")
    p.add_argument("--round-id", default="offline-round-1")
    p.add_argument("--critic-steps", type=int, default=1000)
    p.add_argument("--actor-steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heldout-fraction", type=float, default=.2)
    p.add_argument("--tail-steps", type=int, default=60)
    p.add_argument("--legacy-expert-id-base", type=int, help="explicit archived materialize_warmup_experts.py ID convention; not production negative-ID rule")
    p.add_argument("--delta-weight", type=float, default=1.)
    p.add_argument("--failure-anchor", type=float, default=.1)
    p.add_argument("--actor-target", choices=["reference_bc", "outcome_bc"], default="outcome_bc")
    p.add_argument("--critic-target", choices=["guarded_td", "mc"], default="guarded_td")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.critic_steps, a.actor_steps) < 0 or min(a.eval_every, a.batch) <= 0:
        p.error("invalid update budget")
    out = a.output.resolve()
    if not a.output.is_absolute() or "candidates" not in out.parts:
        p.error("absolute independent candidates output required")
    inputs = [a.journal, a.init_checkpoint, a.norm_stats] + ([a.development_journal] if a.development_journal else []) + ([a.rounds] if a.rounds else [])
    for source in inputs:
        source = source.resolve()
        if source == out or source in out.parents or out in source.parents:
            p.error("output overlaps a source asset")
    if out.exists() and not a.resume:
        p.error("fresh output required; --resume explicitly restores an existing round")
    out.mkdir(parents=True, exist_ok=a.resume)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, allowed[-4:])
    for relative in ["rlt_online_rl/src", "packages/openpi-client/src"]:
        dependency = a.upstream_root.resolve() / relative
        if not dependency.is_dir(): raise ValueError("missing pinned upstream source: " + str(dependency))
        sys.path.insert(0, str(dependency))
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.experiments.roundwise import annotate, RoundSampler, SamplingProfile, derive_batch, episode_key
    from methods.openpi_rlt.experiments.round_learning import make_critic_step, make_actor_step

    identities = {str(path.resolve()): sha(path) for path in inputs}
    checkpoint = pickle.loads(a.init_checkpoint.read_bytes())
    cfg_dict = dict(checkpoint["rl_config"])
    cfg_dict["action_norm_stats_path"] = str(a.norm_stats.resolve())
    cfg = RLTOnlineRLConfig(**cfg_dict)
    adapter = ActionRepresentationAdapter.from_config(cfg)
    if adapter is None: raise ValueError("explicit action normalization is required")
    rows = read_journal(a.journal)
    n = len(rows)
    external = read_journal(a.development_journal) if a.development_journal else []
    if set(episode_key(r) for r in rows) & set(episode_key(r) for r in external):
        raise ValueError("external development Episode overlaps training source")
    rows += external
    rounds = json.loads(a.rounds.read_text()) if a.rounds else {}
    if set(rounds) - set(episode_key(r) for r in rows):
        raise ValueError("round manifest contains unknown Episode identity")
    metadata = annotate(rows, gamma=cfg.gamma, tail_steps=a.tail_steps,
                        heldout_fraction=a.heldout_fraction, rounds=rounds, legacy_expert_id_base=a.legacy_expert_id_base)
    for m in metadata[n:]: m["split"] = "external_development"
    keys = ("z_rl", "proprio", "ref_chunk", "action_chunk", "rewards", "done",
            "next_z_rl", "next_proprio", "next_ref_chunk", "source_chunk")
    raw = {k: np.stack([r[k] for r in rows]) for k in keys}
    normalized = adapter.prepare_training_batch(raw)
    profile = SamplingProfile(recent_round=a.round_id if a.rounds else None)
    sampler = RoundSampler(metadata, profile=profile)
    settings = {k: v for k, v in vars(a).items() if k not in ["resume", "critic_steps", "actor_steps"] and not isinstance(v, Path)}
    settings.update(method="roundwise-td-outcome-bc-v1", source_sha256=identities,
        code_sha256={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__), ROOT/"methods/openpi_rlt/experiments/roundwise.py", ROOT/"methods/openpi_rlt/experiments/round_learning.py"]},
        upstream_root=str(a.upstream_root.resolve()),
        config=cfg_dict, target_q_policy="initial Actor frozen during Critic phase",
        actor_q_gradient=False, actor_advantage_weighting=False, critic_heads=2,
        human_reference_replacement=False, reason_no_reference_replacement="requires aligned next-state human action identity",
        sampling=dataclasses.asdict(profile), bootstrap_units="logical steps", optimizer="fresh Adam per round",
        independent_test_available=False, inherited_actor_training_seen="historical source; development is not certified unseen to initial Actor")
    metadata_path = out / "metadata.json"
    if a.resume:
        if json.loads(metadata_path.read_text()) != json.loads(json.dumps(settings)):
            raise ValueError("resume source/config/split mismatch")
    else:
        write_json(metadata_path, settings); write_json(out / "row_metadata.json", metadata)
    state, actor, critic = trainer.init_train_state(cfg, rng=jax.random.PRNGKey(a.seed))
    initial_actor = trainer._tree_to_jax(checkpoint["state"]["actor_params"])
    state = state.replace(actor_params=initial_actor, target_actor_params=initial_actor,
        actor_opt_state=state.actor_tx.init(initial_actor),
        global_step=jnp.asarray(checkpoint["state"]["global_step"]),
        actor_version=jnp.asarray(checkpoint["state"]["actor_version"]))
    rng = np.random.default_rng(a.seed)
    progress = {"critic_updates": 0, "actor_updates": 0}
    history = []
    saved = out / "round_state.pkl"
    state_fields = ("actor_params", "target_actor_params", "critic_params", "target_critic_params",
                    "actor_opt_state", "critic_opt_state", "rng", "global_step", "actor_version")
    if a.resume:
        payload = pickle.loads(saved.read_bytes())
        if payload["settings"] != settings: raise ValueError("round state identity mismatch")
        state = trainer.RLTTrainState(**{k: trainer._tree_to_jax(v) for k,v in payload["state"].items()},
                                     actor_tx=optax.adam(cfg.actor_lr), critic_tx=optax.adam(cfg.critic_lr))
        progress, history = payload["progress"], payload["history"]
        rng.bit_generator.state = payload["sampler_rng"]
        sampler.draws = payload["sampler_draws"]
        sampler.pool_draws, sampler.total, sampler.recent = payload["pool_draws"], payload["sample_total"], payload["sample_recent"]
    if progress["actor_updates"] and a.critic_steps > progress["critic_updates"]:
        raise ValueError("cannot extend Critic phase after Actor updates; start a new round")
    if a.critic_steps < progress["critic_updates"] or a.actor_steps < progress["actor_updates"]:
        raise ValueError("resume budget smaller than completed updates")
    critic_step = make_critic_step(actor, critic, cfg, mc=a.critic_target == "mc")
    actor_step = make_actor_step(actor, cfg, jnp.asarray(adapter.stats.q01), jnp.asarray(adapter.stats.q99),
        successful_executed=a.actor_target == "outcome_bc", failure_anchor=a.failure_anchor, delta_weight=a.delta_weight)
    Z,P,REF = [jnp.asarray(normalized[k]) for k in ("z_rl","proprio","ref_chunk")]
    @jax.jit
    def predict(params): return actor.actor_mean(params,Z,P,REF)
    @jax.jit
    def values(params, actions):
        q1,q2 = critic.q_values(params,Z,P,actions)
        return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)
    groups = collections.defaultdict(list)
    for i,m in enumerate(metadata):
        if "split" in m: groups[(m["split"],m["episode_key"])].append(i)
    def evaluate():
        prediction = predict(state.actor_params)
        physical = np.asarray(adapter.denormalize_to_abs_chunk(np.asarray(prediction), raw["proprio"]))
        q_exec = np.asarray(values(state.critic_params,jnp.asarray(normalized["action_chunk"])))
        q_actor = np.asarray(values(state.critic_params,prediction))
        q_ref = np.asarray(values(state.critic_params,REF))
        records = []
        for (split,key),ids in sorted(groups.items()):
            slots = {}
            for i in sorted(ids,key=lambda i:int(rows[i]["step_id"])):
                for slot in range(cfg.chunk_len): slots[int(rows[i]["step_id"])+slot]=(i,slot)
            index = np.array([v[0] for v in slots.values()]); slot = np.array([v[1] for v in slots.values()])
            human = np.isin(raw["source_chunk"][index,slot],[2,3])
            autonomous_success = metadata[ids[0]]["outcome"] == "auto_success"
            selected = human if human.any() else np.ones(len(index),bool) if autonomous_success else np.zeros(len(index),bool)
            error = np.abs(physical[index,slot]-raw["action_chunk"][index,slot])
            ref_error = np.abs(raw["ref_chunk"][index,slot]-raw["action_chunk"][index,slot])
            timed = [i for i in ids if metadata[i]["logical_end"]+1-60 <= metadata[i]["logical_start"] < metadata[i]["logical_end"]+1-30]
            rec = dict(episode_key=key,split=split,outcome=metadata[ids[0]]["outcome"],unique_logical_steps=len(slots),
                human_steps=int(human.sum()),mae_per_dim=error[selected].mean(0).tolist() if selected.any() else None,
                reference_mae_per_dim=ref_error[selected].mean(0).tolist() if selected.any() else None,
                actor_reference_rms_per_dim=np.sqrt(((physical[index,slot]-raw["ref_chunk"][index,slot])**2).mean(0)).tolist(),
                q1_actor_mean=float(q_actor[ids,0].mean()),qmin_recorded_mean=float(q_exec[ids,2].mean()),
                q1_recorded_minus_actor=float((q_exec[ids,0]-q_actor[ids,0]).mean()),
                q1_recorded_minus_reference=float((q_exec[ids,0]-q_ref[ids,0]).mean()),
                q_bias_autonomy_mc=float(np.mean(q_exec[ids,2]-np.array([metadata[i]["mc_return"] for i in ids]))),
                raw_q_above_one=float((q_exec[ids,:2]>1).mean()),
                preterminal_time_proxy_q=float(q_exec[timed,2].mean()) if timed else None)
            records.append(rec)
        summary = {}
        for split in {m["split"] for m in metadata if "split" in m}:
            subset=[r for r in records if r["split"]==split]
            targets=[r["mae_per_dim"] for r in subset if r["mae_per_dim"] is not None]
            scores={r["outcome"]:[] for r in subset}
            for r in subset:
                if r["preterminal_time_proxy_q"] is not None: scores[r["outcome"]].append(r["preterminal_time_proxy_q"])
            pos,neg=scores.get("auto_success",[]),scores.get("failure",[])
            auc=float(np.mean((np.array(pos)[:,None]>np.array(neg)[None,:])+ .5*(np.array(pos)[:,None]==np.array(neg)[None,:]))) if pos and neg else None
            summary[split]=dict(episodes=len(subset),supervised_episodes=len(targets),
                mae_per_dim=np.mean(targets,axis=0).tolist() if targets else None,
                autonomous_episode_preterminal_auc=auc,auc_positive_episodes=len(pos),auc_negative_episodes=len(neg),
                mean_episode_q_bias=float(np.mean([r["q_bias_autonomy_mc"] for r in subset])),
                mean_episode_raw_q_above_one=float(np.mean([r["raw_q_above_one"] for r in subset])))
        return dict(progress=dict(progress),summary=summary,episodes=records)
    def save(complete=False):
        payload=dict(state={k:trainer._tree_to_numpy(getattr(state,k)) for k in state_fields},settings=settings,
            progress=dict(progress),history=history,sampler_rng=rng.bit_generator.state,sampler_draws=sampler.draws,
            pool_draws=sampler.pool_draws,sample_total=sampler.total,sample_recent=sampler.recent)
        tmp=saved.with_suffix(".tmp")
        with tmp.open("wb") as f: pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
        tmp.replace(saved)
        write_json(out/"report.json",dict(settings=settings,progress=progress,history=history,
            sampler=sampler.receipt(),complete=complete,release_status="evidence_insufficient",
            limitations=["No independent test or new real robot task result", "HIL assets contain historical feedback, not proven optimal commands",
                "Preterminal windows do not certify semantic task phase or outcome concealment",
                "No Q gradient or Q-derived BC advantage before action-value acceptance",
                "20 Hz logical training does not validate 50 Hz RTC/EMA execution"],
            dataset=dict(rows=len(rows),external_rows=len(external),
                exclusions=dict(collections.Counter(m["reason"] for m in metadata)),
                outcome_episode_counts=dict(collections.Counter(m["outcome"] for key,ids in
                    groups.items() for m in [metadata[ids[0]]])),
                uuid_missing_rows=sum(not m.get("uuid_verified",False) for m in metadata)),
            source_unchanged={str(path.resolve()):sha(path)==identities[str(path.resolve())] for path in inputs}))
    if not history:
        history.append(dict(phase="initial",evaluation=evaluate()));save()
    for phase,budget,step_fn in [("critic",a.critic_steps,critic_step),("actor",a.actor_steps,actor_step)]:
        counter=phase+"_updates"
        for step in range(progress[counter]+1,budget+1):
            idx=sampler.sample(rng,a.batch)
            batch={k:jnp.asarray(v) for k,v in derive_batch(normalized,metadata,idx).items()}
            state,metrics=step_fn(state,batch);progress[counter]=step
            if step%a.eval_every==0 or step==budget:
                metric={k:float(v) for k,v in metrics.items()}
                if not all(np.isfinite(list(metric.values()))): raise FloatingPointError("nonfinite learning metrics")
                history.append(dict(phase=phase,metrics=metric,evaluation=evaluate(),sample_index_sha256=hashlib.sha256(idx.tobytes()).hexdigest()))
                save();print(json.dumps(dict(phase=phase,step=step,metrics=metric,sampling=sampler.receipt())),flush=True)
    snapshot=out/"actor_snapshot";snapshot.mkdir(exist_ok=True)
    with (snapshot/"actor_snapshot.pkl").open("wb") as f:
        pickle.dump(dict(actor_params=trainer._tree_to_numpy(state.actor_params),version=int(state.actor_version),
                        global_step=int(state.global_step),rl_config=cfg_dict),f,pickle.HIGHEST_PROTOCOL)
    import shutil
    shutil.copyfile(a.norm_stats,out/"action_norm_stats.json")
    write_json(out/"publication.json",dict(publication_policy="staged",candidate_snapshot=str(snapshot/"actor_snapshot.pkl"),
        source_checkpoint=str(a.init_checkpoint.resolve()),release_authorized=False,
        training_method=settings["method"],automatic_live_publication=False))
    save(True)
    print(json.dumps(dict(output=str(out),progress=progress,release_status="evidence_insufficient")),flush=True)


if __name__ == "__main__": main()
