#!/usr/bin/env python3
"""Frozen candidate audit: episode uncertainty and local Critic action preference.

No training, RPC to model services, ROS imports, publishers or weight writes.
HIL similarity and Q preference are diagnostic proxies, not robot success.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"),
               str(ROOT / "third_party/openpi-rlt/rlt_online_rl/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np


def paired_interval(values, seed=42, draws=10000):
    """Bootstrap independent episodes, never treat overlapping windows as IID."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Need finite paired episode statistics")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return {"episodes": len(values), "mean": float(values.mean()),
            "bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist()}


def paired_auc_interval(base, candidate, labels, seed=42, draws=10000):
    """Paired stratified episode bootstrap for a diagnostic AUC difference."""
    base, candidate, labels = map(np.asarray, (base, candidate, labels))
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        raise ValueError("Both episode outcomes required")
    def score(a, p, n):
        d = a[p, None] - a[n]
        return ((d > 0) + .5 * (d == 0)).mean()
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(draws):
        p, n = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        changes.append(score(candidate, p, n) - score(base, p, n))
    return {"success_episodes": len(pos), "failure_episodes": len(neg),
            "mean_difference": float(score(candidate, pos, neg) - score(base, pos, neg)),
            "bootstrap_95_interval": np.quantile(changes, [.025, .975]).tolist()}


def check_idle(web_url=None):
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True)
    if gpu.returncode or gpu.stdout.strip():
        raise RuntimeError("GPU idle check failed or GPU is occupied; leave live work untouched")
    if web_url:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(web_url.rstrip("/") + "/api/deployment/status", timeout=15) as r:
            status = json.load(r)
        if status.get("phase") != "offline" or status.get("session_active") or status.get("active"):
            raise RuntimeError("Web model/session is active; postpone offline GPU audit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rlt/plug_v3_yyshadow/online_rl.yaml")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-web-url")
    args = parser.parse_args()
    check_idle(args.guard_web_url)
    import yaml
    import jax
    import jax.numpy as jnp
    from rlt_online_rl import trainer
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from integrations.cobot_runtime.replay_audit import metadata, annotate, atomic_json
    from diagnose_online_learning import episode_split, auc
    from methods.openpi_rlt.experiments.credit import episode_credit

    config = yaml.safe_load(args.config.read_text())
    cfg = RLTOnlineRLConfig(**config["experiment"]["rl"])
    adapter = ActionRepresentationAdapter.from_config(cfg)
    journal = Path(config["runtime"]["replay"]["journal_path"])
    source = journal.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    cohort = json.loads(args.cohort.read_text())
    if digest != cohort["source_sha256"]:
        raise ValueError("Replay differs from the experiment snapshot; do not mix cohorts")
    import io
    stream = io.BytesIO(source)
    rows = []
    while stream.tell() < len(source):
        rows.append(pickle.load(stream))
    meta = annotate([metadata(r) for r in rows])
    _, val, held = episode_split(meta, held_episodes=cohort["holdout_online_episodes"])
    if len(val) != cohort["holdout_transitions"]:
        raise ValueError("Holdout content changed")
    raw = {k: np.stack([r[k] for r in rows]) for k in rows[0] if k != "collection_phase"}
    batch = adapter.prepare_training_batch(raw)
    b = {k: jnp.asarray(v[val]) for k, v in batch.items()}
    human = np.isin(raw["source_chunk"][val], [2, 3])
    hil_ids = np.flatnonzero(human.any(1))
    if not len(hil_ids):
        raise ValueError("No HIL windows in cohort")
    actor, critic = trainer._make_networks(cfg)
    groups = defaultdict(list)
    for j, i in enumerate(val):
        groups[(meta[i]["phase"], meta[i]["episode_id"])].append(j)
    returns, valid = episode_credit(rows, cfg.gamma)

    @jax.jit
    def predict(ap):
        return actor.sample_action(ap, jax.random.PRNGKey(0), b["z_rl"],
                                   b["proprio"], b["ref_chunk"], deterministic=True)

    @jax.jit
    def values(cp, actions):
        q1, q2 = critic.q_values(cp, b["z_rl"], b["proprio"], actions)
        return jnp.stack([q1, jnp.minimum(q1, q2)], -1)

    def load(path):
        data = path.read_bytes()
        state = pickle.loads(data)["state"]
        return trainer._tree_to_jax(state["actor_params"]), trainer._tree_to_jax(state["critic_params"]), {
            "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "learner_step": int(state["global_step"]), "actor_version": int(state["actor_version"])}

    baseline_ap, _, _ = load(args.states / "baseline/research_state.pkl")
    anchor = predict(baseline_ap)
    profiles = ["warmup5000", "baseline", "mc_10", "mc_30", "terminal_only", "tail_mix"]
    report = {"schema": 1, "generated_at": time.time(), "source_journal": str(journal),
        "source_sha256": digest, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "gamma": cfg.gamma, "holdout_online_episodes": [list(x) for x in held],
        "holdout_transitions": len(val), "variants": {},
        "semantics": "Seed42 frozen research states; repeatedly inspected development episodes, no untouched test. Q1 drives Actor, min(Q1,Q2) bootstraps TD. Human endpoint may be counterfactual/OOD and is not proven optimal.",
        "limitations": ["MC terminal outcome includes later human interventions, so early autonomous actions can inherit an assisted success.",
                        "AUC ranks recorded successful/failed episodes, not alternate actions in the same state.",
                        "Bootstrap intervals condition on this cohort and seed42; model selection and training-seed uncertainty are not included.",
                        "Within-window action differences do not measure replanning-boundary or hardware tracking jitter."]}
    for name in profiles:
        path = Path(cohort["initial_checkpoint"]) if name == "warmup5000" else args.states / name / "research_state.pkl"
        ap, cp, provenance = load(path)
        pred = predict(ap)
        native = adapter.denormalize_to_abs_chunk(np.asarray(pred), raw["proprio"][val])
        errors = np.abs(native - raw["action_chunk"][val])
        q_recorded = np.asarray(values(cp, b["action_chunk"]))
        q_actor = np.asarray(values(cp, pred))
        paths = {}
        for anchor_name, start in [("own_actor", pred), ("fixed_baseline_actor", anchor)]:
            sampled = []
            for alpha in [0., .25, .5, .75, 1.]:
                actions = start + alpha * jnp.where(jnp.asarray(human)[..., None], b["action_chunk"] - start, 0.)
                sampled.append(np.asarray(values(cp, actions)))
            curve = np.stack(sampled)
            delta = curve[-1] - curve[0]
            summaries = {}
            for q_index, q_name in enumerate(["q1_actor_objective", "min_q_td_objective"]):
                episode_delta = [float(delta[js, q_index][human[js].any(1)].mean())
                                 for js in groups.values() if human[js].any()]
                summaries[q_name] = {"hil_windows": len(hil_ids),
                    "mean_path": curve[:, hil_ids, q_index].mean(1).tolist(),
                    "human_endpoint_higher_window_ratio": float((delta[hil_ids, q_index] > 0).mean()),
                    "nondecreasing_window_ratio": float((np.diff(curve[:, hil_ids, q_index], axis=0) >= -1e-7).all(0).mean()),
                    "episode_equal_mean_q_change": paired_interval(episode_delta)}
            paths[anchor_name] = summaries
        episode_rows = []
        for key, js in groups.items():
            human_mask = human[js]
            i = val[js[0]]
            row = {"phase": key[0], "episode_id": key[1], "outcome": meta[i]["outcome"],
                   "episode_hil": bool(meta[i]["episode_hil"]), "windows": len(js),
                   "hil_steps": int(human_mask.sum()),
                   "human_joint_mae_rad": float(errors[js, :, :6][human_mask].mean()) if human_mask.any() else None,
                   "human_gripper_mae_m": float(errors[js, :, 6][human_mask].mean()) if human_mask.any() else None,
                   "q_recorded": {}}
            for portion in ["all", "early", "middle", "late"]:
                selected = [j for j in js if portion == "all" or meta[val[j]]["portion"] == portion]
                row["q_recorded"][portion] = float(q_recorded[selected, 1].mean()) if selected else None
            episode_rows.append(row)
        part_auc = {}
        for portion in ["all", "early", "middle", "late"]:
            scores = [e["q_recorded"][portion] for e in episode_rows]
            labels = [int(e["outcome"] == "success") for e in episode_rows]
            part_auc[portion] = auc(scores, labels)
        per_joint = errors[..., :6][human].mean(0).tolist()
        report["variants"][name] = {"provenance": provenance,
            "human_joint_mae_rad": float(np.mean(per_joint)), "per_joint_mae_rad": per_joint,
            "human_gripper_mae_m": float(errors[..., 6][human].mean()),
            "recorded_outcome_auc": part_auc, "q_paths": paths, "episodes": episode_rows}
        autonomous = [e for e in episode_rows if not e["episode_hil"]]
        report["variants"][name]["autonomous_episode_early_auc"] = {
            "episodes": len(autonomous), "success_episodes": sum(e["outcome"] == "success" for e in autonomous),
            "auc": auc([e["q_recorded"]["early"] for e in autonomous],
                       [int(e["outcome"] == "success") for e in autonomous])}
        print(json.dumps({"variant": name, "hil_mae": np.mean(per_joint), "early_auc": part_auc["early"]}), flush=True)
    base, candidate = (report["variants"][n]["episodes"] for n in ["baseline", "mc_30"])
    if [(x["phase"], x["episode_id"]) for x in base] != [(x["phase"], x["episode_id"]) for x in candidate]:
        raise ValueError("Unpaired episode order")
    report["mc30_vs_baseline"] = {
        "episode_equal_hil_mae_change_rad": paired_interval([
            y["human_joint_mae_rad"] - x["human_joint_mae_rad"] for x, y in zip(base, candidate) if x["hil_steps"]]),
        "early_auc_change": paired_auc_interval([x["q_recorded"]["early"] for x in base],
            [x["q_recorded"]["early"] for x in candidate], [int(x["outcome"] == "success") for x in base])}
    episode_lookup = {}
    for row in meta:
        episode_lookup[(row["phase"], row["episode_id"])] = (row["phase"], row["outcome"], row["episode_hil"])
    counts = defaultdict(int)
    for key in episode_lookup.values():
        counts[key] += 1
    report["replay_episode_strata"] = [dict(phase=k[0], outcome=k[1], episode_hil=bool(k[2]), count=v)
                                        for k, v in sorted(counts.items())]
    report["valid_mc_rows_in_assisted_success_episodes"] = int(sum(
        bool(valid[i]) and r["outcome"] == "success" and r["episode_hil"] for i, r in enumerate(meta)))
    first_human = {}
    for r in meta:
        if r["hil"]:
            key = (r["phase"], r["episode_id"])
            first_human[key] = min(first_human.get(key, r["step_id"]), r["step_id"])
    # Windows strictly ending before the first HIL-containing window. This
    # identifies pre-intervention credit, not proof those earlier actions failed.
    prefix_ids = [i for i, r in enumerate(meta) if r["outcome"] == "success" and r["episode_hil"]
                  and r["step_id"] + len(rows[i]["rewards"]) <= first_human[(r["phase"], r["episode_id"])]]
    report["pre_hil_prefix_credit"] = {
        "windows": len(prefix_ids), "positive_mc_windows": int(np.count_nonzero(returns[prefix_ids] > 0)),
        "semantics": "Complete windows before first recorded intervention inherit the eventual assisted outcome; this does not prove their actions were bad."}
    if hashlib.sha256(journal.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Replay changed during audit; refuse report")
    atomic_json(args.output, report)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

