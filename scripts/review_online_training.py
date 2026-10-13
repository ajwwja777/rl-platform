#!/usr/bin/env python3
"""CPU checkpoint audit on explicitly declared TRAIN/reused DEV, never a release gate.

Returns use raw control step_id, not overlapping-window rank. Assisted observed
returns retain recorded rewards; they are not autonomous policy values. Temporal
slices are row positions, never semantic task stages. Q1 and min-Q stay distinct.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
import argparse, dataclasses, hashlib, json, pickle, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.openpi_rlt.experiments.review_metrics import review_metadata, episode_mean, auc

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    result=[]
    with Path(p).open("rb") as f:
        while True:
            try: result.append(pickle.load(f))
            except EOFError: break
    return result

def main():
    ap=argparse.ArgumentParser()
    for name in ["journal","norm-stats","upstream-src","output"]: ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--checkpoint",action="append",required=True)
    ap.add_argument("--split-label",choices=["TRAIN_REUSED","DEV_REUSED"],required=True)
    a=ap.parse_args()
    if a.output.exists(): raise ValueError("Fresh output required")
    sys.path[:0]=[str(a.upstream_src),str(a.upstream_src.parents[1]/"packages/openpi-client/src")]
    import jax, jax.numpy as jnp
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.trainer import _make_networks, _tree_to_jax
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.experiments.held_gripper import HeldGripperCritic
    assert all(d.platform=="cpu" for d in jax.devices())
    rows=read(a.journal)
    if not rows: raise ValueError("Empty journal")
    ck=pickle.loads(Path(a.checkpoint[0].split("=",1)[1]).read_bytes())
    cfg=RLTOnlineRLConfig(**dict(ck["rl_config"],action_norm_stats_path=str(a.norm_stats)))
    ad=ActionRepresentationAdapter.from_config(cfg)
    actor, critic=_make_networks(cfg)
    critic=HeldGripperCritic(critic,float(ad.stats.q01[6]),float(ad.stats.q99[6]))
    fields=["z_rl","proprio","ref_chunk","action_chunk","rewards","done","next_z_rl","next_proprio","next_ref_chunk","source_chunk"]
    raw={k:np.stack([r[k] for r in rows]) for k in fields}
    nb=ad.prepare_training_batch(raw)
    bb={k:jnp.asarray(v) for k,v in nb.items()}
    groups, observed, valid, human, outcome, kind=review_metadata(rows,cfg.gamma)
    auto=[ep for ep,ix in groups.items() if ep[1]>=0 and not human[ix].any() and valid[ix].all() and len(ix)>=6]
    @jax.jit
    def forward(ap_,cp_):
        mu=actor.sample_action(ap_,jax.random.PRNGKey(0),bb["z_rl"],bb["proprio"],bb["ref_chunk"],deterministic=True)
        def q(act):
            q1,q2=critic.q_values(cp_,bb["z_rl"],bb["proprio"],act)
            return jnp.stack([q1,q2,jnp.minimum(q1,q2)],-1)
        return mu,q(bb["action_chunk"]),q(bb["ref_chunk"]),q(mu)
    report=dict(schema=2,journal=str(a.journal),journal_sha256=sha(a.journal),norm_sha256=sha(a.norm_stats),split=a.split_label,rows=len(rows),episodes=len(groups),valid_return_rows=int(valid.sum()),q_columns=["Q1","Q2","minQ"],aggregation="Episode-equal; temporal slices last3/previous3 refer to overlapping rows, not semantic stages",return_definition="sum discounted recorded sparse rewards by actual control step distance; includes future human help",checkpoints={},limitations=["Repeated data, no independent TEST", "Recorded return is not counterfactual autonomous value", "No formal success inference from Q>1 or AUC", "Fixed gripper on all compared actions", "Physical deltas are absolute-target differences; do not sum over chunks"])
    # Quantify historical reviewer error without replacing source labels or Replay.
    legacy=np.zeros(len(rows))
    for ep,ix in groups.items():
        h=human[ix].any(1); first=int(np.argmax(h)) if h.any() else len(ix)
        success=any(bool(rows[i]["done"]) and np.sum(rows[i]["rewards"])>0 for i in ix)
        for rank,i in enumerate(ix):legacy[i]=float(success and not (ep[1]>=0 and h.any() and rank<first))*cfg.gamma**(cfg.chunk_len*(len(ix)-1-rank))
    report["legacy_return_error"]=dict(mean_abs_valid=float(np.abs(legacy[valid]-observed[valid]).mean()),max_abs_valid=float(np.abs(legacy[valid]-observed[valid]).max()),pre_hil_error=episode_mean(np.abs(legacy-observed),valid&(kind=="pre_hil"),groups))
    base=None
    for spec in a.checkpoint:
        name,path=spec.split("=",1); ck=pickle.loads(Path(path).read_bytes())
        other=RLTOnlineRLConfig(**dict(ck["rl_config"],action_norm_stats_path=str(a.norm_stats)))
        for k in ["action_representation","chunk_len","action_dim","gamma"]:
            if getattr(cfg,k)!=getattr(other,k):raise ValueError("Incompatible checkpoint: "+k)
        st=ck["state"];mu,qd,qr,qa=[np.asarray(v) for v in forward(_tree_to_jax(st["actor_params"]),_tree_to_jax(st["critic_params"]))]
        physical=ad.denormalize_to_abs_chunk(mu,raw["proprio"])
        if base is None:base=physical
        d=(physical[...,:6]-raw["ref_chunk"][...,:6])*1000
        result=dict(path=path,sha256=sha(path),global_step=int(st["global_step"]),actor_version=int(st["actor_version"]),by_class={})
        for cls in sorted(set(kind)):
            mask=kind==cls
            result["by_class"][cls]=dict(episodes=sum(mask[ix].any() for ix in groups.values()),q_record=episode_mean(qd,mask,groups),bias_vs_observed=episode_mean(qd-observed[:,None],mask&valid,groups),paired_record_minus_actor=episode_mean(qd-qa,mask,groups))
        result["autonomous_episode_auc"]={}
        for label,sl in [("first3_rows",slice(None,3)),("previous3_rows",slice(-6,-3)),("last3_rows",slice(-3,None))]:
            result["autonomous_episode_auc"][label]=[auc([qd[groups[ep][sl],q].mean() for ep in auto],[int(outcome[groups[ep][0]]=="autonomous_success") for ep in auto]) for q in range(3)]
        mask=np.zeros(len(rows),bool)
        for ep in auto:
            if outcome[groups[ep][0]]=="failure":mask[groups[ep][-3:]]=True
        result["failure_last3_actor_minus_ref_joint_mrad"]=episode_mean(d.mean(1),mask,groups)
        result["actor_change_joint_mae_mrad"]=episode_mean(np.abs(physical[...,:6]-base[...,:6]).mean(1)*1000,np.ones(len(rows),bool),groups)
        # Actual HUMAN/MIXED control steps only, six moving joints only.
        per=[]
        for ep,ix in groups.items():
            h=human[ix]
            if ep[1]>=0 and h.any():per.append((physical[ix][...,:6]-raw["action_chunk"][ix][...,:6])[h])
        result["human_mae_mrad"]=np.mean([np.abs(x).mean(0)*1000 for x in per],0).tolist() if per else None
        report["checkpoints"][name]=result
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda x:int(x)),encoding="utf-8")
    print(json.dumps(dict(output=str(a.output),episodes=len(groups),legacy_return_error=report["legacy_return_error"])))
if __name__=="__main__":main()
