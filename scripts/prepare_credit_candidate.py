#!/usr/bin/env python3
"""Prepare separate candidate weights/config; no Replay writes or robot actions."""
import argparse,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--source-config",type=Path,default=ROOT/"configs/rlt/plug_v3_yyshadow/online_rl.yaml")
    a=p.parse_args()
    report=json.loads(a.report.read_text())
    if not report.get("finished_at"):raise ValueError("Complete ablation first")
    result=next(r for r in report["experiments"] if r["variant"]=="mc_30" and r["seed"]==42)
    source=Path(result["research_state"])
    if "candidates" not in source.parts:raise ValueError("Expected research candidate")
    cfg=yaml.safe_load(a.source_config.read_text())
    candidate=source.parent/"online_candidate"
    if candidate.exists():raise FileExistsError("Never overwrite continuing candidate training")
    data=pickle.load(source.open("rb"));state=data["state"];step=int(state["global_step"])
    journal=Path(cfg["runtime"]["replay"]["journal_path"])
    rows=0
    with journal.open("rb") as f:
        while True:
            try:pickle.load(f);rows+=1
            except EOFError:break
    ratio=int(cfg["experiment"]["rl"]["grad_updates_per_cycle"])
    warmup=int(cfg["experiment"]["rl"]["warmup_post_collect_updates"])
    if (step-warmup)%ratio:raise ValueError("Cannot represent update anchor exactly")
    anchor=rows-(step-warmup)//ratio
    if anchor<0:raise ValueError("Invalid Replay anchor")
    candidate.mkdir(parents=True)
    checkpoint=candidate/"checkpoints";checkpoint.mkdir()
    actor=candidate/"actor_snapshot";actor.mkdir()
    normal=Path(cfg["experiment"]["rl"]["action_norm_stats_path"]).resolve()
    (candidate/"action_norm_stats.json").symlink_to(normal)
    payload={"state":state,"rl_config":cfg["experiment"]["rl"],
             "progress":{"warmup_ready_adds_total":anchor},
             "experiment":{"profile":"mc_30","source_report":str(a.report),
                           "replay_adds_at_branch":rows,"inherited_updates":step}}
    with (checkpoint/"latest.pkl").open("wb") as f:pickle.dump(payload,f)
    snapshot={"actor_params":state["actor_params"],"version":int(state["actor_version"]),
              "global_step":step,"rl_config":cfg["experiment"]["rl"]}
    with (actor/"actor_snapshot.pkl").open("wb") as f:pickle.dump(snapshot,f)
    cfg["runtime"]["learner_service"].update(checkpoint_dir=str(checkpoint),actor_snapshot_path=str(actor/"actor_snapshot.pkl"))
    cfg["runtime"]["actor_service"]["snapshot_path"]=str(actor/"actor_snapshot.pkl")
    cfg["runtime"]["monitoring"]["wandb_run_name"]="credit-mc30-candidate"
    path=ROOT/"runtime/experiments/credit_mc30/online.yaml";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    receipt={"candidate":str(candidate),"config":str(path),"step":step,
             "actor_version":int(state["actor_version"]),"replay_adds":rows,
             "warmup_ready_adds_total":anchor,"pending_updates_at_creation":0,
             "replay_shared":True,"rtc":False,"hardware_acceptance":"pending"}
    (candidate/"provenance.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
if __name__=="__main__":main()
