"""Initialize an isolated online run from reviewed actor 5416; no ROS I/O."""
import argparse, hashlib, json, os, pickle, sys, uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.openpi_rlt.cobot_adapter.online_cycle import atomic_json,complete_episodes,load_journal

def prepare(root,run):
    root=Path(root).resolve();run=Path(run).resolve()
    if run.parent!=root/"runs" or run.name not in {"plug-online-from5416-20260914","plug-online-from5416-20260914-shadow"}:
        raise ValueError("unexpected continuation destination")
    if run.exists():
        manifest=json.loads((run/"continuation_manifest.json").read_text())
        if manifest["seed_actor_version"]!=5416:raise ValueError("wrong continuation seed")
        print("Resuming existing 5416 continuation; replay is not reimported.")
        return manifest
    review=root/"runs/plug-action-contract-review-20260914"
    meta=json.loads((review/"review_manifest.json").read_text())
    if meta["actor_version"]!=5416 or meta["global_step"]!=10832:raise ValueError("wrong review seed")
    seed={"actor":str(review/"artifacts/actor.pkl"),"checkpoint":str(review/"artifacts/checkpoint.pkl"),"actor_version":5416,"global_step":10832}
    for key in ["actor","checkpoint"]:
        if hashlib.sha256(Path(seed[key]).read_bytes()).hexdigest()!=meta["sha256"][key]:raise ValueError("seed hash mismatch")
    historical=Path(meta["source_run"])/"replay/replay_journal.pkl"
    recent=review/"replay/replay_journal.pkl"
    sources={}
    def stable_groups(path):
        if not path.exists():return {}
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        groups=complete_episodes(load_journal(path))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise RuntimeError("source replay is changing; end Session first")
        sources[str(path)]=before
        return groups
    old=stable_groups(historical);new=stable_groups(recent)
    if not old:raise ValueError("historical training replay missing")
    if max(old)>=10000:raise ValueError("historical episode ID exceeds mapping range")
    staging=run.with_name(run.name+".initializing-"+uuid.uuid4().hex)
    (staging/"replay").mkdir(parents=True)
    mapping={str(ep):10000+ep for ep in new}
    with (staging/"replay/replay_journal.pkl").open("wb") as handle:
        for ep,rows in sorted(old.items()):
            for row in rows:pickle.dump(row,handle)
        for ep,rows in sorted(new.items()):
            for row in rows:pickle.dump({**row,"episode_id":mapping[str(ep)]},handle)
        handle.flush();os.fsync(handle.fileno())
    manifest={"seed_actor_version":5416,"seed":seed,"source_sha256":sources,"historical_episodes":sorted(old),"review_episode_mapping":mapping,"pending_review_episodes":len(new)}
    atomic_json(staging/"current.json",{**seed,"seed":seed,"processed":sorted(old),"transactions":0,"previous":None})
    atomic_json(staging/"continuation_manifest.json",manifest)
    atomic_json(staging/"update_request.json",{"request_id":"import-5416-"+uuid.uuid4().hex})
    os.rename(staging,run)
    print("Initialized from actor 5416; historical episodes:",len(old),"; pending review episodes:",len(new))
    return manifest

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",type=Path,required=True);parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args();prepare(args.project_root,args.run)
