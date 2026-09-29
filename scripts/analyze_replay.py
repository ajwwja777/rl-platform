#!/usr/bin/env python3
"""Build a read-only posture/action PCA + k-means snapshot from the registered trusted journal."""
import argparse
import json
import os
import pickle
import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
from integrations.cobot_runtime.replay_audit import build_report, atomic_json

def build(journal, output, clusters=6):
    # Pickle is a trusted local training artifact, never accepted from an HTTP request.
    rows, features, details = [], [], {}
    size = journal.stat().st_size
    skipped = 0
    with journal.open("rb") as f:
        while f.tell() < size:
            try: row = pickle.load(f)
            except EOFError: break
            if f.tell() > size: break
            if not isinstance(row, dict): skipped += 1; continue
            p = np.asarray(row.get("proprio", []), dtype=float).reshape(-1)
            action = np.asarray(row.get("action_chunk", []), dtype=float)
            ref = np.asarray(row.get("ref_chunk", []), dtype=float)
            if p.shape != (7,) or action.ndim != 2 or action.shape != ref.shape or action.shape[1] != 7:
                skipped += 1; continue
            feature = np.r_[p, (action-ref).mean(axis=0)]
            if not np.isfinite(feature).all(): skipped += 1; continue
            meta = {"episode_id": int(row.get("episode_id",-1)), "step_id": int(row.get("step_id",-1)),
                "phase": str(row.get("collection_phase","unknown")), "source": int(row.get("source",0)),
                "intervention": bool(row.get("intervention_flag",False))}
            features.append(feature); rows.append(meta)
            # Retain a bounded representative chunk per episode for exact replay identity.
            key = str(meta["episode_id"])
            if key not in details or meta["intervention"]:
                details[key] = dict(meta, proprio=p.tolist(), action=action.tolist(), reference=ref.tolist())
    if len(features) < 2: raise ValueError("At least two finite Replay transitions are required")
    x = np.asarray(features)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    normalized = (x-mean)/scale
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    coords = normalized @ vt[:2].T
    # Clustering is in 14-dimensional standardized space, not in the 2D picture.
    k = min(clusters, len(x))
    rng = np.random.default_rng(42)
    centers = normalized[rng.choice(len(x), k, replace=False)]
    for _ in range(40):
        labels = ((normalized[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        updated = np.array([normalized[labels==i].mean(axis=0) if (labels==i).any() else centers[i] for i in range(k)])
        if np.allclose(updated, centers): break
        centers = updated
    labels = ((normalized[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
    indices = np.sort(rng.choice(len(x), min(2400,len(x)), replace=False))
    groups = []
    for i in range(k):
        members = [rows[j] for j in np.flatnonzero(labels==i)]
        groups.append({"cluster": i, "count": len(members),
            "online": sum(v["phase"]=="online" for v in members),
            "hil": sum(v["intervention"] for v in members)})
    result = {"schema":1, "generated_at":time.time(), "journal":str(journal),
        "journal_bytes":size, "transitions":len(x), "skipped":skipped,
        "feature":"proprio[7] + mean(action_chunk - ref_chunk)[7]; standardized",
        "method":"PCA + k-means in 14D", "seed":42,
        "explained_variance":(singular[:2]**2/max(float((singular**2).sum()),1e-12)).tolist(),
        "clusters":groups,
        "points":[dict(rows[i], x=round(float(coords[i,0]),5), y=round(float(coords[i,1]),5),
                       cluster=int(labels[i])) for i in indices],
        "episode_chunks":details}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False))
    os.replace(temporary,output)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/rlt/plug_v3_yyshadow/online_rl.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/rlt/plug_v3_yyshadow/analysis/replay_projection.json")
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text())
    journal=Path(cfg["runtime"]["replay"]["journal_path"])
    result=build(journal,args.output)
    atomic_json(args.output.with_name("replay_composition.json"), build_report(journal))
    print(json.dumps({k:result[k] for k in ("transitions","skipped","explained_variance","generated_at")}))
    print(args.output)
if __name__=="__main__": main()
