#!/usr/bin/env python3
"""Materialize read-only Stage1 experts into isolated offline replay."""
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av
from openpi_client.websocket_client_policy import WebsocketClientPolicy
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.openpi_rlt.cobot_adapter.offline_warmup import feature_indices,expert_transitions

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",type=Path,required=True);ap.add_argument("--run",type=Path,required=True)
    args=ap.parse_args();d=args.dataset;out=args.run
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((d/"cobot_release.json").read_text())
    candidates=[];excluded=[]
    for e in manifest["episodes"]:
        idx=e["release_episode_index"]
        p=d/f"data/chunk-000/episode_{idx:06d}.parquet"
        t=pq.read_table(p)
        state=np.array(t["observation.state"].to_pylist(),np.float32)
        action=np.array(t["action"].to_pylist(),np.float32)
        ts=np.array(t["timestamp"].to_pylist())
        good=state.shape==action.shape and state.shape[1]==14 and np.isfinite(state).all() and np.isfinite(action).all() and np.all(np.diff(ts)>0)
        jump=float(np.max(np.abs(np.diff(action[:,7:13],axis=0))))
        if not good or jump>.1 or len(ts)<20:
            excluded.append(dict(index=idx,finite=bool(good),max_action_jump=jump));continue
        v=np.concatenate([state[0,7:13],state[-1,7:13],np.ptp(state[:,7:13],axis=0)])
        candidates.append(dict(entry=e,state=state,action=action,ts=ts,vector=v,max_jump=jump,parquet_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    def choose(split,n):
        pool=[x for x in candidates if x["entry"]["split"]==split]
        if len(pool)<n: raise ValueError(f"not enough {split} expert episodes")
        vectors=np.stack([x["vector"] for x in pool]);vectors=(vectors-vectors.mean(0))/np.maximum(vectors.std(0),.05)
        chosen=[int(np.argmin(np.sum(vectors**2,axis=1)))]
        while len(chosen)<n:
            dist=np.min(np.sum((vectors[:,None]-vectors[chosen][None])**2,axis=-1),axis=1)
            dist[chosen]=-1;chosen.append(int(np.argmax(dist)))
        return [pool[i] for i in chosen]
    selected=choose("train",20)+choose("val",4)
    selection=dict(excluded=excluded,selection_method="farthest-point initial/final/range right-arm state within original train/val splits",episodes=[{k:v for k,v in x.items() if k not in ["state","action","ts","vector"]} for x in selected],source_manifest_sha256=hashlib.sha256((d/"cobot_release.json").read_bytes()).hexdigest())
    (out/"expert_selection.json").write_text(json.dumps(selection,indent=2))
    client=WebsocketClientPolicy("127.0.0.1",8000)
    meta=client.get_server_metadata()
    assert meta["has_rl_token"] and meta["action_dim"]==14 and meta["z_dim"]==2048
    features_dir=out/"expert_features";features_dir.mkdir(exist_ok=True)
    all_records=[];start_time=time.perf_counter()
    for x in selected:
        e=x["entry"];idx=e["release_episode_index"];ts=x["ts"]
        # Nearest original frame on a 20 Hz grid; preserve final terminal frame.
        grid=np.arange(ts[0],ts[-1],.05)
        pos=np.searchsorted(ts,grid);pos=np.clip(pos,1,len(ts)-1)
        pos=np.where(np.abs(ts[pos]-grid)<np.abs(ts[pos-1]-grid),pos,pos-1)
        pos=np.unique(np.r_[pos,len(ts)-1]).astype(int)
        states=x["state"][pos];actions=x["action"][pos]
        needed=feature_indices(len(pos),10);wanted={int(pos[i]) for i in needed}
        images={}
        video_hashes={}
        for cam,key in [("cam_high","base_0_rgb"),("cam_left_wrist","left_wrist_0_rgb"),("cam_right_wrist","right_wrist_0_rgb")]:
            p=d/f"videos/chunk-000/observation.images.{cam}/episode_{idx:06d}.mp4"
            video_hashes[cam]=hashlib.sha256(p.read_bytes()).hexdigest()
            with av.open(str(p)) as container:
                frames={i:frame.to_ndarray(format="rgb24") for i,frame in enumerate(container.decode(video=0)) if i in wanted}
            if set(frames)!=wanted:raise ValueError(f"missing video frames {idx} {cam}")
            images[key]=frames
        cache=features_dir/f"episode_{idx:06d}.pkl"
        if cache.exists(): raise FileExistsError(cache)
        features={}
        for i in needed:
            obs=dict(state=states[i],images={k:v[int(pos[i])] for k,v in images.items()},prompt="Insert the held plug into the socket.")
            payload=client.infer(obs)
            z=np.asarray(payload["z_rl"],np.float32).reshape(-1)
            ref=np.asarray(payload["ref_chunk"],np.float32)[:10,:14]
            assert z.shape==(2048,) and ref.shape==(10,14) and np.isfinite(z).all() and np.isfinite(ref).all()
            features[i]=dict(z_rl=z,ref_chunk=ref,proprio=states[i])
        records=expert_transitions(states,actions,features,episode_id=100000+idx)
        for r in records:r["split"]=e["split"];r["expert_release_index"]=idx;r["expert_uuid"]=e["episode_uuid"]
        with cache.open("xb") as f:pickle.dump(dict(features=features,indices=pos,video_hashes=video_hashes,records=records),f)
        all_records.extend(records)
        print(json.dumps(dict(episode=idx,split=e["split"],source_frames=len(ts),resampled_frames=len(pos),features=len(features),transitions=len(records),elapsed=time.perf_counter()-start_time)),flush=True)
    with (out/"expert_replay.pkl").open("xb") as f:pickle.dump(all_records,f)
    (out/"expert_complete.json").write_text(json.dumps(dict(episodes=len(selected),transitions=len(all_records),feature_seconds=time.perf_counter()-start_time,metadata=meta),indent=2))
if __name__=="__main__":main()
