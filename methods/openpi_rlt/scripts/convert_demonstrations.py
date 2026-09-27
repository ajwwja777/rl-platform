#!/usr/bin/env python3
"""Convert successful segmented demonstrations to boundary-safe LeRobot v2.1.

Raw sources are immutable. Invalid frames split derived episodes; all fragments
of a source UUID share one split. Source facts/reviews remain as small archives.
Deletion is a separate operation allowed only after verified publication.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from functools import partial
import h5py
import numpy as np

CAMERAS=("cam_high","cam_left_wrist","cam_right_wrist")
VALID=("action","qpos","camera_high","camera_left","camera_right")
BASE=Path("/media/agilex/Getea1/jiaan/data")
BACKEND=Path("/media/agilex/Getea1/jiaan/projects/cobot-platform/app/backend")

def jsonable(v):
    if isinstance(v,bytes):return v.decode("utf8")
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def write_json(path,value):
    with path.open("x",encoding="utf8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,default=jsonable)
        f.write("\n");f.flush();os.fsync(f.fileno())

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(np.int8))
    return list(zip(np.where(edges==1)[0].tolist(),np.where(edges==-1)[0].tolist()))

def assign_split(uuids,val_count,seed=42):
    if len(set(uuids))!=len(uuids) or not 0<val_count<len(uuids):raise ValueError("invalid split")
    ranked=sorted(uuids,key=lambda u:hashlib.sha256(f"{seed}:{u}".encode()).digest())
    val=set(ranked[-val_count:])
    return {u:"val" if u in val else "train" for u in uuids}

def safe_root(root):
    root=root.absolute()
    if root.is_symlink():raise ValueError("source root symlink refused")
    resolved=root.resolve(strict=True)
    resolved.relative_to(BASE.resolve(strict=True))
    return resolved

def inventory(root):
    sources=[]
    paths=sorted(root.glob("episode_*.hdf5"))
    if not paths:raise ValueError("no complete source episodes")
    sidecars={}
    for p in (root/".segments").glob("*/sidecar.json"):
        s=json.loads(p.read_text())
        name=s.get("source_hdf5_relative")
        if name in sidecars:raise ValueError("duplicate source lineage")
        sidecars[name]=(p,s)
    for p in paths:
        if p.is_symlink():raise ValueError("source symlink refused")
        sc,s=sidecars[p.name]
        if s["commit_state"]!="complete" or s["capture_state"]!="committed":raise ValueError("uncommitted sidecar")
        with h5py.File(p,"r") as h:
            attrs={k:jsonable(v) for k,v in h.attrs.items()}
            n=len(h["action"])
            if attrs["completion_state"]!="complete" or attrs["episode_uuid"]!=s["episode_uuid"]:raise ValueError("source identity")
            if s["training_frame_count"]!=n or float(attrs["fps"])!=30:raise ValueError("frame/fps contract")
            valid=np.ones(n,dtype=bool)
            for key in ["action","observations/qpos"]:
                v=h[key][:]
                if v.shape!=(n,14):raise ValueError("vector shape")
                valid &= np.isfinite(v).all(axis=1)
            for key in VALID:valid &= h["rollout/valid_mask/"+key][:]
            for cam in CAMERAS:
                if h["observations/images/"+cam].shape!=(n,480,640,3):raise ValueError("camera contract")
            intervals=s["training_intervals"]
            spans=[]
            for it in intervals:
                start,end=int(it["start_frame"]),int(it["end_frame_exclusive"])
                if not 0<=start<end<=n:raise ValueError("invalid interval")
                spans.extend((start+a,start+b) for a,b in runs(valid[start:end]))
            if not spans:raise ValueError("source has no valid fragment")
        st=p.stat()
        sources.append(dict(path=p.name,uuid=s["episode_uuid"],source_index=s["episode_index"],
            frames=n,sidecar=str(sc.relative_to(root)),sidecar_sha256=sha(sc),
            source_sha256=sha(p),size=st.st_size,mtime_ns=st.st_mtime_ns,
            attrs=attrs,spans=spans,invalid_frames=np.where(~valid)[0].tolist()))
    return sources

def archive_facts(root,out,source):
    folder=out/"source_metadata"/Path(source["path"]).stem
    folder.mkdir(parents=True)
    source_path=root/source["path"]
    arrays={}
    with h5py.File(source_path,"r") as h:
        def visitor(name,ds):
            if not isinstance(ds,h5py.Dataset) or name.startswith("observations/images/"):return
            if h5py.check_string_dtype(ds.dtype):arrays[name]=np.asarray(ds.asstr()[:],dtype=str)
            else:arrays[name]=ds[:]
        h.visititems(visitor)
    np.savez_compressed(folder/"facts.npz",**arrays)
    write_json(folder/"source.json",source)
    sidecar_root=(root/source["sidecar"]).parent
    for name in ["sidecar.json","segment-review.json"]:
        p=sidecar_root/name
        if p.is_file():(folder/name).write_bytes(p.read_bytes())
    label=(root/source["path"]).with_suffix(".labels.json")
    if label.is_file():(folder/"labels.json").write_bytes(label.read_bytes())

def verify(root,out,manifest):
    import pyarrow.parquet as pq
    import av
    worst_mae=0.; decoded_total=0
    for item in manifest["fragments"]:
        n=item["end"]-item["start"];ep=item["episode_index"]
        table=pq.read_table(out/f"data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
        if len(table)!=n:raise ValueError("parquet count")
        with h5py.File(root/item["source"],"r") as h:
            for key,src in [("observation.state","observations/qpos"),("action","action")]:
                got=np.stack(table[key].to_pylist()).astype(np.float32)
                want=h[src][item["start"]:item["end"]]
                if not np.array_equal(got,want):raise ValueError("state/action mismatch")
                if not np.isfinite(got).all():raise ValueError("nonfinite training vectors")
            for cam in CAMERAS:
                p=out/f"videos/chunk-{ep//1000:03d}/observation.images.{cam}/episode_{ep:06d}.mp4"
                checks={0,n//2,n-1};count=0
                with av.open(str(p)) as video:
                    stream=video.streams.video[0]
                    if stream.width!=640 or stream.height!=480 or float(stream.average_rate)!=30:raise ValueError("video shape/fps")
                    for frame in video.decode(stream):
                        if count in checks:
                            image=frame.to_ndarray(format="rgb24")
                            want=h["observations/images/"+cam][item["start"]+count]
                            mae=float(np.abs(image.astype(np.float32)-want.astype(np.float32)).mean())
                            worst_mae=max(worst_mae,mae)
                            if mae>5.:raise ValueError(f"video quality mismatch {cam}: {mae}")
                        count+=1
                if count!=n:raise ValueError("decoded video frame count")
                decoded_total+=count
        print("VERIFY",ep,n,flush=True)
    for s in manifest["sources"]:
        facts=out/"source_metadata"/Path(s["path"]).stem/"facts.npz"
        with np.load(facts,allow_pickle=False) as archive,h5py.File(root/s["path"],"r") as h:
            for key in archive.files:
                ds=h[key];original=np.asarray(ds.asstr()[:],dtype=str) if h5py.check_string_dtype(ds.dtype) else ds[:]
                a=archive[key]
                same=np.array_equal(a,original,equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a,original)
                if not same:raise ValueError("source facts mismatch "+key)
        st=(root/s["path"]).stat()
        if st.st_size!=s["size"] or st.st_mtime_ns!=s["mtime_ns"] or sha(root/s["sidecar"])!=s["sidecar_sha256"]:raise ValueError("source modified during conversion")
    return dict(status="passed",parquet_vectors="all_frames_exact",videos="all_videos_fully_decoded",
        video_frame_count=decoded_total,max_sample_rgb_mae=worst_mae,source_facts="all_nonimage_datasets_exact",
        raw_source_count=len(manifest["sources"]))

def convert(args):
    from importlib.metadata import version
    if version("lerobot")!="0.1.0":raise ValueError("pinned v2.1 writer version mismatch")
    root=safe_root(args.root);out=root/"lerobot";staging=root/".lerobot-converting-20260918"
    if out.exists() or staging.exists():raise FileExistsError("output/staging already exists; inspect instead of overwriting")
    sources=inventory(root)
    if len(sources)!=args.expected_episodes:raise ValueError("source count changed")
    splits=assign_split([s["uuid"] for s in sources],args.val_episodes)
    sys.path.insert(0,str(BACKEND))
    from segmented_capture.review import SegmentReviewStore
    review=SegmentReviewStore(root/".segments")
    for s in sources:
        r=review.load(s["uuid"])
        selected=[int(i["interval_id"]) for i in r["intervals"]]
        if r["selected_interval_ids"]!=selected:
            review.save(s["uuid"],expected_revision=int(r["review_revision"]),
                selected_interval_ids=selected,note="2026-09-18: user confirmed all retained episodes are successful insertion-only demonstrations; invalid frames are split in derived training data.")
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets.video_utils import encode_video_frames
    lerobot_dataset.encode_video_frames=partial(encode_video_frames,vcodec="h264",crf=10)
    features={k:{"dtype":"float32","shape":(14,),"names":None} for k in ["observation.state","action"]}
    for cam in CAMERAS:features["observation.images."+cam]={"dtype":"video","shape":(480,640,3),"names":["height","width","channels"]}
    dataset=lerobot_dataset.LeRobotDataset.create(repo_id="jiaan/plug_v2_demonstrations",fps=30,
        root=staging,robot_type="cobot",features=features,use_videos=True,image_writer_threads=4)
    manifest=dict(schema_version="cobot-successful-demonstrations-v1",status="converting",
        task=args.task,source_root=str(root),repo_id="jiaan/plug_v2_demonstrations",
        lerobot_package="0.1.0",format="v2.1",codec="H264 CRF10 yuv420p",
        success_source_episode_count=len(sources),source_frame_count=sum(s["frames"] for s in sources),
        sources=sources,fragments=[],split_seed=42,source_split_counts={"train":len(sources)-args.val_episodes,"val":args.val_episodes},
        action_contract="14D absolute front joint targets (rad), grippers (m); delta transform applied by training loader")
    try:
        for s in sources:
            archive_facts(root,staging,s)
            with h5py.File(root/s["path"],"r") as h:
                for a,b in s["spans"]:
                    ep=len(manifest["fragments"])
                    for i in range(a,b):
                        frame={f"observation.images.{cam}":h["observations/images/"+cam][i] for cam in CAMERAS}
                        frame.update({"observation.state":h["observations/qpos"][i],"action":h["action"][i],"task":args.task})
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    manifest["fragments"].append(dict(episode_index=ep,source=s["path"],source_uuid=s["uuid"],
                        start=a,end=b,split=splits[s["uuid"]],h50_anchor_frames=list(range(max(0,b-a-50+1)))))
                    print("CONVERT",ep,s["path"],a,b,splits[s["uuid"]],flush=True)
    finally:dataset.stop_image_writer()
    manifest["training_frame_count"]=sum(f["end"]-f["start"] for f in manifest["fragments"])
    manifest["derived_episode_count"]=len(manifest["fragments"])
    manifest["excluded_frame_count"]=manifest["source_frame_count"]-manifest["training_frame_count"]
    manifest["validation"]=verify(root,staging,manifest)
    manifest["status"]="validated"
    manifest["bytes"]=sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    write_json(staging/"conversion_manifest.json",manifest)
    write_json(staging/"splits.json",{split:[f["episode_index"] for f in manifest["fragments"] if f["split"]==split] for split in ["train","val"]})
    checks={str(p.relative_to(staging)):sha(p) for p in staging.rglob("*") if p.is_file()}
    write_json(staging/"artifact_checksums.json",dict(manifest_sha256=sha(staging/"conversion_manifest.json"),files=checks))
    os.rename(staging,out)
    print("SUCCESS",json.dumps({k:manifest[k] for k in ["source_frame_count","training_frame_count","derived_episode_count","excluded_frame_count","bytes"]}),flush=True)

def delete_validated(args):
    root=safe_root(args.root);out=root/"lerobot"
    manifest=json.loads((out/"conversion_manifest.json").read_text())
    if manifest["status"]!="validated" or manifest["validation"]["status"]!="passed" or manifest["source_root"]!=str(root):raise ValueError("validated release required")
    checks=json.loads((out/"artifact_checksums.json").read_text())
    if checks["manifest_sha256"]!=sha(out/"conversion_manifest.json"):raise ValueError("manifest changed after validation")
    for relative,expected in checks["files"].items():
        artifact=out/relative
        if artifact.is_symlink() or not artifact.is_file() or not artifact.resolve().is_relative_to(out.resolve()) or sha(artifact)!=expected:
            raise ValueError("converted artifact missing/changed: "+relative)
    sources=manifest["sources"]
    paths=[root/s["path"] for s in sources]
    if set(root.glob("episode_*.hdf5"))!=set(paths):raise ValueError("source inventory changed")
    for s,p in zip(sources,paths):
        if p.is_symlink() or p.resolve().parent!=root or p.suffix!=".hdf5":raise ValueError("unsafe deletion target")
        if sha(p)!=s["source_sha256"]:raise ValueError("source hash changed; refuse deletion")
        if not (out/"source_metadata"/p.stem/"facts.npz").is_file():raise ValueError("facts archive missing")
    opened=[]
    targets={str(p) for p in paths}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (proc/"fd").iterdir():
                try:
                    target=os.readlink(fd)
                    if target in targets:opened.append((proc.name,target))
                except OSError:pass
        except OSError:pass
    if opened:raise ValueError("source files currently open: "+str(opened))
    write_json(out/"deletion_plan.json",dict(source_root=str(root),paths=[s["path"] for s in sources],
        source_sha256=[s["source_sha256"] for s in sources],bytes=sum(s["size"] for s in sources)))
    deleted=[]
    for s,p in zip(sources,paths):
        st=p.stat()
        if st.st_size!=s["size"] or st.st_mtime_ns!=s["mtime_ns"]:raise ValueError("source stat changed")
        p.unlink();deleted.append(s["path"])
    previews=[]
    for s in sources:
        p=root/".previews"/s["uuid"]/"preview.mp4"
        if p.is_file() and not p.is_symlink() and p.resolve().parent.parent==root/".previews":
            p.unlink();previews.append(str(p.relative_to(root)))
    result=dict(status="complete",deleted_source_count=len(deleted),freed_source_bytes=sum(s["size"] for s in sources),
        deleted_preview_videos=previews,remaining_hdf5=len(list(root.glob("*.hdf5"))),
        success_source_episode_count=manifest["success_source_episode_count"],training_root=str(out))
    write_json(out/"deletion_receipt.json",result)
    print("DELETE_SUCCESS",json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--expected-episodes",type=int,default=82)
    p.add_argument("--val-episodes",type=int,default=8)
    p.add_argument("--task",default="Insert the held plug into the socket.")
    p.add_argument("--delete-validated-hdf5",action="store_true")
    a=p.parse_args()
    if a.delete_validated_hdf5:delete_validated(a)
    else:convert(a)
if __name__=="__main__":main()
