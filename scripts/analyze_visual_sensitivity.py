#!/usr/bin/env python3
"""Offline camera/patch occlusion on recorded frames; no ROS or policy server."""
import argparse,base64,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
overlay=ROOT/"envs/machine-a-py311-overlay"
if overlay.is_dir():sys.path.insert(0,str(overlay))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",".55")

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--checkpoint",type=Path,required=True)
 p.add_argument("--recordings",type=Path,nargs="+",required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 import h5py,numpy as np
 from PIL import Image
 from methods.openpi_rlt.plug_v3_yyshadow.serve_stage1 import load
 from integrations.cobot_runtime.replay_audit import atomic_json
 policy=load(ROOT,a.checkpoint)
 cameras={"cam_high":"base_0_rgb","cam_left_wrist":"left_wrist_0_rgb","cam_right_wrist":"right_wrist_0_rgb"}
 report={"schema":1,"generated_at":time.time(),"checkpoint":str(a.checkpoint),"method":"4x4 camera patch occlusion; same frame/state/prompt and fixed denoising seed 42",
  "units":"First 10 reference actions, six-joint RMS difference in radians. Gripper excluded.",
  "limitations":["Occlusion sensitivity is NOT attention or causal proof.","Constant-color replacement may be out of distribution.","Two deliberately selected recordings; no population-level conclusion.","Local HDF5 episode numbers are not assumed to equal Replay IDs."],"frames":[]}
 for path in a.recordings:
  with h5py.File(path) as f:
   n=len(f["observations/qpos"]);labels=path.with_suffix(".labels.json")
   label=json.loads(labels.read_text()) if labels.exists() else {}
   for index in sorted(set([0,n//2,n-1])):
    keys=["qpos","camera_high","camera_left","camera_right"]
    if any("rollout/valid_mask/"+k in f and not f["rollout/valid_mask/"+k][index] for k in keys):continue
    images={v:np.asarray(f["observations/images/"+k][index],dtype=np.uint8) for k,v in cameras.items()}
    state=np.asarray(f["observations/qpos"][index,7:14],dtype=np.float32)
    obs=dict(images=images,state=state)
    base=policy.infer(obs)
    frame=dict(path=str(path),episode_uuid=label.get("episode_uuid"),outcome=label.get("episode_outcome","unknown"),frame=index,total_frames=n,
     human=bool(f["rollout/is_intervention_right"][index]) if "rollout/is_intervention_right" in f else None,cameras=[])
    for camera,key in cameras.items():
     image=images[key];h,w=image.shape[:2];fill=np.mean(image,axis=(0,1)).astype(np.uint8)
     def measure(changed):
      altered=dict(images);altered[key]=changed
      result=policy.infer(dict(images=altered,state=state))
      return float(np.sqrt(np.mean((result["ref_chunk"][...,:6]-base["ref_chunk"][...,:6])**2))),float(np.linalg.norm(result["z_rl"]-base["z_rl"])/max(np.linalg.norm(base["z_rl"]),1e-8))
     full,token=measure(np.broadcast_to(fill,image.shape).copy());patch=[]
     for y in range(4):
      for x in range(4):
       changed=image.copy();changed[y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]=fill
       shift,zshift=measure(changed);patch.append(dict(row=y,column=x,action_shift_rad=shift,relative_rl_feature_change=zshift))
     buf=io.BytesIO();thumb=Image.fromarray(image);thumb.thumbnail((320,240));thumb.save(buf,format="JPEG",quality=75)
     frame["cameras"].append(dict(camera=camera,image_jpeg=base64.b64encode(buf.getvalue()).decode(),patches=patch,full_camera_shift_rad=full,full_camera_feature_change=token))
    report["frames"].append(frame);atomic_json(a.output,report);print(path.name,index,flush=True)
 report["finished_at"]=time.time();atomic_json(a.output,report)
 print(a.output,flush=True)
if __name__=="__main__":main()
