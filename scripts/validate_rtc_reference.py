#!/usr/bin/env python3
"""Recorded-image RTC comparison; no ROS imports, sockets or robot publishers."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
overlay=ROOT/"envs/machine-a-py311-overlay"
if overlay.is_dir():sys.path.insert(0,str(overlay))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import numpy as np
from integrations.cobot_runtime.replay_audit import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--overlay",type=Path,required=True)
    p.add_argument("--recording",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    from methods.openpi_rlt.plug_v3_yyshadow.serve_stage1 import load
    policy=load(ROOT,args.checkpoint,rtc_overlay=args.overlay)
    import h5py,cv2
    with h5py.File(args.recording) as f:
        print("HDF5_KEYS",list(f.keys()),flush=True)
        states=np.asarray(f["observations/qpos"])
        image_group=f["observations/images"]
        keys=list(image_group)
        def image(key,index):
            raw=np.asarray(image_group[key][index])
            if raw.ndim==1:
                raw=cv2.imdecode(raw,cv2.IMREAD_COLOR)
                raw=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
            return raw.astype(np.uint8)
        names={"base_0_rgb":"cam_high","left_wrist_0_rgb":"cam_left_wrist","right_wrist_0_rgb":"cam_right_wrist"}
        report={"schema":1,"recording":str(args.recording),"checkpoint":str(args.checkpoint),
                "robot_publishers":0,"frames":[],"metadata":policy.metadata}
        for index in [0,min(40,len(states)-1),min(80,len(states)-1)]:
            state=states[index];state=state[-7:] if len(state)==14 else state
            obs={"state":state.astype(np.float32),"images":{k:image(v,index) for k,v in names.items()}}
            before=policy.infer(obs)
            # Perturb the pending prefix in one joint by a small physical amount.
            prefix=before["ref_chunk"].copy();prefix[:5,0]+=.01
            out=policy.infer(dict(obs,rtc={"previous_actions":prefix,"delay_steps":2,"execution_horizon":5}))
            repeated=policy.infer(obs)
            error_before=float(np.sqrt(np.mean((before["ref_chunk"][:2,:6]-prefix[:2,:6])**2)))
            error_after=float(np.sqrt(np.mean((out["ref_chunk"][:2,:6]-prefix[:2,:6])**2)))
            row={"frame":index,"baseline_ms":before["policy_timing"]["infer_ms"],
                "rtc_ms":out["policy_timing"]["infer_ms"],"prefix_error_before_rad":error_before,
                "prefix_error_after_rad":error_after,
                "baseline_repeat_max_abs":float(np.max(np.abs(before["ref_chunk"]-repeated["ref_chunk"]))),
                "token_max_abs":float(np.max(np.abs(before["z_rl"]-out["z_rl"])))}
            report["frames"].append(row);atomic_json(args.output,report);print(json.dumps(row),flush=True)
        report["finished_at"]=time.time();atomic_json(args.output,report)

if __name__=="__main__":main()
