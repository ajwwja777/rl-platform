#!/usr/bin/env python3
"""Quantify noise units and time-preserving publication; never emits robot commands."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--norm",type=Path,required=True)
    p.add_argument("--credit-report",type=Path,required=True)
    p.add_argument("--rtc-report",type=Path,required=True)
    p.add_argument("--vla-root",type=Path,default=ROOT.parent/"vla-platform")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.vla_root/"integrations/cobot/pi05/dagger/common/runtime_lib"))
    from execution_methods.action_processing import resample_actions,bounded_noise
    stats=json.loads(a.norm.read_text())["norm_stats"]["actions"]
    scale=(np.asarray(stats["q99"])-np.asarray(stats["q01"]))/2
    report={"schema":1,"robot_publishers":0,"noise":[],"timing":[],"interpolation":[],
        "limitations":["No robot success/safety claim.","Timing rows are calculated from recorded-frame latency, not measured control-loop timing.","Noise sweep is action-space sensitivity, not environment exploration."]}
    for std in [.001,.002,.004,.01]:
        for rho in [0,.8]:
            noise=bounded_noise(np.random.default_rng(42),(10000,7),std,rho=rho,active_dimensions=[1]*6+[0])
            native=noise*scale
            report["noise"].append({"normalized_std":std,"rho":rho,"joint_std_rad":np.std(native[:,:6],axis=0).tolist(),
                "gripper_noise_m":0.,"per_command_delta_rms_rad":float(np.sqrt(np.mean(np.diff(native[:,:6],axis=0)**2)))})
    rtc=json.loads(a.rtc_report.read_text())
    latency=float(np.median([r["baseline_ms"] for r in rtc["frames"][1:]]))/1000
    for horizon in [5,10]:
        motion=horizon/20
        report["timing"].append({"executed_steps":horizon,"logical_hz":20,"nominal_open_loop_sec":motion,
            "synchronous_plan_hz":1/(motion+latency),"blocking_fraction":latency/(motion+latency)})
    credit=json.loads(a.credit_report.read_text())
    candidate=next(r for r in credit["experiments"] if r["variant"]=="mc_30" and r["seed"]==42)
    for trace in candidate["validation"]["traces"]:
        actions=np.asarray(trace["actor"])
        initial=actions[0]
        t,smooth=resample_actions(actions,initial,20,40)
        original_delta=np.diff(np.vstack([initial,actions]),axis=0)[:,:6]
        smooth_delta=np.diff(np.vstack([initial,smooth]),axis=0)[:,:6]
        report["interpolation"].append({"episode":trace["episode"],"step":trace["step"],
            "duration_sec":float(t[-1]),"original_duration_sec":len(actions)/20,
            "endpoint_max_error":float(np.max(np.abs(smooth[1::2]-actions))),
            "command_delta_rms_before":float(np.sqrt(np.mean(original_delta**2))),
            "command_delta_rms_after":float(np.sqrt(np.mean(smooth_delta**2)))})
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=="__main__":main()
