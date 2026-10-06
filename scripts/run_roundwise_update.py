#!/usr/bin/env python3
"""Explicit optional closed-round entry. No learner service or robot deployment."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",required=True)
    p.add_argument("--registry",type=Path,default=ROOT/"configs/experiments/roundwise_online.json")
    p.add_argument("--dry-run",action="store_true")
    args,extra=p.parse_known_args()
    registry=json.loads(args.registry.read_text())
    if registry.get("enabled_by_default") or registry.get("release",{}).get("automatic"):
        p.error("roundwise entry requires optional, staged publication")
    if args.profile not in registry["profiles"]:p.error("unknown explicit profile")
    forbidden={"--actor-target","--actor-scope","--critic-target","--critic-steps","--actor-steps","--actor-lr","--delta-weight","--failure-anchor"}
    if any(x.split("=",1)[0] in forbidden for x in extra):
        p.error("profile-controlled settings must be changed in a reviewed registry")
    command=[sys.executable,str(ROOT/registry["entry"])]
    for key,value in registry["profiles"][args.profile].items():command += ["--"+key.replace("_","-"),str(value)]
    command += extra
    if args.dry_run:print(json.dumps(command));return
    raise SystemExit(subprocess.call(command))
if __name__=="__main__":main()
