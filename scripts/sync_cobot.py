#!/usr/bin/env python3
"""Publish a committed project from A6000 without restarting any services."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def run(args, **kw):
    return subprocess.run(args, check=True, text=True, **kw)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="agilex@10.7.165.64")
    args = parser.parse_args()
    if ROOT.name not in {"rl-platform", "cobot-control"}:
        parser.error("Unregistered project")
    target = "/home/agilex/jiaan/project/" + ROOT.name
    run(["git","diff","--quiet","HEAD"], cwd=ROOT)
    revision=run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True).stdout.strip()
    names=run(["git","ls-files","--recurse-submodules","-z"],cwd=ROOT,capture_output=True).stdout.split("\0")
    def runtime_file(name):
        if not name: return False
        prefix="third_party/openpi-rlt/"
        if name.startswith(prefix):
            relative=name[len(prefix):]
            return relative.startswith(("src/","scripts/","rlt_online_rl/","packages/openpi-client/","third_party/lerobot/lerobot/")) or relative in {"LICENSE","README.md","pyproject.toml"}
        return True
    files=[name for name in names if runtime_file(name)]
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    output=ROOT/"outputs/deployments";output.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=output) as listing:
        listing.write("\n".join(files)+"\n");listing.flush()
        run(["rsync","-a","--no-owner","--no-group","--files-from="+listing.name,
             "-e","ssh -o BatchMode=yes",str(ROOT)+"/",args.host+":"+target+"/"])
    payload={"project":ROOT.name,"revision":revision,"target":target,"hashes":hashes}
    verify="from pathlib import Path\nimport json,hashlib\n"
    verify+="receipt="+repr(payload)+"\nroot=Path(receipt['target'])\n"
    verify+="bad=[p for p,h in receipt['hashes'].items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]\n"
    verify+="assert not bad,bad\n(root/'.release.json').write_text(json.dumps(receipt,indent=2))\nprint('Verified',len(receipt['hashes']),'files; services unchanged')\n"
    result=run(["ssh",args.host,"python3","-"],input=verify,capture_output=True)
    (output/(revision+".json")).write_text(json.dumps(payload,indent=2))
    print(result.stdout.strip())
if __name__=="__main__": main()
