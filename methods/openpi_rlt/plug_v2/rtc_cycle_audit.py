"""Actual paused replay batch qualification with isolated publish pointer; no new data claimed."""
import json,argparse
from pathlib import Path
from . import rtc_online_cycle as cycle
from .training_flow import atomic
from .rtc_release import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);a=p.parse_args();path=a.release.resolve();m=json.loads(path.read_text())
    folder=cycle.RUN/'learning/rtc-v5/testing';folder.mkdir(exist_ok=True)
    pointer=folder/'current.json';atomic(pointer,dict(release=str(path),release_sha256=digest(path)))
    cycle.CURRENT=pointer;cycle.selected=lambda:(path,m)
    entries=json.loads((Path(m['dataset'])/'manifest.json').read_text())
    uuids=sorted({x['uuid'] for x in entries if not x['expert'] and x['split']=='train'})[:5]
    result=cycle.run_batch(path,m,uuids)
    report=dict(scope='existing replay five-episode qualification; not new online improvement',production_pointer_untouched=True,accepted=result is not None,release=result,pointer=json.loads(pointer.read_text()))
    atomic(folder/'cycle-audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
