import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[3]
ENTRY=ROOT/"scripts/run_roundwise_update.py"

def run(*args):
    return subprocess.run([sys.executable,str(ENTRY),*args],capture_output=True,text=True)

def test_explicit_profiles_keep_single_factor_controls():
    r=json.loads((ROOT/"configs/experiments/roundwise_online.json").read_text())
    assert not r["enabled_by_default"] and not r["release"]["automatic"]
    p=r["profiles"]
    def changed(a,b):
        return {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    assert changed(p["conservative"],p["learning_rate_control"])=={"actor_lr"}
    assert changed(p["learning_rate_control"],p["coverage_control"])=={"actor_scope"}
    r=run("--profile","conservative","--dry-run","--journal","/closed/journal","--output","/private/candidates/test")
    assert r.returncode==0
    command=json.loads(r.stdout)
    assert command[command.index("--actor-lr")+1]=="1e-05"
    assert command[command.index("--actor-scope")+1]=="full"
    assert command[command.index("--output")+1]=="/private/candidates/test"

@pytest.mark.parametrize("args",[[],["--profile","unknown","--dry-run"],["--profile","conservative","--actor-lr=.5","--dry-run"]])
def test_invalid_selection_does_not_start_training(args):
    r=run(*args)
    assert r.returncode!=0
    assert not r.stdout

def test_registry_cannot_enable_automatic_publication(tmp_path):
    r=json.loads((ROOT/"configs/experiments/roundwise_online.json").read_text())
    r["release"]["automatic"]=True
    p=tmp_path/"registry.json";p.write_text(json.dumps(r))
    result=run("--registry",str(p),"--profile","conservative","--dry-run")
    assert result.returncode!=0 and "staged publication" in result.stderr
