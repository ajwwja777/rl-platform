import hashlib,json,pickle
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.scripts.prepare_5416_continuation import prepare

def fixture(root):
    review=root/"runs/plug-action-contract-review-20260914"
    (review/"artifacts").mkdir(parents=True)
    hashes={}
    for key in ["actor","checkpoint"]:
        content=key.encode();(review/"artifacts"/(key+".pkl")).write_bytes(content)
        hashes[key]=hashlib.sha256(content).hexdigest()
    history=root/"history";(history/"replay").mkdir(parents=True)
    (review/"replay").mkdir()
    def record(ep,done=True):
        return {"episode_id":ep,"done":done,"rewards":np.array([1. if done else 0.]),"source_chunk":np.array([3]),"action_chunk":np.zeros((1,14))}
    for path,records in [(history/"replay/replay_journal.pkl",[record(24)]),(review/"replay/replay_journal.pkl",[record(0),record(1,False)])]:
        with path.open("wb") as f:
            for r in records:pickle.dump(r,f)
    (review/"review_manifest.json").write_text(json.dumps({"actor_version":5416,"global_step":10832,"sha256":hashes,"source_run":str(history)}))
    return root/"runs/plug-online-from5416-20260914"

def test_import_maps_ids_preserves_seed_and_does_not_repeat(tmp_path):
    run=fixture(tmp_path);m=prepare(tmp_path,run)
    assert m["review_episode_mapping"]=={"0":10000}
    cur=json.loads((run/"current.json").read_text())
    assert cur["actor_version"]==5416 and cur["processed"]==[24]
    before=(run/"replay/replay_journal.pkl").read_bytes()
    assert prepare(tmp_path,run)==m
    assert (run/"replay/replay_journal.pkl").read_bytes()==before

def test_bad_seed_refuses_before_run_created(tmp_path):
    run=fixture(tmp_path)
    (tmp_path/"runs/plug-action-contract-review-20260914/artifacts/actor.pkl").write_bytes(b"wrong")
    with pytest.raises(ValueError,match="hash"):prepare(tmp_path,run)
    assert not run.exists()

def test_wrong_destination_rejected(tmp_path):
    with pytest.raises(ValueError):prepare(tmp_path,tmp_path/"wrong")
