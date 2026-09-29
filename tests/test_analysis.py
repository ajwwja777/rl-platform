import importlib.util
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.cobot_runtime.analysis import segments, episode_groups, snapshot

def test_restarts_do_not_join_curves():
    rows=[{"global_step":x} for x in [1,2,3,2,3,3,4]]
    assert [[x["global_step"] for x in g] for g in segments(rows)]==[[1,2,3],[2,3],[3,4]]

def test_assisted_and_uncommitted_are_not_autonomous_failures():
    def row(e,success,written,hil=0):
        return dict(episode_id=e,timestamp=e,success=success,transitions_written=written,
            intervention_count=hil,actor_version_start=2,actor_version_end=2,
            collection_phase="online",actor_deterministic=False)
    values=[row(1,1,10),row(2,1,10,1),row(3,0,10),row(4,0,0)]
    groups,accepted,excluded=episode_groups(values+[values[0]])
    assert excluded==1 and len(accepted)==3
    assert groups[0]["count"]==3
    assert groups[0]["autonomous_successes"]==1
    assert groups[0]["assisted_successes"]==1
    assert groups[0]["failures"]==1
    assert groups[0]["autonomous_ci"][0]<1/3<groups[0]["autonomous_ci"][1]

def test_missing_and_partial_telemetry(tmp_path):
    metrics=tmp_path/"online/metrics";metrics.mkdir(parents=True)
    (metrics/"learner_metrics.jsonl").write_text('{"global_step":1,"actor_loss":0,"did_actor_update":0}\n{"global_step":2,"actor_loss":-1,"did_actor_update":1}\n{"unfinished":')
    result=snapshot(tmp_path,tmp_path/"absent.yaml")
    assert result["stale"] and len(result["actor_series"])==1
    assert result["actor_series"][0]["actor_loss"]==-1
    assert result["sources"][0]["invalid_lines"]==1
    assert result["config_error"] and result["episode_count"]==0

def test_projection_preserves_journal_and_identity(tmp_path):
    np=pytest.importorskip("numpy")
    import pickle,hashlib
    spec=importlib.util.spec_from_file_location("replay_analysis",ROOT/"scripts/analyze_replay.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/"journal.pkl"
    with path.open("wb") as f:
        for i in range(20):
            pickle.dump(dict(proprio=np.ones(7)*i,action_chunk=np.ones((10,7))*i,
                ref_chunk=np.zeros((10,7)),source=1,episode_id=i//2,step_id=i%2,
                collection_phase="online",intervention_flag=False),f)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    a=module.build(path,tmp_path/"projection.json")
    b=module.build(path,tmp_path/"other.json")
    assert a["points"]==b["points"] and a["transitions"]==20
    assert len(a["clusters"])==6 and sum(x["count"] for x in a["clusters"])==20
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert a["episode_chunks"]["0"]["step_id"]==0
    assert a["explained_variance"][0]>.99
