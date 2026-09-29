import pickle
from pathlib import Path
import numpy as np
from integrations.cobot_runtime.replay_audit import annotate, metadata, composition, JournalIndex
from scripts.diagnose_online_learning import episode_split, choose

def row(episode,step,done=False,success=0,phase="online",source=1):
    return dict(episode_id=episode,step_id=step,collection_phase=phase,source=source,
                source_chunk=np.array([source]*10),done=done,success=success)

def test_episode_success_is_not_terminal_transition_ratio():
    rows=annotate([metadata(row(1,0)),metadata(row(1,10,True,1)),
                   metadata(row(2,0,True,0)),metadata(row(3,0))])
    assert [x["outcome"] for x in rows]==["success","success","failure","unknown"]
    assert composition(rows)["dimensions"]["outcome"][1]["count"]==2

def test_identity_resolves_phase_and_duplicate_samples(tmp_path):
    path=tmp_path/"journal.pkl"
    with path.open("wb") as f:
        for r in [row(1,0,True,1),row(2,0,True,0)]:pickle.dump(r,f)
    index=JournalIndex(path)
    result=index.batch(dict(episode_id=np.array([1,1,2]),step_id=np.array([0,0,0]),
                            collection_phase_id=np.array([2,2,2]),source=np.array([1,1,1])))
    assert result["unique_transitions"]==2
    assert result["dimensions"]["outcome"]==[dict(label="failure",count=1,ratio=1/3),dict(label="success",count=2,ratio=2/3)]
    with path.open("ab") as f:pickle.dump(row(3,0,True,1),f)
    index.refresh()
    assert len(index.rows)==3

def test_split_keeps_episodes_together_and_warmup_in_training():
    rows=annotate([metadata(row(ep,step,step==10,ep%2,phase))
        for phase in ("warmup","online") for ep in range(20) for step in (0,10)])
    train,val,held=episode_split(rows)
    assert len(val)>0 and not set(train)&set(val)
    assert all(rows[i]["phase"]=="online" for i in val)
    assert not {(rows[i]["phase"],rows[i]["episode_id"]) for i in train}&set(held)
    assert len(train)+len(val)==len(rows)

def test_sampling_counts_are_explicit_and_reproducible():
    ids=np.arange(100);success=ids<80
    a=choose(np.random.default_rng(42),ids,success,.7)
    assert len(a)==128 and success[a].sum()==90
    assert np.array_equal(a,choose(np.random.default_rng(42),ids,success,.7))

def test_instrumentation_does_not_resample_or_change_training(tmp_path,monkeypatch):
    import sys,types,json
    from integrations.cobot_runtime.replay_audit import install_batch_audit
    journal=tmp_path/"replay.pkl"
    with journal.open("wb") as f:pickle.dump(row(1,0,True,1),f)
    batch=dict(episode_id=np.array([1]),step_id=np.array([0]),collection_phase_id=np.array([2]),source=np.array([1]))
    class Source:
        calls=0
        def sample_batch(self,n):
            self.calls+=1
            return batch
    class Service:
        def __init__(self):
            self._replay_source=Source();self._metrics_path=str(tmp_path/"run/online/metrics/learner_metrics.jsonl")
            Path(self._metrics_path).parent.mkdir(parents=True)
        def train_once(self):
            self.seen=self._replay_source.sample_batch(1)
            return dict(global_step=1,actor_version=0)
    fake=types.ModuleType("rlt_online_rl");fake.trainer=types.SimpleNamespace(LearnerService=Service)
    monkeypatch.setitem(sys.modules,"rlt_online_rl",fake)
    config=tmp_path/"config.yaml"
    config.write_text("runtime:\n  replay:\n    journal_path: "+str(journal)+"\n")
    service=Service();source=service._replay_source
    install_batch_audit(config)
    result=service.train_once()
    assert service._replay_source is source and source.calls==1 and service.seen is batch
    report=json.loads((tmp_path/"run/online/metrics/batch_composition.jsonl").read_text())
    assert report["global_step"]==result["global_step"]
    assert report["dimensions"]["outcome"][0]["label"]=="success"

    snapshot=json.loads((tmp_path/"run/analysis/replay_composition.json").read_text())
    assert snapshot["transitions"]["count"]==1
    first_time=(tmp_path/"run/analysis/replay_composition.json").stat().st_mtime_ns
    service.train_once()
    assert source.calls==2
    assert (tmp_path/"run/analysis/replay_composition.json").stat().st_mtime_ns==first_time

def test_version_audit_keeps_held_episodes_when_journal_grows():
    rows=annotate([metadata(row(ep,0,True,ep%2)) for ep in range(20)])
    _,original_val,held=episode_split(rows)
    expanded=annotate([metadata(row(ep,0,True,ep%2)) for ep in range(40)])
    _,new_val,new_held=episode_split(expanded,held_episodes=held)
    assert held==new_held
    assert [rows[i]["episode_id"] for i in original_val]==[expanded[i]["episode_id"] for i in new_val]
