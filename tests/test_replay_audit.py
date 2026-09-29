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
