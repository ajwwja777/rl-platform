import sys
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from episode_journal import EpisodeJournal,read_journal
from online_protocol import transitions


def obs(value):
    return dict(state=np.full(7,value,np.float32),left_wrist_image=np.full((180,320,3),30,np.uint8),right_wrist_image=np.full((180,320,3),40,np.uint8))


def test_records_executed_command_with_preceding_observation(tmp_path):
    j=EpisodeJournal(tmp_path);identity=str(uuid4());j.begin(identity,7)
    j.sample(obs(1),np.full(7,10),1.0,False)
    j.sample(obs(2),np.full(7,20),1.1,False)
    j.finish(identity,'success');meta,data=read_journal(tmp_path,identity)
    assert np.all(data['state'][0]==1) and np.all(data['actions'][0]==20)
    assert np.all(data['terminal_state']==2)
    assert data['actor_version']==7 and data['dones'].tolist()==[True]
    row=list(transitions(data))[0]
    assert row['rewards']==1 and row['masks']==0 and row['is_success']


def test_hil_and_pause_boundaries_never_create_fake_continuations(tmp_path):
    j=EpisodeJournal(tmp_path);identity=str(uuid4());j.begin(identity,0)
    for i in range(3):j.sample(obs(i),np.full(7,i),1+i*.1,False)
    j.sample(obs(3),np.full(7,3),1.3,True) # HIL transition interval excluded
    j.sample(obs(4),np.full(7,4),1.4,True)
    j.pause(True);j.sample(obs(5),np.full(7,5),1.5,True)
    j.pause(False)
    j.sample(obs(6),np.full(7,6),2.,False);j.sample(obs(7),np.full(7,7),2.1,False)
    j.finish(identity,'success');_,data=read_journal(tmp_path,identity)
    assert data['is_hil'].tolist()==[False,False,True,False]
    assert data['dones'].tolist()==[False,True,True,True]
    rows=list(transitions(data))
    assert sum(x['rewards'] for x in rows)==1
    assert rows[1]['masks']==rows[2]['masks']==1 and rows[-1]['masks']==0
    assert all(x['is_success'] for x in rows) # HIL from successful episode enters native BC pool


def test_retry_idempotency_and_conflicting_outcome(tmp_path):
    j=EpisodeJournal(tmp_path);identity=str(uuid4());j.begin(identity,2)
    first=j.finish(identity,'unknown')
    assert j.finish(identity,'unknown')==first
    with pytest.raises(ValueError):j.finish(identity,'success')
    with pytest.raises(ValueError):j.begin(identity,2)


def test_failed_episode_has_no_positive_reward_or_actor_success(tmp_path):
    j=EpisodeJournal(tmp_path);identity=str(uuid4());j.begin(identity,0)
    j.sample(obs(1),np.zeros(7),1.,True);j.sample(obs(1),np.zeros(7),1.1,True)
    j.finish(identity,'failure');_,data=read_journal(tmp_path,identity)
    row=list(transitions(data))[0]
    assert row['is_hil'] and not row['is_success'] and row['rewards']==0 and row['masks']==0


def test_stale_sampling_gap_is_a_truncation(tmp_path):
    j=EpisodeJournal(tmp_path);identity=str(uuid4());j.begin(identity,0)
    for t in (1.,1.1,2.,2.1):j.sample(obs(1),np.zeros(7),t,False)
    j.finish(identity,'unknown');_,data=read_journal(tmp_path,identity)
    rows=list(transitions(data))
    assert len(rows)==2 and all(x['dones'] and x['masks']==1 and x['rewards']==0 for x in rows)


def test_interrupted_journal_keeps_labels_but_is_not_trainable(tmp_path):
    j=EpisodeJournal(tmp_path);identity=str(uuid4());j.begin(identity,3)
    j.sample(obs(1),np.zeros(7),1.,True);j.sample(obs(2),np.ones(7),1.1,True)
    recovered=EpisodeJournal(tmp_path)
    result=recovered.finish(identity,'success')
    assert result['count']==1 and not result['training_eligible']
    meta,data=read_journal(tmp_path,identity)
    assert not meta['training_eligible'] and 'terminal_state' not in data
