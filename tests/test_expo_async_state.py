import hashlib,json,sys
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from episode_journal import EpisodeJournal
from remote_journal import files
from async_state import Inbox,Outbox,rollout_rng,collection_gate

def episode(root,version=3,outcome='success'):
    identity=str(uuid4());j=EpisodeJournal(root);j.begin(identity,version)
    obs=dict(state=np.ones(7,np.float32),left_wrist_image=np.zeros((180,320,3),np.uint8),right_wrist_image=np.zeros((180,320,3),np.uint8))
    for i in range(4):j.sample(obs,np.zeros(7,np.float32),1+i*.1,False)
    j.finish(identity,outcome)
    return identity

def transfer(box,local,identity):
    for p in files(local,identity):
        b=p.read_bytes();box.receive(identity,p.name,b,hashlib.sha256(b).hexdigest())
    return box.admit(identity)

def test_durable_admission_is_separate_from_training_and_idempotent(tmp_path):
    local=tmp_path/'local';i=episode(local);root=tmp_path/'remote'
    box=Inbox(root,known_versions=lambda:{3});r=transfer(box,local,i)
    assert r['received'] and not r['processed'] and r['eligible']
    assert box.pending()==[i]
    box=Inbox(root,known_versions=lambda:{3});assert transfer(box,local,i)==r
    (root/'committed.json').write_text(json.dumps({'accepted':[i]}))
    assert box.receipt(i)['processed'] and box.pending()==[]

def test_partial_future_and_mutated_journals_rejected(tmp_path):
    local=tmp_path/'local';i=episode(local,4);box=Inbox(tmp_path/'remote',known_versions=lambda:{3})
    with pytest.raises(ValueError,match='published'):transfer(box,local,i)
    i=episode(local,3);ps=files(local,i);p=ps[0];b=p.read_bytes();box.receive(i,p.name,b,hashlib.sha256(b).hexdigest())
    with pytest.raises(ValueError):box.admit(i)
    with pytest.raises(ValueError):
        b=ps[-1].read_bytes();box.receive(i,'finished.json',b,hashlib.sha256(b).hexdigest())
    transfer(box,local,i)
    b=b'changed'
    with pytest.raises(ValueError):box.receive(i,'started.json',b,hashlib.sha256(b).hexdigest())

def test_bound_and_outbox_recovery(tmp_path):
    local=tmp_path/'local';a=episode(local);b=episode(local);box=Inbox(tmp_path/'remote',known_versions=lambda:{3},max_pending=1)
    transfer(box,local,a)
    with pytest.raises(ValueError,match='backlog'):transfer(box,local,b)
    out=Outbox(local);assert out.pending()==[a,b]
    r=box.receipt(a);out.ack(a,r);assert Outbox(local).pending()==[b]
    with pytest.raises(ValueError):out.ack(b,r)

def test_unknown_is_received_but_never_eligible(tmp_path):
    local=tmp_path/'local';i=episode(local,outcome='unknown');box=Inbox(tmp_path/'remote',known_versions=lambda:{3})
    assert not transfer(box,local,i)['eligible']

def test_rng_independent_and_gate_keeps_active_episode_alive():
    i=str(uuid4());a=rollout_rng('run',i)
    assert a==rollout_rng('run',i) and a!=rollout_rng('run',str(uuid4()))
    assert len(a)==2 and all(0<=x<2**32 for x in a)
    assert collection_gate(9,1,False,10,2) is None
    assert collection_gate(10,1,False,10,2)=='backlog'
    assert collection_gate(1,2,False,10,2)=='policy_lag'
    assert collection_gate(100,100,True,10,2) is None

def test_partial_batch_counts_towards_untrained_bound(tmp_path):
    local=tmp_path/'local';root=tmp_path/'remote';box=Inbox(root,lambda:{3},max_pending=2)
    a=episode(local);transfer(box,local,a)
    (root/'committed.json').write_text(json.dumps({'accepted':[a],'replay':[a],'episodes_since_update':1}))
    assert box.pending()==[] and box.backlog()==[a]
    b=episode(local);transfer(box,local,b)
    c=episode(local)
    with pytest.raises(ValueError,match='backlog'):transfer(box,local,c)

def test_failed_admission_write_cannot_ack_durability(tmp_path,monkeypatch):
    import async_state
    local=tmp_path/'local';root=tmp_path/'remote';i=episode(local);box=Inbox(root,lambda:{3})
    for p in files(local,i):
        b=p.read_bytes();box.receive(i,p.name,b,hashlib.sha256(b).hexdigest())
    def fail(*a):raise OSError('disk failure')
    monkeypatch.setattr(async_state,'atomic_json',fail)
    with pytest.raises(OSError):box.admit(i)
    assert not box.receipt(i)['received']


def test_startup_seals_finished_journal_without_actor_receipt(tmp_path):
    local=tmp_path/'local';i=episode(local);out=Outbox(local)
    assert 'rollout_rng' not in json.loads((local/i/'finished.json').read_text())
    out.recover_finished()
    meta=json.loads((local/i/'finished.json').read_text())
    assert meta['actor_receipt_missing_after_restart'] is True
    assert 'rollout_rng' not in meta and meta['outcome']=='success'
    assert out.pending()==[i]

def test_ack_stays_counted_until_status_observes_admission_revision(tmp_path):
    local=tmp_path/'local';i=episode(local);box=Inbox(tmp_path/'remote',lambda:{3})
    stale=box.transport_status();receipt=transfer(box,local,i);out=Outbox(local);out.ack(i,receipt)
    assert out.pending()==[] and out.unconfirmed(stale['admission_revision'])==[i]
    assert Outbox(local).unconfirmed(stale['admission_revision'])==[i]
    fresh=box.transport_status()
    assert i in fresh['pending_episode_ids'] and out.unconfirmed(fresh['admission_revision'])==[]

def test_deferred_learning_ids_remain_in_bound(tmp_path):
    local=tmp_path/'local';i=episode(local);box=Inbox(tmp_path/'remote',lambda:{3});transfer(box,local,i)
    (box.root/'committed.json').write_text(json.dumps({'accepted':[i],'replay':[i],
        'episodes_since_update':0,'credit':150,'async_untrained':[i]}))
    assert box.backlog()==[i]


def test_unpublished_snapshot_from_interrupted_commit_is_retired(tmp_path):
    from async_state import retire_uncommitted_snapshots
    (tmp_path/'committed.json').write_text(json.dumps({'updates':20}))
    (tmp_path/'publications.json').write_text(json.dumps({'20':{'generation':0}}))
    for version in [20,25]:
        p=tmp_path/'actors'/str(version);p.mkdir(parents=True);(p/'manifest.json').write_text('snapshot')
    retired=retire_uncommitted_snapshots(tmp_path)
    assert len(retired)==1 and (tmp_path/'actors/20/manifest.json').exists()
    assert not (tmp_path/'actors/25').exists() and Path(retired[0]).joinpath('manifest.json').read_text()=='snapshot'
    assert retire_uncommitted_snapshots(tmp_path)==[]

def test_published_snapshot_ahead_of_commit_is_not_silently_retired(tmp_path):
    from async_state import retire_uncommitted_snapshots
    (tmp_path/'committed.json').write_text(json.dumps({'updates':20}))
    (tmp_path/'publications.json').write_text(json.dumps({'25':{'generation':1}}))
    p=tmp_path/'actors/25';p.mkdir(parents=True)
    with pytest.raises(ValueError,match='ahead'):retire_uncommitted_snapshots(tmp_path)
    assert p.exists()
