import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"methods/expo_ft/cobot_adapter"))
from episode_journal import EpisodeJournal,read_journal
from remote_journal import receive,files


def fixture(root):
    identity=str(uuid4());journal=EpisodeJournal(root);journal.begin(identity,3)
    obs=dict(state=np.ones(7,np.float32),left_wrist_image=np.full((180,320,3),40,np.uint8),right_wrist_image=np.full((180,320,3),50,np.uint8))
    for i in range(6):journal.sample(obs,np.ones(7,np.float32)*i,1+i*.1,i>=3)
    journal.finish(identity,'success')
    return identity


def put(root,identity,path,active=None):
    payload=path.read_bytes()
    return receive(root,active or identity,[],identity,path.name,payload,hashlib.sha256(payload).hexdigest())


def test_lossless_transfer_retry_and_hil(tmp_path):
    local=tmp_path/'local';remote=tmp_path/'remote';identity=fixture(local)
    for p in files(local,identity):
        put(remote,identity,p)
        assert put(remote,identity,p)['duplicate']
    lm,ld=read_journal(local,identity);rm,rd=read_journal(remote,identity)
    assert lm==rm
    for k in ld:np.testing.assert_array_equal(ld[k],rd[k])
    assert ld['is_hil'].tolist()==[False,False,True,True]


def test_incomplete_transfer_cannot_commit(tmp_path):
    identity=fixture(tmp_path/'local');paths=files(tmp_path/'local',identity)
    put(tmp_path/'remote',identity,paths[0])
    with pytest.raises(ValueError,match='Incomplete'):put(tmp_path/'remote',identity,paths[-1])
    assert not (tmp_path/'remote'/identity/'finished.json').exists()


def test_wrong_identity_hash_path_and_mutation_rejected(tmp_path):
    identity=fixture(tmp_path/'local');p=files(tmp_path/'local',identity)[0];payload=p.read_bytes();digest=hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError):receive(tmp_path/'remote',str(uuid4()),[],identity,p.name,payload,digest)
    with pytest.raises(ValueError):receive(tmp_path/'remote',identity,[],identity,'../escape',payload,digest)
    with pytest.raises(ValueError):receive(tmp_path/'remote',identity,[],identity,p.name,payload,'0'*64)
    put(tmp_path/'remote',identity,p)
    changed=json.loads(payload);changed['actor_version']=100;data=json.dumps(changed).encode()
    with pytest.raises(ValueError,match='immutable'):receive(tmp_path/'remote',identity,[],identity,p.name,data,hashlib.sha256(data).hexdigest())


def test_already_committed_episode_is_not_reinserted(tmp_path):
    identity=str(uuid4())
    assert receive(tmp_path,None,[identity],identity,'finished.json',b'old','unused')=={'accepted':True}
    assert not list(tmp_path.iterdir())
