import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.experiments.supported_runtime import SupportedIndex,load_profile,sha256


def rows():
    return [dict(collection_phase_id=2,episode_id=1,step_id=s,source_chunk=np.array([1,2]),rewards=np.array([0.,0.]if s==0 else[0.,1.]),done=s==1,success=int(s==1))for s in [0,1]]


def write(path,rs):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb')as f:
        for r in rs:pickle.dump(r,f)
    temporary.replace(path)


def batch():return dict(collection_phase_id=np.array([2]),episode_id=np.array([1]),step_id=np.array([0]))


def test_observed_credit_rejects_conflicting_sources_and_unchanged_labels(tmp_path):
    rs=rows();rs[1]['source_chunk']=np.array([2,2]);p=tmp_path/'journal.pkl';write(p,rs)
    idx=SupportedIndex(p,.99,[(2,1)])
    b=idx.attach(batch());assert b['mc_return'][0]==pytest.approx(.99**2)
    assert b['retention_mask'][0] and b['mc_valid'][0]
    np.testing.assert_array_equal(rs[0]['source_chunk'],[1,2])
    rs[1]['source_chunk'][0]=1;write(p,rs)
    with pytest.raises(ValueError,match='conflicting'):idx.attach(batch())


@pytest.mark.parametrize('kind',['missing_terminal','missing_prefix','duplicate','unknown_source','missing_identity'])
def test_index_fail_closed(tmp_path,kind):
    rs=rows();rs[1]['source_chunk']=np.array([2,2]);p=tmp_path/'journal.pkl'
    if kind=='missing_terminal':rs[1]['done']=False
    if kind=='missing_prefix':rs=rs[1:]
    if kind=='duplicate':rs.append(rs[0])
    if kind=='unknown_source':rs[0]['source_chunk'][0]=9
    write(p,rs);idx=SupportedIndex(p,.99,[]);b=batch()
    if kind=='missing_identity':b['episode_id'][0]=9
    with pytest.raises(ValueError):idx.attach(b)


def test_profile_asset_and_publication_identity(tmp_path):
    teacher=tmp_path/'teacher.pkl';teacher.write_bytes(b'immutable')
    norm=tmp_path/'norm.json';norm.write_text('{}')
    profile=dict(schema='held-gripper-retention-v1',teacher=dict(file=teacher.name,sha256=sha256(teacher)),normalization=dict(file=norm.name,sha256=sha256(norm)),mc_weight=.3,retention_weight=50.,actor_q_weight=.01,publication_policy='staged')
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    assert load_profile(path)['teacher']['path']==str(teacher)
    teacher.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):load_profile(path)
    profile['teacher']['sha256']=sha256(teacher);profile['publication_policy']='automatic';path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='staged'):load_profile(path)
