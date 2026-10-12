import copy
import numpy as np
import pytest
from methods.openpi_rlt.experiments.journal_import import remap_collection


def fixture():
    old=[dict(collection_phase_id=2,episode_id=10010,step_id=0)]
    new=[dict(collection_phase_id=2,episode_id=i,step_id=0,action_chunk=np.ones((10,7),np.float32)) for i in (2,3)]
    manifest=[dict(source_namespace='new',original_native_episode=i,experimental_episode=10011+j,split=s,transitions=1) for j,(i,s) in enumerate([(2,'TRAIN'),(3,'DEV')])]
    return old,new,manifest


def test_remap_preserves_values_sources_and_partition():
    old,new,m=fixture(); original=copy.deepcopy(new)
    tr,dv=remap_collection(old,new,m)
    assert tr[0]['episode_id']==10011 and dv[0]['episode_id']==10012
    for a,b in zip(new,original):
        assert a['episode_id']==b['episode_id'];np.testing.assert_array_equal(a['action_chunk'],b['action_chunk'])
    np.testing.assert_array_equal(tr[0]['action_chunk'],new[0]['action_chunk'])
    tr[0]['action_chunk'][0,0]=9
    assert new[0]['action_chunk'][0,0]==1


@pytest.mark.parametrize('error',['duplicate','missing','old_id','count','phase','same_new_id'])
def test_invalid_import_rejected(error):
    old,new,m=fixture()
    if error=='duplicate':new.append(new[0])
    elif error=='missing':m.pop()
    elif error=='old_id':m[0]['experimental_episode']=3
    elif error=='count':m[0]['transitions']=2
    elif error=='phase':new[0]['collection_phase_id']=1
    elif error=='same_new_id':m[1]['experimental_episode']=m[0]['experimental_episode']
    with pytest.raises(ValueError):remap_collection(old,new,m)
