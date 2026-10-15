import numpy as np
import pytest
from methods.openpi_rlt.realtime_expo.replay_seed import episode_transitions,normalization

def fixture(n=23):
    identity=dict(native_id=3,uuid='expert3',outcome='success',outcome_source='user_confirmation_2026-10-15',segments=[[2,n+2]])
    state=np.tile(np.arange(7,dtype=np.float32),(n,1));action=np.tile(state[:,None,:],(1,50,1));action[:,:,:6]+=.1
    p=dict(source_frame_indices=np.arange(2,n+2),state=state,z_rl=np.zeros((n,2048)),predicted=action.copy(),target=action.copy(),future_valid=np.arange(50)[None,:]<n-np.arange(n)[:,None])
    return identity,p

def test_terminal_next_state_and_no_padding_or_reference_replacement():
    i,p=fixture();p['predicted']+=.2;rows=episode_transitions(i,p)
    assert [r['step_id'] for r in rows]==[2,12,15]
    assert sum(r['done'] for r in rows)==1
    assert sum(float(r['rewards'].sum()) for r in rows)==1
    assert all(r['episode_id']==-3 and not r['autonomous_success'] for r in rows)
    np.testing.assert_array_equal(rows[0]['next_proprio'],p['state'][10])
    np.testing.assert_array_equal(rows[-1]['next_proprio'],p['state'][-1])
    np.testing.assert_allclose(rows[0]['ref_chunk']-rows[0]['action_chunk'],.2,atol=1e-6)
    norm=normalization(rows)['norm_stats']['actions']['mean']
    np.testing.assert_allclose(norm[:6],.1,atol=1e-6);assert norm[6]==6

@pytest.mark.parametrize('bad',['gap','multiple','outcome','padding'])
def test_no_invented_terminal_or_padded_expert(bad):
    i,p=fixture()
    if bad=='gap':p['source_frame_indices'][3]+=1
    if bad=='multiple':i['segments']=[[2,12],[13,25]]
    if bad=='outcome':i['outcome']='unknown'
    if bad=='padding':p['future_valid'][0,9]=False
    with pytest.raises(ValueError):episode_transitions(i,p)
