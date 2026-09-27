import numpy as np
from methods.openpi_rlt.plug_v2.critic_targets import actor_q_mask,training_target

def values():
    return {'rewards':np.zeros(16,np.float32),'done':False,'duration':10,'td_valid':True}

def test_td_target_is_unchanged():
    source=values();out=training_target(source,success=True,mode='td')
    assert out is not source
    np.testing.assert_array_equal(out['rewards'],source['rewards'])
    assert out['done'] is False and out['duration']==10

def test_mc_success_makes_every_factual_chunk_a_terminal_return():
    out=training_target(values(),success=True,mode='mc_success')
    assert out['rewards'][0]==1 and out['rewards'][1:].sum()==0
    assert out['done'] is True and out['duration']==1 and out['td_valid'] is True
    fail=training_target(values(),success=False,mode='mc_success')
    assert fail['rewards'].sum()==0 and fail['done'] is True

def test_unknown_mode_rejected():
    import pytest
    with pytest.raises(ValueError):training_target(values(),success=True,mode='bad')


def test_actor_q_mask_uses_terminal_or_hil_only():
    done=np.asarray([False,True,False])
    source=np.asarray([[1]*10,[1]*10,[1]*5+[2]*5],dtype=np.uint8)
    np.testing.assert_array_equal(actor_q_mask(done,source),[False,True,True])
