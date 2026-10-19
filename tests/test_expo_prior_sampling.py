import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from prior_sampling import draw_indices

class Buffer:
    _replan_steps=8
    _seed=42
    def __init__(self,n):self.n=n;self._insert_index=n
    def __len__(self):return self.n

def test_every_minibatch_quota_and_no_boundary_crossing():
    b=Buffer(8522)
    indices=draw_indices(b,1280,3301).reshape(20,64)
    assert np.all((indices<3301).sum(axis=1)==32)
    assert np.all(indices[:,:32]+8<3301)
    assert np.all(indices[:,32:]>=3301)
    assert np.all(indices[:,32:]+8<8522)
    assert b.prior_sampling_stats['demo_per_minibatch']==32

def test_rng_resume_and_growth():
    b=Buffer(3401);draw_indices(b,64,3301);saved=np.array(b.rng)
    first=draw_indices(b,128,3301)
    other=Buffer(3401);other.rng=saved
    np.testing.assert_array_equal(first,draw_indices(other,128,3301))
    b.n=100000;b._insert_index=b.n
    assert np.count_nonzero(draw_indices(b,64,3301)<3301)==32

@pytest.mark.parametrize('n,seed,total',[(3301,3301,64),(3309,3301,64),(100,4,64),(100,40,65)])
def test_invalid_pool_or_batch_rejected(n,seed,total):
    with pytest.raises(ValueError):draw_indices(Buffer(n),total,seed)

def test_ring_wrap_rejected():
    b=Buffer(100);b._insert_index=5
    with pytest.raises(ValueError,match='ring'):draw_indices(b,64,40)

def test_launcher_default_unchanged_and_opt_in():
    from remote_launcher import learner_command
    c=dict(python='python',project='/source',bundle='/bundle',run_dir='/run',port=18443,seed_dir='/seed',model_id='test',rtc_delay=5)
    assert '--critic-prior' not in learner_command(c,Path('/adapter'))
    a=learner_command(dict(c,critic_prior='demo50'),Path('/adapter'))
    assert a[a.index('--critic-prior')+1]=='demo50'


def test_independent_continuation_preserves_partial_batch(tmp_path):
    import json,hashlib
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from fork_expo_batch_run import fork
    from migrate_expo_device_layout import digest
    adapter=Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'
    source=tmp_path/'original';source.mkdir()
    meta=dict(checkpoint=121,updates=121,completed_episodes=33,episodes_since_update=1,credit=99,accepted=['ep'],replay=['ep'],steps=100,critic_updates=2420)
    contract=dict(recipe='native_sync',update_layout='native_full_batch',learner_only=True,episodes_per_update=5,adapter_source_sha256=digest(adapter,True))
    for name,value in [('status',dict(phase='ready',pid=2000000000,actor_version=121)),('committed',meta),('contract',contract)]:
        (source/(name+'.json')).write_text(json.dumps(value))
    ep=source/'episodes/ep';ep.mkdir(parents=True);(ep/'finished.json').write_text('{}');(ep/'0.npz').write_bytes(b'data')
    cp=source/'checkpoints/121';cp.mkdir(parents=True);(cp/'weight').write_bytes(b'checkpoint')
    (cp/'manifest.json').write_text(json.dumps(dict(records=[dict(file='weight',sha256=hashlib.sha256(b'checkpoint').hexdigest())])))
    before=(source/'committed.json').read_bytes()
    target=tmp_path/'demo50';fork(source,target,adapter,adapter,5,'demo50')
    assert json.loads((target/'committed.json').read_text())==meta
    assert (source/'committed.json').read_bytes()==before
    assert json.loads((target/'contract.json').read_text())['critic_prior']=='demo50'
    (target/'episodes/ep/finished.json').write_text('{"new":true}')
    assert (ep/'finished.json').read_text()=='{}'
