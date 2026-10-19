import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from outcome_sampling import pools,draw_indices,deferred_schedule
from sync_schedule import Schedule
class Buffer:
    _replan_steps=8
    _seed=42
    def __init__(self,n):self.n=n;self._insert_index=n
    def __len__(self):return self.n

def test_exact_532_and_episode_boundaries():
    b=Buffer(430);p=pools(b,100,[(70,'success'),(200,'failure'),(60,'success')])
    a=draw_indices(b,1280,p).reshape(20,64)
    assert sum(np.isin(a,p[0]).flat)==640
    assert sum(np.isin(a,p[1]).flat)==384
    assert sum(np.isin(a,p[2]).flat)==256
    assert np.all(np.isin(a,p[0]).sum(1)==32)
    assert set(np.isin(a,p[1]).sum(1))=={19,20}
    assert not np.isin(a,list(range(92,100))+list(range(162,170))+list(range(362,370))+list(range(422,430))).any()
    assert b.prior_sampling_stats['online_failure_samples']==256

def test_rng_restore():
    b=Buffer(300);p=pools(b,100,[(100,'success'),(100,'failure')]);draw_indices(b,1280,p);saved=np.array(b.rng)
    expected=draw_indices(b,1280,p);c=Buffer(300);c.rng=saved
    np.testing.assert_array_equal(expected,draw_indices(c,1280,p))

def test_missing_outcome_retains_credit_without_training():
    b=Buffer(200);p=pools(b,100,[(100,'failure')])
    s,n=Schedule(14,1000,120,4).finish(150,episodes_per_update=5)
    assert n==9
    s,n=deferred_schedule(s,n,p)
    assert n==0 and s.credit==270 and s.completed_episodes==15
    with pytest.raises(ValueError,match='awaits'):draw_indices(b,1280,p)
    # Subsequent successful data allows the accumulated budget at next batch boundary.
    p=pools(Buffer(300),100,[(100,'failure'),(100,'success')])
    s,n=Schedule(19,1750,s.credit+600,4).finish(150,episodes_per_update=5)
    before=(s,n);assert deferred_schedule(s,n,p)==before and n==34

@pytest.mark.parametrize('episodes', [[(50,'success')],[(100,'unknown')]])
def test_invalid_identity_rejected(episodes):
    with pytest.raises(ValueError):pools(Buffer(200),100,episodes)

def test_launcher_and_catalogue_fresh_namespace():
    import json
    from remote_launcher import learner_command
    r=Path(__file__).parents[1]
    models=json.loads((r/'configs/deployment_models.json').read_text())['models']
    n=next(m for m in models if m.get('critic_prior')=='demo50_success30_failure20')
    assert n['checkpoint'].endswith('step_2000')
    assert len({m.get('online_run_dir') for m in models if m.get('online_run_dir')==n['online_run_dir']})==1
    assert sum(m.get('online_run_dir')==n['online_run_dir'] for m in models)==1
    c=dict(python='python',project='/source',bundle='/bundle',run_dir='/fresh',port=18444,seed_dir='/seed',model_id=n['id'],rtc_delay=5,critic_prior=n['critic_prior'])
    cmd=learner_command(c,r);assert cmd[cmd.index('--critic-prior')+1]==n['critic_prior']
