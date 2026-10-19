"""Durable batch scheduling and comparison against upstream episode triggers."""
import json
from dataclasses import asdict
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from sync_schedule import Schedule

def test_warmup_then_batch_with_restart_and_remainder():
    s=Schedule()
    for _ in range(10):
        s,n=s.finish(150,episodes_per_update=5)
        assert n==0 and s.credit==0 and s.episodes_since_update==0
    for expected in range(1,5):
        s,n=s.finish(31,episodes_per_update=5)
        assert n==0 and s.episodes_since_update==expected
        s=Schedule(**json.loads(json.dumps(asdict(s))))
    s,n=s.finish(31,episodes_per_update=5)
    assert n==5 and s.credit==5 and s.episodes_since_update==0
    for _ in range(4):s,n=s.finish(29,episodes_per_update=5);assert n==0
    s,n=s.finish(29,episodes_per_update=5)
    assert n==5 and s.credit==0

def test_batch_preserves_total_gradient_budget_at_batch_boundaries():
    episode=batch=Schedule();one=five=0
    for index,size in enumerate([73,125,91,154,32]*8,1):
        episode,n=episode.finish(size);one+=n
        batch,n=batch.finish(size,episodes_per_update=5);five+=n
        if index%5==0:
            assert one==five and episode.credit==batch.credit
    assert episode.completed_episodes==batch.completed_episodes==40

@pytest.mark.parametrize('size',[0,-1,1.5,True])
def test_invalid_batch_size_rejected(size):
    with pytest.raises(ValueError):Schedule().finish(10,episodes_per_update=size)

def test_launcher_passes_batch_policy_and_rejects_invalid_values():
    from remote_launcher import learner_command
    c={k:k for k in ('python','project','bundle','run_dir','port','seed_dir','model_id','rtc_delay')}
    c['episodes_per_update']=5
    cmd=learner_command(c,Path('/adapter'))
    assert cmd[cmd.index('--episodes-per-update')+1]=='5'
    c['episodes_per_update']=0
    with pytest.raises(ValueError):learner_command(c,Path('/adapter'))


def test_real_learner_commits_partial_batches_and_ignores_duplicate_receipts(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import jax
    import numpy as np
    from flax import struct
    import online_learner as module
    import stream_checkpoint,actor_snapshot
    @struct.dataclass
    class Agent:
        rng:object
        train_base_actor:bool=True
        critic:object=None
        actor_train_state:object=None
    class Buffer:
        rng=np.asarray([1,2],np.uint32)
        def sample_jax(self,*args,**kwargs):return {}
        def _convert_to_openpi_format(self,x):return x
    journals={};saved=[];published=[];updates=[]
    monkeypatch.setattr(module,'read_journal',lambda root,identity:journals[identity])
    monkeypatch.setattr(module,'insert_episode',lambda *args,**kwargs:None)
    monkeypatch.setattr(stream_checkpoint,'save',lambda *args:saved.append(args[-1]))
    monkeypatch.setattr(actor_snapshot,'prune',lambda *args:None)
    l=module.Learner.__new__(module.Learner)
    l.root=tmp_path;l.meta_path=tmp_path/'committed.json';l.aligned=True;l.learner_only=True;l.native_batch_update=True
    l.critic_prior='seeded';l.args=SimpleNamespace(capacity=500000);l.seed={'frames':3301};l.buffer=Buffer();l.episodes_per_update=5
    l.meta=dict(checkpoint=None,updates=0,accepted=[],replay=[],steps=1000,credit=0,actor_updates=0,critic_updates=0,completed_episodes=10,episodes_since_update=0)
    l.agent=Agent(jax.numpy.asarray([3,4],dtype=jax.numpy.uint32))
    def update(agent,batch,utd,actor_batch):
        updates.append(utd)
        return agent.replace(critic=SimpleNamespace(step=20),actor_train_state=SimpleNamespace(step=1)),{'loss':1.}
    # The test exercises journal/metadata/checkpoint ordering; numerical updates have separate GPU coverage.
    monkeypatch.setattr(jax,'block_until_ready',lambda x:x)
    l.updater=SimpleNamespace(update=update);l.save_status=lambda:None;l.publish_actor=lambda:published.append(l.meta['updates'])
    for i in range(7):
        identity=str(i);outcome='unknown' if i==0 else 'aborted' if i==1 else 'failure'
        data={'actions':np.zeros((6,7),np.float32)}
        info={'actor_version':0,'rollout_rng':[7,i],'training_eligible':True,'outcome':outcome,'count':6}
        journals[identity]=(info,data)
        directory=tmp_path/'episodes'/identity;directory.mkdir(parents=True)
        (directory/'finished.json').write_text(json.dumps(info))
        l.learn_episode(identity)
        committed=json.loads(l.meta_path.read_text())
        assert identity in committed['accepted']
        l.learn_episode(identity);assert l.meta==committed
        if i<6:assert not updates and not saved and not published
        if i==4:
            # Simulate process restart from durable metadata between eligible episodes 3 and 4.
            assert committed['episodes_since_update']==3 and committed['credit']==18
            l.meta=committed
    assert updates==[20] and saved==[1] and published==[1]
    assert l.meta['completed_episodes']==15 and l.meta['episodes_since_update']==0
    assert len(l.meta['replay'])==5 and l.meta['credit']==0
