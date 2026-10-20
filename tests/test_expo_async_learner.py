import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"methods/expo_ft/cobot_adapter"))
@pytest.mark.parametrize('prior',['seeded','demo50_success30_failure20'])
def test_async_old_policy_preserves_budget_and_independent_rng(tmp_path,monkeypatch,prior):
    from types import SimpleNamespace
    import jax
    import numpy as np
    from flax import struct
    import async_learning as module
    from async_learner import AsyncLearner
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
    l=AsyncLearner.__new__(AsyncLearner)
    l.root=tmp_path;l.meta_path=tmp_path/'committed.json';l.aligned=True;l.learner_only=True;l.native_batch_update=True
    l.critic_prior=prior;l.args=SimpleNamespace(capacity=500000);l.seed={'frames':3301};l.buffer=Buffer();l.episodes_per_update=5
    l.known_versions=lambda:{0}
    l.publish_committed=lambda:None
    l.timings={}
    l.pending_updates=0
    l.meta=dict(checkpoint=None,updates=0,accepted=[],replay=[],steps=1000,credit=0,actor_updates=0,critic_updates=0,completed_episodes=10,episodes_since_update=0)
    if prior=='demo50_success30_failure20':
        import outcome_sampling
        l.outcome_pools=lambda:tuple(np.arange(10) for _ in range(3))
        monkeypatch.setattr(outcome_sampling,'sample_critic',lambda *args: {'quota':(640,384,256)})
    l.agent=Agent(jax.numpy.asarray([3,4],dtype=jax.numpy.uint32))
    def update(agent,batch,utd,actor_batch):
        if prior=='demo50_success30_failure20':assert batch['quota']==(640,384,256)
        updates.append(utd)
        return agent.replace(critic=SimpleNamespace(step=20),actor_train_state=SimpleNamespace(step=1)),{'loss':1.}
    # The test exercises journal/metadata/checkpoint ordering; numerical updates have separate GPU coverage.
    monkeypatch.setattr(jax,'block_until_ready',lambda x:x)
    l.updater=SimpleNamespace(update=update);l.save_status=lambda:None;l.publish_actor=lambda:published.append(l.meta['updates'])
    for i in range(8):
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
    assert l.meta['completed_episodes']==16 and l.meta['episodes_since_update']==1
    np.testing.assert_array_equal(l.agent.rng,np.asarray([3,4],np.uint32))
    assert len(l.meta['replay'])==6 and l.meta['credit']==6

def test_http_admission_does_not_take_optimizer_lock(tmp_path):
    import threading,json,hashlib
    from types import SimpleNamespace
    from urllib.request import build_opener,ProxyHandler,Request
    from test_expo_async_state import episode
    from async_state import Inbox
    from remote_journal import files
    from async_http import make_server
    local=tmp_path/'local';identity=episode(local)
    lock=threading.Lock();lock.acquire()
    learner=SimpleNamespace(root=tmp_path/'remote',lock=lock,wake=threading.Event(),published_version=3,
        inbox=Inbox(tmp_path/'remote',lambda:{3}),known_versions=lambda:{3},snapshot=lambda:{'ready':True,'phase':'updating'})
    server=make_server(learner,0);threading.Thread(target=server.serve_forever,daemon=True).start()
    op=build_opener(ProxyHandler({}));base='http://127.0.0.1:%d'%server.server_port
    try:
        for path in files(local,identity):
            payload=path.read_bytes();req=Request(base+'/async/journal/'+identity+'/'+path.name,data=payload,headers={'X-Content-SHA256':hashlib.sha256(payload).hexdigest()})
            with op.open(req,timeout=2) as r:assert r.status==200
        req=Request(base+'/async/episode/end',data=json.dumps({'episode_uuid':identity}).encode())
        with op.open(req,timeout=2) as r:receipt=json.load(r)
        assert receipt['received'] and not receipt['processed'] and learner.wake.is_set()
        assert lock.locked()
    finally:lock.release();server.shutdown();server.server_close()
