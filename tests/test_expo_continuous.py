import json,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))

def test_schedule_is_continuous_and_timeout_resumes_without_batch_credit():
    from continuous_state import Activity
    a=Activity(timeout=120,now=0)
    assert a.reason(64,0) is None
    assert a.reason(64,119) is None
    assert a.reason(64,121)=='waiting_episode'
    a.episode(122)
    assert a.reason(64,122) is None and a.reason(63,122)=='waiting_replay'
    assert Activity(timeout=0,now=0).reason(64,99999) is None

def test_inserted_receipt_is_independent_of_training_commit(tmp_path):
    from continuous_state import ContinuousInbox
    from test_expo_async_state import episode,transfer
    local=tmp_path/'local';i=episode(local);box=ContinuousInbox(tmp_path/'run',lambda:{3},0)
    transfer(box,local,i)
    assert not box.receipt(i)['processed']
    (box.root/'replay-state.json').write_text(json.dumps({'accepted':[i],'replay':[i]}))
    assert box.receipt(i)['processed'] and box.pending()==[] and box.backlog()==[]
    assert not (box.root/'committed.json').exists()

def test_episode_insert_during_optimizer_and_restart_is_idempotent(tmp_path,monkeypatch):
    from continuous_state import ingest
    from test_expo_async_state import episode
    import continuous_state
    lock=threading.RLock();called=[]
    l=SimpleNamespace(root=tmp_path,meta={'accepted':[],'replay':[],'steps':0,'completed_episodes':0},
        replay_lock=lock,seed={'frames':0},args=SimpleNamespace(capacity=100),buffer=object(),
        activity=SimpleNamespace(episode=lambda t:called.append(t)),wake=threading.Event(),save_status=lambda:None)
    monkeypatch.setattr(continuous_state,'insert_episode',lambda *a,**kw:called.append('insert'))
    i=episode(tmp_path/'episodes')
    optimizer=threading.Lock();optimizer.acquire()
    # The optimizer can hold its own lock without blocking ingestion.
    assert ingest(l,i) and not ingest(l,i)
    assert optimizer.locked() and l.meta['steps']==3 and l.meta['completed_episodes']==1
    d=json.loads((tmp_path/'replay-state.json').read_text());assert d['accepted']==[i]
    l.meta=d;assert not ingest(l,i) and called.count('insert')==1
    optimizer.release()

def test_native_batch_processor_is_used_instead_of_outcome_quotas(monkeypatch):
    import continuous_learning
    seen=[]
    class Processor:
        def next_batch(self,rng):seen.append(rng);return 'critic','success','next_rng'
    l=SimpleNamespace(processor=Processor(),combine_rng='rng',replay_lock=threading.RLock())
    assert continuous_learning.sample(l)==('critic','success') and l.combine_rng=='next_rng' and seen==['rng']


def test_recovery_merges_durable_arrivals_after_optimizer_checkpoint(tmp_path,monkeypatch):
    from continuous_learner import ContinuousLearner
    from online_learner import Learner
    l=ContinuousLearner.__new__(ContinuousLearner)
    l.root=tmp_path;l._replay_restored=False
    l.meta=dict(updates=20,checkpoint=7,replay=['old'],accepted=['old'],steps=10,completed_episodes=1)
    (tmp_path/'replay-state.json').write_text(json.dumps(dict(replay=['old','new'],accepted=['old','new'],steps=15,completed_episodes=2)))
    observed=[]
    monkeypatch.setattr(Learner,'rebuild',lambda self:observed.append(dict(self.meta)))
    l.rebuild()
    assert l.meta['updates']==20 and l.meta['checkpoint']==7 and l.meta['replay']==['old','new']
    l.meta['replay'].append('later');l.rebuild()
    assert observed[-1]['replay']==['old','new','later']


def test_checkpoint_retains_later_replay_admission(tmp_path,monkeypatch):
    import continuous_persistence,continuous_checkpoint
    meta=dict(updates=21,checkpoint=7,replay=['old'],accepted=['old'],steps=10,completed_episodes=1)
    l=SimpleNamespace(root=tmp_path,meta_path=tmp_path/'committed.json',meta=meta,error=None,
        replay_lock=threading.RLock(),agent=SimpleNamespace(rng=[1,2]),buffer=SimpleNamespace(rng=[3,4]),
        combine_rng=[5,6],updater=object(),timings={},stop=threading.Event(),wake=threading.Event(),
        save_status=lambda:None,publish_committed=lambda version:None)
    entered=threading.Event();release=threading.Event()
    def write(*_):
        entered.set();assert release.wait(2)
        return dict(timings=dict(checkpoint_write_seconds=.1))
    monkeypatch.setattr(continuous_checkpoint,'capture',lambda *a:dict(capture_seconds=.01,host_bytes=123))
    monkeypatch.setattr(continuous_checkpoint,'write',write)
    assert continuous_persistence.checkpoint(l)
    assert entered.wait(1) and continuous_persistence.busy(l)
    l.meta.update(replay=['old','new'],accepted=['old','new'],steps=15,completed_episodes=2)
    (tmp_path/'replay-state.json').write_text(json.dumps({k:l.meta[k] for k in ('replay','accepted','steps','completed_episodes')}))
    assert not continuous_persistence.checkpoint(l)
    release.set();l.checkpoint_writer.join(2)
    committed=json.loads(l.meta_path.read_text())
    assert committed['replay']==['old'] and l.meta['replay']==['old','new']
    assert l.committed_updates==21 and committed['checkpoint']!=7
    assert json.loads((tmp_path/'replay-state.json').read_text())['accepted']==['old','new']


def test_request_does_not_train_or_add_episode_credit():
    from continuous_learner import ContinuousLearner
    l=ContinuousLearner.__new__(ContinuousLearner)
    l.phase='waiting_episode';l.meta={'updates':20};l.published_version=20
    l.publish_requested=threading.Event();l.wake=threading.Event()
    assert not l.request_publish()['publication_pending']
    l.phase='updating'
    assert l.request_publish()['publication_pending'] and l.wake.is_set()
    assert l.meta=={'updates':20}


def test_continuous_launcher_has_no_batch_budget_or_outcome_prior():
    from remote_launcher import learner_command
    c=dict(python='/bin/python',project='/native',bundle='/bundle',run_dir='/run',port=18443,
        seed_dir='/seed',model_id='test',rtc_delay=5,edit_scale=.01,async_mode='native_async_v1')
    cmd=learner_command(c,Path('/adapter'))
    assert cmd[1]=='/adapter/continuous_learner.py'
    assert cmd[cmd.index('--episodes-per-update')+1]=='1'
    assert '--async-max-pending' not in cmd and '--steps-per-update' not in cmd
    with pytest.raises(ValueError):learner_command(dict(c,critic_prior='demo50'),Path('/adapter'))
    with pytest.raises(ValueError):learner_command(dict(c,episodes_per_update=5),Path('/adapter'))


def test_worker_setup_failure_reports_fault(monkeypatch):
    import continuous_learning,builtins
    original=builtins.__import__
    def fail(name,*a,**kw):
        if name=='expo_ft.data.batch_processor':raise ImportError('broken native dependency')
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',fail)
    saved=[]
    l=SimpleNamespace(error=None,phase='ready',stop=threading.Event(),save_status=lambda:saved.append(True))
    continuous_learning.work(l)
    assert l.phase=='fault' and 'broken native dependency' in l.error and l.stop.is_set() and saved


def test_native_timeout_starts_when_optimizer_observes_finish_event():
    from continuous_state import Activity
    activity=Activity(timeout=120,now=0)
    activity.episode(100)
    assert activity.reason(64,180) is None
    assert activity.reason(64,299) is None
    assert activity.reason(64,301)=='waiting_episode'


def test_shutdown_drains_publication_requested_during_nonpublishing_writer(monkeypatch):
    import continuous_persistence
    l=SimpleNamespace(error=None,meta={'updates':22},committed_updates=22,published_version=20,
        publish_requested=threading.Event())
    l.publish_requested.set();calls=[]
    monkeypatch.setattr(continuous_persistence,'checkpoint',lambda obj,**kw:calls.append(kw))
    continuous_persistence.drain(l)
    assert calls==[dict(publish=True,wait=True)]
