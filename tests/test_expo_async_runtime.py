import json,sys,threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
from async_remote import AsyncRemote
from async_state import Outbox

def runtime(tmp_path):
    c=AsyncRemote();c.args=SimpleNamespace(run_dir=tmp_path,model_id='async-test',port=8043)
    c.remote={'async_max_pending':10,'async_max_policy_lag':2,'learner_host':'HPC'}
    c.remote_status={'ready':True,'phase':'updating','learner_step':30,'actor_version':20,'published_generation':1,'publications':{'0':{'generation':0},'20':{'generation':1}},'pending_episode_ids':[]}
    c.local_actor_version=0;c.ready=True;c.episode=None;c.installing=False;c.staged=None;c.download_state={};c.upload_state={};c.network_error=None
    c.outbox=Outbox(tmp_path/'episodes');c.control_lock=threading.RLock();c.gate=SimpleNamespace(paused=True);c.journal=SimpleNamespace(active=None);c.future=None
    c.actor_root=tmp_path/'actors';c.actor_root.mkdir();c.stop=threading.Event();c.failures=[];c.fail=c.failures.append
    return c

def test_learning_and_transfer_do_not_block_bounded_collection(tmp_path):
    c=runtime(tmp_path);c.download_state={'state':'downloading'}
    v=c.remote_online_status();assert v['ready'] and v['served_actor_version']==0 and v['published_actor_version']==20
    c.remote_status['pending_episode_ids']=[str(uuid4()) for _ in range(10)]
    assert not c.remote_online_status()['ready']
    c.episode=str(uuid4());assert c.remote_online_status()['ready']

def test_new_episode_uses_local_policy_and_independent_rng(tmp_path):
    c=runtime(tmp_path);calls=[]
    c.local_rpc=lambda path,body,**kw:calls.append((path,body)) or {'ready':True,'actor_version':0}
    c.remote_rpc=lambda *a:pytest.fail('Episode start must not wait for remote learner')
    identity=str(uuid4());r=c.json_rpc('/episode/start',{'episode_uuid':identity})
    assert r['actor_version']==0 and calls[-1][1]['actor_version']==0 and len(calls[-1][1]['rollout_rng'])==2

def test_install_waits_for_episode_boundary_and_keeps_service(tmp_path):
    c=runtime(tmp_path);d=c.actor_root/'20';d.mkdir();c.staged=(d,{'actor_version':20})
    calls=[];c.local_rpc=lambda route,body,**kw:calls.append(route) or {'ready':True,'actor_version':20}
    c.episode=str(uuid4());assert not c.maybe_install();assert not calls
    c.episode=None;c.journal.active={'id':'still-finishing'};assert not c.maybe_install()
    c.journal.active=None;c.gate.paused=False;assert not c.maybe_install()
    c.gate.paused=True;assert c.maybe_install()
    assert calls==['/actor/install'] and c.local_actor_version==20 and c.staged is None
    assert not c.installing and not c.failures

def test_failed_install_blocks_new_rollout(tmp_path):
    c=runtime(tmp_path);c.staged=(tmp_path/'actors/20',{'actor_version':20})
    def failed(*a,**kw):raise ValueError('bad snapshot')
    c.local_rpc=failed
    assert not c.maybe_install() and c.failures
    assert not c.remote_online_status()['ready']

def test_optional_catalogue_keeps_sync_and_separate_async_run():
    root=Path(__file__).parents[1]
    models=json.loads((root/'configs/deployment_models.json').read_text())['models']
    sync=next(m for m in models if m['id']=='expo-plug-left-2cam-532-hpc4-2000')
    new=next(m for m in models if m['id']=='expo-plug-left-2cam-532-hpc4-async-2000')
    assert sync['online_run_dir']!=new['online_run_dir']
    assert sync['data_directories']!=new['data_directories']
    assert new['runtime_adapter']=='expo_online' and new['async_mode']=='bounded_async_v1'
    assert 'async_runtime.py' in (root/new['entry_relative']).read_text()
    assert any('async-ready.json' in p for p in new['required'])
