"""Exercise the local episode HTTP adapter without ROS or robot publishers."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from uuid import uuid4
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
from online_runtime import OnlineRuntime,Runtime

@pytest.fixture
def runtime(tmp_path,monkeypatch):
    def initialize(self,args,ros):
        self.args=args;self.ros=ros;self.options={'rtc':True};self.gate=SimpleNamespace(paused=True)
        self.joint_type=object;self.lock=threading.RLock();self.save=lambda:None
    monkeypatch.setattr(Runtime,'__init__',initialize)
    monkeypatch.setattr(Runtime,'close',lambda self:None)
    monkeypatch.setattr(OnlineRuntime,'collect',lambda self:None)
    value=OnlineRuntime(SimpleNamespace(run_dir=tmp_path,model_id='expo-test',control_port=0),SimpleNamespace(Subscriber=lambda *a,**k:None))
    value.events=[]
    value.json_rpc=lambda path,body:value.events.append((path,body)) or {'actor_version':4}
    yield value
    value.close();value.control.server_close()

def post(runtime,path,**body):
    payload={'model_id':'expo-test',**body}
    request=Request('http://127.0.0.1:%d%s'%(runtime.control.server_port,path),data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=3) as response:return json.load(response)

def test_http_episode_pins_version_and_requires_pause(runtime):
    identity=str(uuid4())
    post(runtime,'/episode/start',episode_uuid=identity)
    assert runtime.journal.active['actor_version']==4 and runtime.episode==identity
    runtime.gate.paused=False
    with pytest.raises(HTTPError):post(runtime,'/episode/end',episode_uuid=identity,outcome='success')
    assert runtime.journal.active
    runtime.gate.paused=True
    post(runtime,'/episode/end',episode_uuid=identity,outcome='success')
    post(runtime,'/episode/end',episode_uuid=identity,outcome='success')
    assert runtime.episode is None
    assert json.loads((runtime.journal.root/identity/'finished.json').read_text())['outcome']=='success'
    with pytest.raises(HTTPError):post(runtime,'/episode/end',episode_uuid=identity,outcome='failure')

def test_abandoned_start_without_journal_is_cancelled(runtime):
    identity=str(uuid4())
    post(runtime,'/episode/end',episode_uuid=identity,outcome='aborted')
    assert runtime.events[-1][0]=='/episode/cancel-start'
    assert not (runtime.journal.root/identity).exists()

def test_stale_model_and_episode_do_not_change_recording(runtime):
    identity=str(uuid4());post(runtime,'/episode/start',episode_uuid=identity)
    with pytest.raises(HTTPError):post(runtime,'/episode/pause',episode_uuid=str(uuid4()))
    with pytest.raises(HTTPError):post(runtime,'/episode/pause',episode_uuid=identity,model_id='other')
    assert runtime.journal.recording


@pytest.mark.parametrize('episodes', [1, 5])
def test_remote_runtime_start_checks_local_actor_without_learner_batch(tmp_path, monkeypatch, episodes):
    """Exercise the actual inherited readiness hook that previously rejected batch5."""
    import io
    import ros_runtime
    runtime = OnlineRuntime.__new__(OnlineRuntime)
    runtime.args = SimpleNamespace(port=8043, step=2000, project=tmp_path/'native'/'project',
                                   model_id='expo-test', rtc_delay=5, remote_config=tmp_path/'remote.json')
    runtime.remote = dict(episodes_per_update=episodes, edit_scale=.01)
    local = dict(ready=True, step=2000, deployment_id='expo-test', rtc_delay=5,
                 update_layout='native_full_batch', training_enabled=False, edit_scale=.01, actor_version=11)
    runtime.remote_status = {**local, 'training_enabled':True, 'episodes_per_update':episodes}
    runtime.stop = threading.Event()
    runtime.ready = False
    runtime.transfer = None
    runtime.save = lambda: None
    runtime.policy_command = lambda: ['fake-actor']
    calls = []
    def opened(url, **kwargs):
        calls.append(url)
        return io.StringIO(json.dumps(local))
    runtime.rpc = SimpleNamespace(open=opened)
    class FreePort:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def connect_ex(self, address): return 1
    monkeypatch.setattr(ros_runtime.socket, 'socket', FreePort)
    monkeypatch.setattr(ros_runtime.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(poll=lambda:None))
    runtime.validate_remote_status(runtime.remote_status)
    runtime.start_server()
    assert runtime.ready and runtime.local_actor_version == 11
    assert calls == ['http://127.0.0.1:8043/ready', 'http://127.0.0.1:8043/status']
    with pytest.raises(RuntimeError, match='episode batch size mismatch'):
        runtime.validate_remote_status({**runtime.remote_status, 'episodes_per_update':6})
    for bad in ({**local,'training_enabled':True}, {**local,'deployment_id':'other'},
                {**local,'rtc_delay':8}, {**local,'edit_scale':.05}):
        with pytest.raises(RuntimeError): runtime.validate_server_status(bad)
    with pytest.raises(RuntimeError, match='role mismatch'):
        runtime.validate_remote_status({**runtime.remote_status, 'training_enabled':False})


@pytest.mark.parametrize('path', ['startup', 'poll'])
def test_remote_batch_mismatch_rejected_on_both_status_paths(monkeypatch, path):
    import io
    import remote_runtime_support as support
    remote = support.RemoteSupport()
    remote.remote = dict(episodes_per_update=5, edit_scale=.01)
    remote.args = SimpleNamespace(model_id='expo-test', rtc_delay=5)
    info = dict(deployment_id='expo-test',rtc_delay=5,update_layout='native_full_batch',
                edit_scale=.01,training_enabled=True,episodes_per_update=1,ready=True)
    remote.rpc = SimpleNamespace(open=lambda *a, **kw: io.StringIO(json.dumps(info)))
    waits = iter([False, True])
    remote.stop = SimpleNamespace(wait=lambda delay: next(waits))
    remote.transfer = None
    remote.remote_process_lock = threading.Lock()
    remote.remote_command = lambda: ['fake-learner']
    class FreePort:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def connect_ex(self, address): return 1
    monkeypatch.setattr(support.socket, 'socket', FreePort)
    monkeypatch.setattr(support.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(poll=lambda:None))
    if path == 'startup':
        with pytest.raises(RuntimeError, match='episode batch size mismatch'): remote.start_remote()
    else:
        errors=[]
        remote.fail=errors.append
        remote.remote_poll()
        assert errors == ['Remote learner episode batch size mismatch']
        assert remote.remote_status['phase'] == 'fault'


def test_feedback_execution_provenance_survives_episode_finish(runtime):
    runtime.args.gripper_feedback=True
    identity=str(uuid4());post(runtime,'/episode/start',episode_uuid=identity)
    started=json.loads((runtime.journal.root/identity/'started.json').read_text())
    assert started['execution_profile']=='gripper_feedback_v1'
    assert started['actor_version']==4
    post(runtime,'/episode/end',episode_uuid=identity,outcome='failure')
    finished=json.loads((runtime.journal.root/identity/'finished.json').read_text())
    assert finished['execution_profile']=='gripper_feedback_v1'
    assert finished['outcome']=='failure'


def test_corrected_coordinator_command_is_recorded_without_relimiting(runtime):
    import numpy as np
    runtime.args.gripper_feedback=True
    identity=str(uuid4());post(runtime,'/episode/start',episode_uuid=identity)
    obs={'state':np.array([0.]*6+[.027]),'left_wrist_image':np.ones((180,320,3),np.uint8)*10,'right_wrist_image':np.ones((180,320,3),np.uint8)*20}
    corrected=np.array([0.]*6+[.031],np.float32)
    runtime.journal.sample(obs,corrected,1.,False)
    runtime.journal.sample(obs,corrected,1.1,False)
    with np.load(runtime.journal.root/identity/'000000.npz') as data:
        np.testing.assert_array_equal(data['actions'],corrected)
        assert data['state'][6]==pytest.approx(.027)
    post(runtime,'/episode/end',episode_uuid=identity,outcome='failure')
