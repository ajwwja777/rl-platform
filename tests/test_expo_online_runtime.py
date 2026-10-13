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
