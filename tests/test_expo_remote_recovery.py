import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import build_opener,ProxyHandler
from urllib.error import URLError
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"methods/expo_ft/cobot_adapter"))
from actor_http import serve
from remote_runtime_support import RemoteSupport

class Once:
    def __init__(self):self.calls=0
    def wait(self,seconds):self.calls+=1;return self.calls>1
    def is_set(self):return False

@pytest.mark.parametrize('transferring',[True,False])
def test_transfer_timeout_preserves_local_actor_and_rollout_still_faults(transferring):
    c=RemoteSupport();c.stop=Once();c.ready=True;c.remote_seen=0;c.remote={'local_port':123};c.transfer={'episode_uuid':'test'} if transferring else None
    c.remote_process_lock=threading.Lock();c.remote_process=SimpleNamespace(poll=lambda:None)
    c.remote_status={'actor_version':2};c.failures=[];c.fail=c.failures.append
    class Broken:
        def open(self,*args,**kwargs):raise URLError('temporary interruption')
    c.rpc=Broken();c.remote_poll()
    assert bool(c.failures)==(not transferring)
    assert c.remote_status['phase']==('reconnecting' if transferring else 'fault')

def test_dead_transfer_tunnel_is_restarted_without_stopping_actor(monkeypatch):
    c=RemoteSupport();c.stop=Once();c.ready=True;c.remote_seen=0;c.remote={'local_port':123};c.transfer={'episode_uuid':'test'}
    c.remote_process_lock=threading.Lock();c.remote_process=SimpleNamespace(poll=lambda:1);c.reconnect_after=0
    c.remote_status={};c.fail=lambda error:pytest.fail(error);c.remote_command=lambda:['owned-ssh']
    class Broken:
        def open(self,*args,**kwargs):raise URLError('disconnected')
    c.rpc=Broken();calls=[];new=object()
    monkeypatch.setattr('remote_runtime_support.subprocess.Popen',lambda *args,**kwargs:(calls.append(args),new)[1])
    c.remote_poll();assert c.remote_process is new;assert calls==[(['owned-ssh'],)]

def test_partial_actor_file_resumes_with_range_and_hash(tmp_path):
    payload=bytes(range(256))*16384;source=tmp_path/'source';source.write_bytes(payload)
    record=dict(seed=False,file='actor-0.gz',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    manifest=dict(actor_version=2,records=[record]);ranges=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/actor/manifest':
                body=json.dumps(manifest).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            else:ranges.append(self.headers.get('Range'));serve(self,source,len(payload))
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        c=RemoteSupport();c.remote={'local_port':server.server_port};c.actor_root=tmp_path/'actors';c.rpc=build_opener(ProxyHandler({}))
        directory=c.actor_root/'2';directory.mkdir(parents=True);(directory/'actor-0.download').write_bytes(payload[:1234567])
        result,_=c.download_actor()
        assert ranges==['bytes=1234567-']
        assert (result/'actor-0.gz').read_bytes()==payload
        assert not (result/'actor-0.download').exists()
        c.download_actor();assert len(ranges)==1
    finally:server.shutdown();server.server_close()
