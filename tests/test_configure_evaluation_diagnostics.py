import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("status,success", [
    ({"phase":"offline","process_started":False,"model_ready":False,"ready_confirmed":True},True),
    ({"phase":"ready","process_started":True,"model_ready":True},False),
    ({"phase":"offline","process_started":False},False)])
def test_real_cli_only_get_and_rejects_active_or_unknown_runtime(tmp_path,status,success):
    methods=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            methods.append("GET")
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        def log_message(self,*args): pass
    server=HTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    runtime=tmp_path/"runtime"
    env=dict(os.environ,COBOT_RUNTIME_ROOT=str(runtime))
    env.pop("COBOT_RLT_TRACE_DIR",None)
    try:
        args=[sys.executable,str(ROOT/"scripts/configure_evaluation_diagnostics.py"),"--enable",
            "--trace-root",str(tmp_path/"numeric"),"--status-url",
            "http://127.0.0.1:%d/status"%server.server_port]
        result=subprocess.run(args,env=env,capture_output=True,text=True)
        assert (result.returncode==0)==success,result.stderr
        assert methods==["GET"]
        target=runtime/"evaluation-diagnostic.json"
        assert target.exists()==success
        if success:
            assert json.loads(target.read_text())["enabled"]
            assert not (tmp_path/"numeric").exists() # Configuring never starts capture.
            check=subprocess.run(args+["--check-only"],env=env,capture_output=True,text=True)
            assert check.returncode==0
            disable=args[:2]+["--disable","--status-url",args[-1]]
            result=subprocess.run(disable,env=env,capture_output=True,text=True)
            assert result.returncode==0
            assert not json.loads(target.read_text())["enabled"]
            assert methods==["GET"]*3
    finally:
        server.shutdown();server.server_close();thread.join()
