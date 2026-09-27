from __future__ import annotations
import json
from pathlib import Path
import pytest

def test_preflight_rejects_normal_mode_and_stale_cameras():
    from methods.openpi_rlt.scripts.backend_lifecycle import console_preflight, BackendLifecycleError
    values = {
        "/api/console/identity": {"service": "cobot-data-console-v1"},
        "/api/console/status": {"selected_mode": "normal", "active_mode": None, "ros_readiness": {"status": "ok"}},
    }
    get = lambda path: values[path]
    with pytest.raises(BackendLifecycleError, match="select_rlt_mode"):
        console_preflight(get)
    values["/api/console/status"]["selected_mode"] = "rlt"
    values["/api/console/status"]["ros_readiness"] = {"status": "not_ready", "error_code": "camera_stale"}
    with pytest.raises(BackendLifecycleError, match="camera_stale"):
        console_preflight(get)

def test_preflight_accepts_only_idle_unified_console():
    from methods.openpi_rlt.scripts.backend_lifecycle import console_preflight, BackendLifecycleError
    values = {
        "/api/console/identity": {"service": "cobot-data-console-v1"},
        "/api/console/status": {"selected_mode": "rlt", "active_mode": None, "ros_readiness": {"status": "ok"}},
    }
    console_preflight(lambda path: values[path])
    values["/api/console/status"]["active_mode"] = "normal"
    with pytest.raises(BackendLifecycleError, match="writer_busy"):
        console_preflight(lambda path: values[path])

def test_no_broad_kill_and_continuation_is_default():
    root = Path(__file__).resolve().parents[1]
    text = (root/"scripts/backend_lifecycle.py").read_text()
    assert "interface_task2_teach_rlt_continue5416.sh" in text
    assert "validate_process" in text
    assert "SIGTERM" in text
    assert "pkill" not in text and "killall" not in text
    assert "rlt_model_server.sh" in text
    assert "shutdown_via_http" in text

def test_start_wait_preserves_loading_without_starting_session(monkeypatch,tmp_path):
    from methods.openpi_rlt.scripts import backend_lifecycle as lifecycle
    calls=[]
    class FakeProcess:
        pid=4242
        def poll(self):return None
    monkeypatch.setattr(lifecycle,"http_json",lambda base,path: {
        "service":"cobot-data-console-v1"} if path.endswith("identity") else {
        "selected_mode":"rlt","active_mode":None,"ros_readiness":{"status":"ok"}})
    monkeypatch.setattr(lifecycle,"port_open",lambda port:False)
    monkeypatch.setattr(lifecycle,"process_start_ticks",lambda pid:123)
    monkeypatch.setattr(lifecycle.subprocess,"Popen",lambda command,**kwargs: (calls.append(command) or FakeProcess()))
    state=lifecycle.start_backend(tmp_path/"state.json",Path("/test/interface.sh"),4000,"eval",[],wait_seconds=0)
    assert state.phase=="loading_machine_a"
    assert calls[0][-3:]==["--managed-backend","--lifecycle-state",str(tmp_path/"state.json")]
    assert not any("session/start" in str(command) for command in calls)

def test_stop_missing_state_is_idempotent_and_checks_ports(monkeypatch,tmp_path):
    from methods.openpi_rlt.scripts import backend_lifecycle as lifecycle
    calls=[]
    monkeypatch.setattr(lifecycle.subprocess,"run",lambda command,**kw:calls.append(command))
    monkeypatch.setattr(lifecycle,"port_open",lambda port:False)
    lifecycle.stop_backend(tmp_path/"state.json",Path("/test/rlt_model_server.sh"),4000)
    assert calls==[["bash","/test/rlt_model_server.sh","stop","4000"]]

def test_real_registered_shell_can_start_and_stop_without_robot_services(monkeypatch,tmp_path):
    from methods.openpi_rlt.scripts import backend_lifecycle as lifecycle
    import os, signal, time
    script=tmp_path/"fake_interface.sh"
    script.write_text("trap 'exit 0' TERM\nwhile true; do sleep 0.1; done\n")
    monkeypatch.setattr(lifecycle,"http_json",lambda base,path: {
        "service":"cobot-data-console-v1"} if path.endswith("identity") else {
        "selected_mode":"rlt","active_mode":None,"ros_readiness":{"status":"ok"}})
    monkeypatch.setattr(lifecycle,"port_open",lambda port:False)
    monkeypatch.setattr(lifecycle.subprocess,"run",lambda *args,**kwargs:None)
    state_path=tmp_path/"state.json"
    state=lifecycle.start_backend(state_path,script,4000,"eval",[],wait_seconds=0)
    try:
        reused=lifecycle.start_backend(state_path,script,4000,"eval",[],wait_seconds=0)
        assert reused.supervisor_pid==state.supervisor_pid
        lifecycle.stop_backend(state_path,tmp_path/"fake_model.sh",4000,timeout_seconds=3)
        assert not lifecycle.process_alive(state.supervisor_pid,state.supervisor_start_ticks)
        assert lifecycle.read_lifecycle(state_path).phase=="offline"
    finally:
        if lifecycle.process_alive(state.supervisor_pid,state.supervisor_start_ticks):
            os.kill(state.supervisor_pid,signal.SIGTERM)
