"""Contract tests for the dedicated continuous Task5 recorder (no ROS)."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

MODULE = Path(__file__).parents[1] / "scripts/rollout_recorder_app.py"

def load():
    import sys
    if str(MODULE.parent) not in sys.path:
        sys.path.insert(0, str(MODULE.parent))
    assert MODULE.is_file(), "RLT needs a continuous recorder app, not segmented-teach"
    spec = importlib.util.spec_from_file_location("rollout_recorder_app", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_unique_ros_node():
    calls = []
    fake = SimpleNamespace(init_node=lambda *a, **k: calls.append((a,k)), core="core")
    proxy = load().RltRosProxy(fake)
    proxy.init_node("capture_core_recorder", anonymous=False, disable_signals=True)
    assert calls[0][0] == ("task5_rlt_rollout_recorder",)
    assert proxy.core == "core"

def test_continuous_recorder_contract(tmp_path):
    module = load()
    app = module.create_rollout_app(data_root=tmp_path, min_free_bytes=1)
    schema = app.openapi()
    paths = schema["paths"]
    for path, method in [
        ("/api/storage/prepare","post"),("/api/episodes/start","post"),
        ("/api/status","get"),("/api/episodes/stop","post"),
        ("/api/episodes","get"),("/api/episodes/{episode_uuid}/outcome","post"),
        ("/api/episodes/{episode_uuid}/labels","put")]:
        assert method in paths[path]
    assert "/api/segmented-teach/start" not in paths
    with pytest.raises(ValueError):
        module.create_rollout_app(data_root=Path("relative"))

def test_wrong_backend_fails_before_start(tmp_path):
    module = load()
    from segmented_capture.api import create_app
    # Actual segmented API schema exposes the incompatible interface.
    assert not module.supports_rollout_contract(create_app().openapi())
    assert module.supports_rollout_contract(module.create_rollout_app(data_root=tmp_path, min_free_bytes=1).openapi())

@pytest.mark.parametrize("outcome", ["success", "failure", "aborted"])
def test_real_rlt_client_roundtrip(tmp_path, outcome):
    # Use production client + actual Task5 HTTP routes, fake recorder hardware only.
    import sys
    import types
    from capture_core.api import create_app
    from tests.test_api import FakeRecorder, _write_episode
    method = Path(__file__).parents[1]
    trace_name = "methods.openpi_rlt.cobot_adapter.trace"
    saved = sys.modules.get(trace_name)
    trace = types.ModuleType(trace_name)
    # Task5 runtime is Python 3.8; postponed annotations make this pure enum module loadable.
    exec((method / "cobot_adapter/trace.py").read_text(), trace.__dict__)
    sys.modules[trace_name] = trace
    try:
        spec = importlib.util.spec_from_file_location(
            "rlt_contract_client", method / "cobot_adapter/task5_client.py")
        client_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = client_module
        spec.loader.exec_module(client_module)
        class Recorder(FakeRecorder):
            def stop(self):
                result = super().stop()
                fixture = _write_episode(tmp_path / "fixture", self.request.identity.episode_uuid)
                import h5py
                index = self.request.identity.episode_index
                with h5py.File(fixture, "r+") as stream:
                    stream.attrs["episode_index"] = index
                fixture.replace(tmp_path / "in_the_pot/pi05/round_001" / ("episode_%06d.hdf5" % index))
                return result
        class Bridge:
            def __init__(self, cache): pass
            def start(self): pass
            def shutdown(self): pass
            def status(self): return {"state": "ready", "error_code": None}
        with TestClient(create_app(recorder=Recorder(), ros_bridge_factory=Bridge, readiness_provider=lambda: {"status": "ok"})) as http:
            class Client(client_module.Task5Client):
                def _request(self, method, path, body=None):
                    response = http.request(method, path, json=body)
                    assert response.status_code == 200, response.text
                    return response.json()
            client = Client("http://testserver")
            identity = client_module.Task5EpisodeIdentity(
                "in_the_pot", "pi05", "step_2000", "round_001", str(tmp_path), 3600)
            ref = client.start_episode(identity)
            assert client.status()["state"] == "recording"
            done = client.finish_episode(ref, trace.EpisodeOutcome(outcome))
            assert done.episode_uuid
            if outcome == "aborted":
                assert not list(tmp_path.rglob("*.hdf5"))
                assert not list(tmp_path.rglob("*.labels.json"))
            else:
                labels = http.get("/api/episodes/" + done.episode_uuid + "/labels",
                                  params={"data_root": str(tmp_path)}).json()
                assert labels["episode_outcome"] == outcome
    finally:
        sys.modules.pop("rlt_contract_client", None)
        if saved is None:
            sys.modules.pop(trace_name, None)
        else:
            sys.modules[trace_name] = saved

def test_identity_route_is_not_shadowed_by_static_mount(tmp_path):
    app = load().create_rollout_app(data_root=tmp_path, min_free_bytes=1)
    # No context manager: do not start the ROS lifespan for this routing test.
    client = TestClient(app)
    response = client.get("/rlt/identity")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "service": "rlt-continuous-task5-v1", "data_root": str(tmp_path)}
    assert client.get("/healthz").status_code == 200
    assert load().supports_rollout_contract(client.get("/openapi.json").json())
    client.close()


def test_cli_bounds_shutdown_with_open_preview_connections(tmp_path, monkeypatch):
    import runpy, sys
    calls = []
    monkeypatch.syspath_prepend(str(MODULE.parent))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kw: calls.append(kw)))
    monkeypatch.setattr(sys, "argv", [str(MODULE), "--data-root", str(tmp_path)])
    runpy.run_path(str(MODULE), run_name="__main__")
    assert calls[0]["timeout_graceful_shutdown"] == 5
