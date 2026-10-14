import subprocess
import sys
from pathlib import Path


def test_fresh_worker_signal_has_fault_stack(tmp_path):
    # Isolated process, no model/ROS/GPU; disable core dumps before SIGSEGV.
    root = str(Path(__file__).resolve().parents[1])
    program = f"""
import sys, os, signal, resource
sys.path.insert(0, {root!r})
from methods.openpi_rlt.cobot_adapter import process_bootstrap as bootstrap
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
bootstrap.initialize_process = lambda config: None
def crash():
    os.kill(os.getpid(), signal.SIGSEGV)
bootstrap.run_initialized_role(crash, None)
"""
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == -11
    assert "Fatal Python error: Segmentation fault" in result.stderr
    assert "run_initialized_role" in result.stderr


def test_only_learner_changes_backend():
    from methods.openpi_rlt.cobot_adapter.process_bootstrap import configure_worker_backend
    for role in ["actor_service", "env_driver", "replay_manager", "MainProcess"]:
        env = {"COBOT_RLT_LEARNER_PLATFORM": "cpu", "CUDA_VISIBLE_DEVICES": "0"}
        before = dict(env)
        assert configure_worker_backend(role, env) is None
        assert env == before
    env = {"COBOT_RLT_LEARNER_PLATFORM": "cpu", "CUDA_VISIBLE_DEVICES": "0"}
    assert configure_worker_backend("learner_service", env) == "cpu"
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["JAX_PLATFORMS"] == "cpu"


def test_legacy_backend_remains_unchanged():
    from methods.openpi_rlt.cobot_adapter.process_bootstrap import configure_worker_backend
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    assert configure_worker_backend("learner_service", env) is None
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}


def test_invalid_backend_fails_closed():
    import pytest
    from methods.openpi_rlt.cobot_adapter.process_bootstrap import configure_worker_backend
    with pytest.raises(ValueError):
        configure_worker_backend("learner_service", {"COBOT_RLT_LEARNER_PLATFORM": "typo"})
