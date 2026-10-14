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
