"""Restricted SSH command: own one configured learner and drain on disconnect."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if os.environ.get("SSH_ORIGINAL_COMMAND", "run") != "run":
        raise ValueError("Only the configured EXPO run is allowed")
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / "owner.lock").open("a")
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = str(config["gpu"])
    check = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    if check.stdout.strip():
        raise RuntimeError("Reserved EXPO GPU is already in use")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    def disconnected():
        while sys.stdin.buffer.read(1):
            pass
        stop.set()
    threading.Thread(target=disconnected, daemon=True).start()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform", OMP_NUM_THREADS="4", JAX_COMPILATION_CACHE_DIR=config["cache_dir"])
    adapter = Path(__file__).resolve().parent
    command = [config["python"], str(adapter / "online_learner.py"), "--project", config["project"], "--bundle", config["bundle"], "--run-dir", str(root), "--port", str(config["port"]), "--recipe", "native_sync", "--seed-dir", config["seed_dir"], "--deployment-id", config["model_id"], "--rtc-delay", str(config["rtc_delay"]), "--native-batch-update"]
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while process.poll() is None and not stop.wait(.5):
            pass
        # An operator unload / disconnected SSH stops rollout immediately locally.
        # Finish an already-started update/save before retiring the learner.
        deadline = time.monotonic() + 2400
        while process.poll() is None and time.monotonic() < deadline:
            try:
                status = json.loads((root / "status.json").read_text())
            except (OSError, ValueError):
                status = {}
            if status.get("pid") != process.pid or status.get("phase") != "updating":
                break
            time.sleep(1)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
