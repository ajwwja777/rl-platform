"""Start or validate only the RLT-owned read-only ROS recording service."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
from rollout_recorder_app import supports_rollout_contract

BASE = Path("/media/agilex/Getea1/jiaan/projects/cobot-realworld-vla")
DATA = Path("/media/agilex/Getea1/jiaan/data/cobot-realworld-vla/task5/plug-insertion-rlt-v1/raw")
STATE = BASE / "runs/openpi-rlt/plug-insertion-stage1-v2-session-v1/task5-recorder"
CODE = BASE / "task5/segmented-teach-v1/code"
DEFAULT = "http://127.0.0.1:8017"

def get(url):
    with build_opener(ProxyHandler({})).open(url, timeout=2) as response:
        return json.load(response)

def ready(url):
    identity = get(url + "/rlt/identity")
    if identity != {"service": "rlt-continuous-task5-v1", "data_root": str(DATA)}:
        raise RuntimeError("Recorder identity/data root mismatch")
    if not supports_rollout_contract(get(url + "/openapi.json")):
        raise RuntimeError("Task5 API is incompatible with RLT continuous recording")
    health = get(url + "/healthz")
    if health.get("status") != "ok":
        raise RuntimeError("Recorder ROS is not ready: " + str(health))

def ensure(url):
    url = url.rstrip("/")
    try:
        ready(url)
        print("RLT continuous Task5 ready: " + url)
        return
    except Exception as error:
        reason = str(error)
    if url != DEFAULT:
        raise RuntimeError("Custom recorder is not ready/compatible: " + reason)
    with socket.socket() as probe:
        occupied = probe.connect_ex(("127.0.0.1", 8017)) == 0
    if occupied:
        raise RuntimeError("Port 8017 is occupied or recorder is not ready; no restart: " + reason)
    # Refuse to spawn a stale not-ready recorder before ROS exists.
    from capture_core.ros_subscriber import _ros_master_reachable
    if not _ros_master_reachable():
        raise RuntimeError("ROS master unavailable; prepare ROS before starting RLT")
    STATE.mkdir(parents=True, exist_ok=True)
    pidfile = STATE / "service.pid"
    if pidfile.exists():
        pid = int(pidfile.read_text().strip())
        if Path("/proc/" + str(pid)).exists():
            raise RuntimeError("Registered recorder PID still exists; inspect before restarting")
    env = dict(os.environ)
    env["ROS_HOME"] = str(STATE / "ros")
    env["ROS_LOG_DIR"] = str(STATE / "ros")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (STATE / "ros").mkdir(exist_ok=True)
    app = Path(__file__).with_name("rollout_recorder_app.py").resolve()
    logfile = STATE / ("service-" + time.strftime("%Y%m%d-%H%M%S") + ".log")
    with logfile.open("xb") as log:
        child = subprocess.Popen(
            [sys.executable, str(app), "--data-root", str(DATA), "--port", "8017"],
            cwd=str(STATE), env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(child.pid) + "\n")
    print("RLT recorder log: " + str(logfile), flush=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Recorder exited; inspect " + str(logfile))
            try:
                ready(url)
                print("RLT continuous Task5 ready: " + url)
                return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("Recorder readiness timeout; inspect " + str(logfile))
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        raise

def stop():
    pidfile = STATE / "service.pid"
    if not pidfile.exists():
        print("No RLT recorder registered")
        return
    pid = int(pidfile.read_text().strip())
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        print("RLT recorder already stopped")
        return
    command = (proc / "cmdline").read_bytes().split(b"\0")
    app = str(Path(__file__).with_name("rollout_recorder_app.py").resolve()).encode()
    if app not in command or b"8017" not in command:
        raise RuntimeError("Registered PID identity mismatch; refusing to stop")
    ready(DEFAULT)
    if get(DEFAULT + "/api/status").get("state") not in ("idle", "stopped"):
        raise RuntimeError("Recorder is active; end RLT Session before stopping it")
    import signal
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not proc.exists():
            print("RLT recorder stopped")
            return
        # A reparented process may remain a zombie briefly.
        if (proc / "stat").read_text().split(") ", 1)[1].startswith("Z"):
            print("RLT recorder exited")
            return
        time.sleep(0.1)
    raise RuntimeError("Recorder did not stop; no stronger signal sent")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    try:
        if args.stop:
            stop()
        else:
            ensure(args.url)
    except Exception as error:
        print("RLT recorder preflight failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
