"""Explicit loading and shutdown of the existing online continuation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from methods.openpi_rlt.cobot_adapter.lifecycle import (
    LifecycleError, LifecycleState, process_start_ticks, read_lifecycle,
    validate_process, write_lifecycle,
)
from methods.openpi_rlt.cobot_adapter.session_shutdown import shutdown_via_http

CONSOLE_URL = "http://127.0.0.1:8015"
RUNTIME = Path("/home/agilex/cobot_magic/task3/jiaan/runtime/cobot-rlt-backend-v1")
PORTS = (8000, 8016, 9101, 9102)


class BackendLifecycleError(RuntimeError):
    pass


def http_json(base, path):
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(base + path, headers={"Accept": "application/json", "Connection": "close"}), timeout=3) as response:
        if "application/json" not in response.headers.get("Content-Type", ""):
            raise BackendLifecycleError("not_json: " + path)
        return json.load(response)


def console_preflight(get):
    identity = get("/api/console/identity")
    if identity.get("service") != "cobot-data-console-v1":
        raise BackendLifecycleError("wrong_console_identity")
    status = get("/api/console/status")
    if status.get("selected_mode") != "rlt":
        raise BackendLifecycleError("select_rlt_mode_on_8015")
    if status.get("active_mode") is not None:
        raise BackendLifecycleError("writer_busy")
    ready = status.get("ros_readiness", {})
    if ready.get("status") != "ok":
        raise BackendLifecycleError("console_not_ready: " + str(ready.get("error_code", "unknown")))


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def process_alive(pid, ticks=None):
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        if text[text.rfind(")") + 2:].split()[0] == "Z":
            return False
        if ticks is not None and process_start_ticks(pid) != ticks:
            return False
        return True
    except (OSError, LifecycleError, IndexError):
        return False


def start_backend(state_path, interface, step, mode, options, *, wait_seconds=0):
    if state_path.exists():
        old = read_lifecycle(state_path)
        if process_alive(old.supervisor_pid, old.supervisor_start_ticks):
            validate_process(old.supervisor_pid, old.supervisor_start_ticks, "--managed-backend")
            validate_process(old.supervisor_pid, old.supervisor_start_ticks, str(state_path))
            cmdline = (Path("/proc") / str(old.supervisor_pid) / "cmdline").read_bytes().decode("utf-8").split(chr(0))
            for flag in ("--explore", "--frozen-actor", "--shadow", "--no-record"):
                if (flag in options) != (flag in cmdline):
                    raise BackendLifecycleError("backend_options_mismatch; run rlt_down first")
            if old.step != step or old.mode != mode:
                raise BackendLifecycleError("backend_configuration_mismatch; run rlt_down first")
            return old
        if old.phase not in {"offline", "fault"}:
            raise BackendLifecycleError("stale_backend_state; run rlt_down before restarting")
    console_preflight(lambda path: http_json(CONSOLE_URL, path))
    # Machine A may be preloaded; its registered model-server script verifies ownership.
    for port in (8016, 9101, 9102):
        if port_open(port):
            raise BackendLifecycleError("unknown_port_owner: " + str(port))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = state_path.parent / ("supervisor-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log")
    command = ["bash", str(interface), str(step), *options,
        "--managed-backend", "--lifecycle-state", str(state_path)]
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state = LifecycleState(str(uuid4()), "loading_machine_a", process.pid,
        process_start_ticks(process.pid), step, mode, datetime.now(timezone.utc).isoformat(),
        None, {}, {})
    write_lifecycle(state_path, state)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        state = read_lifecycle(state_path)
        if state.phase in {"ready_disarmed", "ready", "fault", "offline"}:
            break
        if process.poll() is not None:
            state = replace(state, phase="fault", error_code="supervisor_exited_before_ready")
            write_lifecycle(state_path, state)
            break
        time.sleep(0.5)
    print("RLT backend: " + state.phase)
    print("Log: " + str(log_path))
    print("Console: " + CONSOLE_URL + "/")
    if state.phase == "fault":
        raise BackendLifecycleError(state.error_code or "backend_fault; inspect log")
    return state


def stop_backend(state_path, model_script, step, *, timeout_seconds=180):
    state = read_lifecycle(state_path) if state_path.exists() else None
    live = state is not None and process_alive(state.supervisor_pid, state.supervisor_start_ticks)
    if state is not None and state.step != step:
        raise BackendLifecycleError("step_mismatch")
    if live:
        validate_process(state.supervisor_pid, state.supervisor_start_ticks, "--managed-backend")
        validate_process(state.supervisor_pid, state.supervisor_start_ticks, str(state_path))
    if port_open(8016):
        if not live:
            raise BackendLifecycleError("unregistered_session_port; no process was stopped")
        result = shutdown_via_http("http://127.0.0.1:8016", timeout_sec=60)
        if result.get("phase") not in {"disarmed", "stopped", "waiting_scene"}:
            raise BackendLifecycleError("session_not_finalized: " + str(result.get("phase")))
    if live:
        if state.phase == "loading_machine_a":
            # Break the registered model-server restore wait before signaling its supervisor.
            subprocess.run(["bash", str(model_script), "stop", str(step)], check=True)
        if process_alive(state.supervisor_pid, state.supervisor_start_ticks):
            os.kill(state.supervisor_pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_seconds
        while process_alive(state.supervisor_pid, state.supervisor_start_ticks) and time.monotonic() < deadline:
            time.sleep(0.2)
        if process_alive(state.supervisor_pid, state.supervisor_start_ticks):
            raise BackendLifecycleError("supervisor_stop_timeout; no broad kill performed")
    if state is not None:
        for name, pid in state.children.items():
            if process_alive(pid, state.child_start_ticks[name]):
                raise BackendLifecycleError("child_still_running: " + name)
    subprocess.run(["bash", str(model_script), "stop", str(step)], check=True)
    occupied = [port for port in PORTS if port_open(port)]
    if occupied:
        raise BackendLifecycleError("ports_still_open: " + str(occupied))
    # Verify recording ownership has been released before the data UI is stopped.
    if port_open(8015):
        status = http_json(CONSOLE_URL, "/api/console/status")
        if status.get("active_mode") == "rlt":
            raise BackendLifecycleError("recorder_still_owned; keep data console running and inspect")
    if state is not None:
        write_lifecycle(state_path, replace(state, phase="offline", children={}, child_start_ticks={},
            updated_at=datetime.now(timezone.utc).isoformat(), error_code=None))
    print("RLT stopped; registered model released and RLT ports closed. Data console remains on 8015.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("up", "down", "status"))
    parser.add_argument("step", type=int, nargs="?", default=4000)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--frozen-actor", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.no_record and not args.frozen_actor:
        parser.error("--no-record requires --frozen-actor")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = RUNTIME / "state.json"
    scripts = Path(__file__).resolve().parent
    with (RUNTIME / "lifecycle.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BackendLifecycleError("another_lifecycle_operation_is_running")
        if args.action == "up":
            options = []
            for flag in ("explore", "frozen_actor", "shadow", "no_record"):
                if getattr(args, flag):
                    options.append("--" + flag.replace("_", "-"))
            mode = "explore" if args.explore else "eval"
            start_backend(state_path, scripts / "interface_task2_teach_rlt_continue5416.sh",
                args.step, mode, options, wait_seconds=args.wait_seconds)
        elif args.action == "down":
            stop_backend(state_path, scripts / "rlt_model_server.sh", args.step)
        else:
            print(json.dumps(read_lifecycle(state_path).__dict__, indent=2) if state_path.exists() else '{"phase":"offline"}')

if __name__ == "__main__":
    try:
        main()
    except (BackendLifecycleError, LifecycleError, OSError, subprocess.CalledProcessError) as error:
        print("RLT lifecycle failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)