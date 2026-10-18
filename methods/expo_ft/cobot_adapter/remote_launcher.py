"""Restricted SSH command: own one configured learner and drain on disconnect."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shlex
import re
import subprocess
import sys
import threading
import time



def relay_command(config):
    relay = config['relay']
    destination = relay['destination']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+', destination):
        raise ValueError('Invalid fixed Learner relay destination')
    for key in ('launcher', 'profile'):
        if not Path(relay[key]).is_absolute():
            raise ValueError('Learner relay paths must be absolute')
    local_port, remote_port = int(config['port']), int(relay['port'])
    if not all(1024 <= port <= 65535 for port in (local_port, remote_port)):
        raise ValueError('Invalid Learner relay port')
    interpreter = relay.get('python', '/usr/bin/python3')
    if not Path(interpreter).is_absolute():raise ValueError('Relay interpreter must be absolute')
    remote = shlex.join([interpreter, relay['launcher'], '--config', relay['profile']])
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=3', '-o', 'ServerAliveCountMax=2',
            '-o', 'ConnectTimeout=8', '-L',
            '127.0.0.1:%d:127.0.0.1:%d' % (local_port, remote_port), destination, remote]

def selected_profile(config, command):
    if command == 'run': return config
    match = re.fullmatch(r'run ([a-z][a-z0-9_]{0,31})(?: --rtc-delay ([1-8]) --edit-scale (0\.(?:0[1-9]|10)))?', command)
    if not match or match[1] not in config.get('profiles', {}):
        raise ValueError('Only a configured EXPO profile and bounded options are allowed')
    path = Path(config['profiles'][match[1]])
    if not path.is_absolute(): raise ValueError('Configured profile path must be absolute')
    selected = json.loads(path.read_text())
    if selected.get('port') != config.get('port'): raise ValueError('Profile must use the restricted forward port')
    if match[2] is not None:
        if not selected.get('allow_online_options') or 'relay' in selected:
            raise ValueError('This Learner profile does not allow configurable Online parameters')
        from online_options import configured_run
        selected = configured_run(selected, dict(rtc_delay=int(match[2]), edit_scale=float(match[3])))
    return selected


def select_gpus(config, is_busy):
    candidates=config.get('gpu_candidates',[config['gpu']])
    maximum=config.get('max_devices',1)
    if (not isinstance(candidates,list) or not candidates or maximum not in (1,2,4)
            or any(type(x) is not int or x<0 for x in candidates)
            or len(set(candidates))!=len(candidates)):
        raise ValueError('Invalid EXPO GPU selection')
    free=[gpu for gpu in candidates if not is_busy(gpu)]
    if not free:raise RuntimeError('Reserved EXPO GPUs are already in use')
    count = max(n for n in (1,2,4) if n <= maximum and n <= len(free))
    return free[:count]
def learner_command(config, adapter):
    command = [config['python'], str(adapter / 'observed_learner.py'), '--project', config['project'],
               '--bundle', config['bundle'], '--run-dir', config['run_dir'], '--port', str(config['port']),
               '--recipe', 'native_sync', '--seed-dir', config['seed_dir'], '--deployment-id', config['model_id'],
               '--rtc-delay', str(config['rtc_delay']), '--native-batch-update', '--learner-only']
    episodes=config.get('episodes_per_update',1)
    if type(episodes) is not int or episodes<1:raise ValueError('Invalid episode batch size')
    command += ['--episodes-per-update',str(episodes)]
    if 'edit_scale' in config:
        from sampling_config import edit_scale as validate
        command += ['--edit-scale', str(validate(config['edit_scale']))]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    original = os.environ.get('SSH_ORIGINAL_COMMAND', 'run')
    if original == 'monitor' or original.startswith('monitor '):
        config = selected_profile(config, 'run' + original[len('monitor'):])
        from remote_monitor import snapshot
        print(json.dumps(snapshot(config), allow_nan=False), flush=True)
        return
    config=selected_profile(config,original)
    if "relay" in config:
        command = relay_command(config)
        os.execvp(command[0], command)
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    owner = (root / "owner.lock").open("a")
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def is_busy(gpu):
        check=subprocess.run(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=10)
        return bool(check.stdout.strip())
    selected=select_gpus(config,is_busy)
    gpu=','.join(map(str,selected))
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
    adapter = Path(config.get('adapter_dir', str(Path(__file__).resolve().parent)))
    if not adapter.is_absolute() or not (adapter / 'observed_learner.py').is_file():
        raise ValueError('Invalid configured versioned Learner adapter')
    command = learner_command(config, adapter)
    command+=['--num-devices',str(len(selected))]
    print('EXPO Learner CUDA_VISIBLE_DEVICES='+gpu,flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while process.poll() is None and not stop.wait(.5):
            pass
        # An operator unload / disconnected SSH stops rollout immediately locally.
        # Finish an already-started update/save before retiring the learner.
        deadline = time.monotonic() + 2400
        last_progress = None
        while process.poll() is None and time.monotonic() < deadline:
            try:
                status = json.loads((root / "status.json").read_text())
            except (OSError, ValueError):
                status = {}
            if status.get("pid") != process.pid or status.get("phase") != "updating":
                break
            progress = status.get("updated_at")
            if progress != last_progress:
                deadline = time.monotonic() + 2400
                last_progress = progress
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
