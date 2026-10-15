"""Bounded, read-only Learner snapshots over the existing restricted SSH key.

No learner imports, GPU probing, owner locks, tunnels or process lifecycle calls.
"""
import json
from pathlib import Path
import re
import runpy
import subprocess
import time

MAX_BYTES = 1024 * 1024


def read_object(path):
    try:
        with Path(path).open('rb') as stream:
            data = stream.read(MAX_BYTES + 1)
    except FileNotFoundError:
        return {}
    if len(data) > MAX_BYTES:
        raise ValueError('Monitoring payload exceeds limit')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Invalid monitoring object')
    return value


def process_matches(pid, root, model_id, proc_root=Path('/proc')):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        args = (proc_root / str(pid) / 'cmdline').read_bytes().decode().split('\0')
        return (args[args.index('--run-dir') + 1] == str(root)
                and args[args.index('--deployment-id') + 1] == model_id
                and any(Path(a).name in ('observed_learner.py', 'online_learner.py') for a in args))
    except (OSError, ValueError, IndexError):
        return False


def snapshot(config):
    if 'relay' in config:
        raise ValueError('Independent monitoring is unavailable for relay profiles')
    root = Path(config['run_dir'])
    status = read_object(root / 'status.json')
    if status and (status.get('deployment_id') != config['model_id']
                   or status.get('rtc_delay') != config['rtc_delay']
                   or status.get('edit_scale') != config['edit_scale']):
        raise ValueError('Learner monitoring identity mismatch')
    history = read_object(root / 'telemetry/learner.json')
    points = history.get('points', [])
    if not isinstance(points, list) or len(points) > 600:
        raise ValueError('Invalid bounded learner history')
    running = process_matches(status.get('pid'), root, config['model_id'])
    return dict(schema=1, model_id=config['model_id'],
                options=dict(rtc_delay=config['rtc_delay'], edit_scale=config['edit_scale']),
                training=history, running=running,
                phase=status.get('phase', 'unknown') if running else 'stopped',
                learner_version=status.get('actor_version'),
                critic_updates=status.get('critic_updates'),
                status_updated_at=status.get('updated_at'), sampled_at=time.time())


def profile_for_model(model, platform_root):
    """Only a profile already required by the trusted model catalogue is used."""
    allowed = (Path(platform_root) / 'runtime/expo-remote').resolve()
    candidates = [Path(p) for p in model.get('required', [])
                  if Path(p).name.startswith('profile') and Path(p).suffix == '.json'
                  and Path(p).resolve().parent == allowed]
    if len(candidates) != 1:
        return None
    profile = read_object(candidates[0])
    if profile.get('model_id') != model['id']:
        raise ValueError('Monitoring profile identity mismatch')
    return profile


def monitor_command(profile, selected):
    name = profile.get('launch_name')
    if name is not None and not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', name):
        raise ValueError('Invalid configured monitor profile')
    destination = profile['destination']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+', destination):
        raise ValueError('Invalid configured monitor destination')
    command = 'monitor' + (' ' + name if name else '')
    defaults = dict(rtc_delay=profile['rtc_delay'], edit_scale=profile['edit_scale'])
    normalize = runpy.run_path(str(Path(__file__).with_name('online_options.py')))['normalize']
    selected = normalize(selected or defaults)
    if selected != defaults and not (name and profile.get('allow_online_options')):
        raise ValueError('Monitor profile does not allow selected options')
    if name and profile.get('allow_online_options'):
        command += ' --rtc-delay %d --edit-scale %.2f' % (selected['rtc_delay'], selected['edit_scale'])
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5',
            '-o', 'ServerAliveInterval=3', '-o', 'ServerAliveCountMax=1',
            '-i', profile['identity_file'], destination, command]


def fetch(profile, selected):
    command = monitor_command(profile, selected)
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=10, check=False)
    if result.returncode:
        raise OSError('Learner monitoring SSH unavailable (exit %d)' % result.returncode)
    if len(result.stdout) > MAX_BYTES:
        raise ValueError('Monitoring response exceeds limit')
    value = json.loads(result.stdout)
    expected = selected or dict(rtc_delay=profile['rtc_delay'], edit_scale=profile['edit_scale'])
    if (not isinstance(value, dict) or value.get('schema') != 1
            or value.get('model_id') != profile['model_id'] or value.get('options') != expected
            or type(value.get('running')) is not bool
            or not isinstance(value.get('training'), dict)
            or not isinstance(value['training'].get('points', []), list)
            or len(value['training'].get('points', [])) > 600):
        raise ValueError('Remote monitoring response identity or schema mismatch')
    return value
