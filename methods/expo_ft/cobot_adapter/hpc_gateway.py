"""Attach to a prestarted EXPO Learner inside one owned Slurm allocation.

No GPU jobs, models or robot operations are started by this gateway. SSH EOF
closes only the tunnel; the scheduler owns the Learner lifetime.
"""
import argparse,json,os,re,shlex,subprocess,sys
from pathlib import Path

def allocated_node(config, output):
    fields=dict(item.split('=',1) for item in output.split() if '=' in item)
    job=str(config['job_id'])
    if not job.isdigit() or fields.get('JobId')!=job:
        raise ValueError('Slurm job identity mismatch')
    if fields.get('UserId')!=config['slurm_user']+'('+str(os.getuid())+')':
        raise ValueError('Slurm job owner mismatch')
    if fields.get('JobState')!='RUNNING':
        raise ValueError('EXPO HPC allocation is not running: '+fields.get('JobState','unknown'))
    node=fields.get('NodeList','')
    if fields.get('NumNodes')!='1' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*',node):
        raise ValueError('EXPO requires exactly one allocated compute node')
    return node

def validate(config):
    if type(config['job_id']) is not int or config['job_id']<=0:raise ValueError('Invalid Slurm job')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',config['slurm_user']):raise ValueError('Invalid Slurm user')
    for key in ('port','learner_port'):
        if type(config[key]) is not int or not 1024<=config[key]<=65535:raise ValueError('Invalid fixed port')
    if not Path(config['run_dir']).is_absolute():raise ValueError('Run path must be absolute')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--monitor',action='store_true')
    p.add_argument('--monitor-local',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text());validate(config)
    if args.monitor_local:
        from remote_monitor import snapshot
        print(json.dumps(snapshot(config)));return
    result=subprocess.run(['scontrol','show','job',str(config['job_id']),'-o'],capture_output=True,text=True,check=True,timeout=8)
    try:node=allocated_node(config,result.stdout)
    except ValueError:
        if not args.monitor:raise
        from remote_monitor import snapshot
        value=snapshot(config);value.update(running=False,phase='allocation_unavailable');print(json.dumps(value));return
    command=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
             '-o','ConnectTimeout=8','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2']
    if args.monitor:
        remote=shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),'--monitor-local'])
        command += [config['slurm_user']+'@'+node,remote]
    else:
        # Existing service must match the immutable configured run and role.
        status=json.loads((Path(config['run_dir'])/'status.json').read_text())
        if status.get('deployment_id')!=config['model_id'] or status.get('learner_devices')!=4 or not status.get('training_enabled'):
            raise ValueError('HPC Learner has not passed startup identity checks')
        command += ['-o','ExitOnForwardFailure=yes','-L',
                    '127.0.0.1:%d:127.0.0.1:%d'%(config['port'],config['learner_port']),
                    config['slurm_user']+'@'+node,
                    '/usr/bin/python3 -c '+shlex.quote('import sys; sys.stdin.buffer.read()')]
    os.execvp(command[0],command)

if __name__=='__main__':main()
