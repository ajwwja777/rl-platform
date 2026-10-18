"""Resume the qualified EXPO HPC deployment on an owned four-GPU allocation.

Keeps training state and updates the gateway job binding. Does not reserve GPUs,
enable the Web gate, or issue robot commands. Holds the learner until allocation
termination. The root defaults to the qualified 2026-10-18 deployment assets.
"""
from pathlib import Path
import fcntl,json,os,shlex,signal,subprocess,sys,time,traceback
from urllib.request import build_opener,ProxyHandler
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job_id',type=int)
parser.add_argument('--root',type=Path,default=Path('/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-hpc-online-20261018'))
parser.add_argument('--check',action='store_true',help='Validate allocation, host and saved state without starting the learner')
args=parser.parse_args();H=args.root.resolve();JOB=args.job_id
if JOB<=0:parser.error('Positive owned Slurm job ID required')
ADAPTER=H/'source/methods/expo_ft/cobot_adapter'
sys.path.insert(0,str(ADAPTER))
from online_protocol import atomic_json
from hpc_gateway import allocated_node
owner=(H/'orchestrator.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
config=json.loads((H/'learner.json').read_text());config['job_id']=JOB
logroot=H/'resumed'/str(JOB);logroot.mkdir(parents=True,exist_ok=True)
service=None;tunnel=None
opener=build_opener(ProxyHandler({}))
def event(phase,**data):
    row=dict(phase=phase,time=time.time(),pid=os.getpid(),job_id=JOB,**data)
    atomic_json(H/'orchestrator-status.json',row);print(json.dumps(row),flush=True)
def step(command,gpus=4):
    return ['srun','--jobid='+str(JOB),'--overlap','--exact','--nodes=1','--ntasks=1',
            '--cpus-per-task='+str(12*gpus),'--gres=gpu:'+str(gpus),'--cpu-bind=cores','--gpu-bind=none','--kill-on-bad-exit=1','--export=ALL']+command
def stop_service():
    global service
    if service is not None and service.poll() is None:
        service.terminate()
        try:service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            service.kill();service.wait(timeout=10)
    service=None
def start_service(profile,name,port):
    global service
    event(name)
    log=(logroot/(name+'.log')).open('w')
    command=[str(H/'python-expo'),str(ADAPTER/'hpc_learner.py'),'--config',str(profile)]
    service=subprocess.Popen(step(command),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(H/'service-process.json',dict(pid=service.pid,job_id=JOB,profile=str(profile),log=str(logroot/(name+'.log'))))
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if service.poll() is not None:raise RuntimeError(name+' exited '+str(service.returncode))
        try:
            with opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=5) as r:value=json.load(r)
            if value.get('phase')=='fault':raise RuntimeError(value.get('error'))
            if value.get('ready'):
                expected=json.loads(profile.read_text())
                assert value['deployment_id']==expected['model_id'] and value['learner_devices']==4
                return value
        except (OSError,ValueError):pass
        time.sleep(2)
    raise TimeoutError(name+' startup timed out')
def stop_requested(signum, frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop_requested)

try:
    fields_result=subprocess.run(['scontrol','show','job',str(JOB),'-o'],capture_output=True,text=True,check=True,timeout=15)
    node=allocated_node(config,fields_result.stdout)
    fields=dict(x.split('=',1) for x in fields_result.stdout.split() if '=' in x)
    assert 'gres/gpu=4' in fields.get('AllocTRES','').split(',')
    known=Path.home()/'.ssh/known_hosts'
    found=subprocess.run(['ssh-keygen','-F',node,'-f',str(known)],capture_output=True)
    if found.returncode:
        # The scan executes on the allocated node, against its own loopback sshd.
        result=subprocess.run(step(['ssh-keyscan','-T','5','-t','ed25519','127.0.0.1']),capture_output=True,text=True,check=True,timeout=60)
        keys=[x.split() for x in result.stdout.splitlines() if len(x.split())==3 and x.split()[1]=='ssh-ed25519']
        assert len(keys)==1
        known.parent.mkdir(exist_ok=True,mode=0o700)
        with known.open('a') as f:f.write(node+' '+keys[0][1]+' '+keys[0][2]+'\n')
    subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','jhe724@'+node,'true'],check=True,timeout=15)
    initial=json.loads((Path(config['run_dir'])/'committed.json').read_text())
    if args.check:
        print(json.dumps(dict(passed=True,job_id=JOB,node=node,actor_version=initial['updates'],model_id=config['model_id'],dry_run=True)),flush=True)
    else:
        gateway=json.loads((H/'gateway.json').read_text());gateway['job_id']=JOB
        assert gateway['run_dir']==config['run_dir'] and gateway['model_id']==config['model_id']
        atomic_json(logroot/'previous-profiles.json',dict(learner=json.loads((H/'learner.json').read_text()),gateway=json.loads((H/'gateway.json').read_text())))
        atomic_json(H/'learner.json',config);atomic_json(H/'gateway.json',gateway)
        tunnel=subprocess.Popen(['ssh','-N','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ExitOnForwardFailure=yes','-L','127.0.0.1:18446:127.0.0.1:18443','jhe724@'+node],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=(logroot/'tunnel.log').open('w'),start_new_session=True)
        time.sleep(1);assert tunnel.poll() is None
        value=start_service(H/'learner.json','resume_production',18446)
        assert value['actor_version']>=initial['updates'] and value['phase']=='ready' and value['training_enabled']
        atomic_json(H/'resume-ready.json',dict(passed=True,job_id=JOB,node=node,end_time=fields.get('EndTime'),status=value,robot_execution=False,web_gate_enabled=False))
        event('resumed_learner_running',node=node,actor_version=value['actor_version'])
        tunnel.terminate();tunnel.wait(timeout=15);tunnel=None
        service.wait();event('learner_exited',exit_code=service.returncode)
except Exception as exc:
    event('failed',error=str(exc));traceback.print_exc();stop_service();raise
finally:
    stop_service()
    if tunnel is not None and tunnel.poll() is None:tunnel.terminate()
