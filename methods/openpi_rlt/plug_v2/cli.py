# Project-owned lifecycle controller. Only its registered processes can be stopped.
import argparse,json,os,signal,subprocess,sys,time,uuid,fcntl,shlex,socket
from pathlib import Path
from urllib.request import Request,ProxyHandler,build_opener
os.environ['NO_PROXY']=','.join(filter(None,[os.environ.get('NO_PROXY',''),'127.0.0.1','localhost','::1']))
os.environ['no_proxy']=os.environ['NO_PROXY']
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/plug_v2';BACKEND=RUN/'backend'
MANIFEST=ROOT/'deployments/plug_v2/manifest.json'
MACHINE_PY='/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python'
CONTROL_PY=str(ROOT/'envs/rlt-online-py310/bin/python')
OLD=ROOT/'deployments/openpi-rlt/plug-insertion-stage1-v2/runtime-overlay'
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def identity(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text();parts=text[text.rfind(')')+2:].split()
        return None if parts[0]=='Z' else int(parts[19])
    except (FileNotFoundError,ProcessLookupError):return None
def alive(record):return identity(record['pid'])==record.get('start_ticks') and record.get('start_ticks') is not None
def occupied(port):
    with socket.socket() as s:return s.connect_ex(('127.0.0.1',port))==0
def http(path,body=None):
    req=Request('http://127.0.0.1:8026'+path,data=None if body is None else json.dumps(body).encode(),
                method='GET' if body is None else 'POST',headers={'Content-Type':'application/json','Connection':'close'})
    with build_opener(ProxyHandler({})).open(req,timeout=20) as response:return json.load(response)
def stop_session(timeout=150):
    if not occupied(8026):return
    status=http('/api/session')
    if status['phase']=='stopped':return
    # Stop itself pauses policy before asynchronous finalize, without homing.
    http('/api/session/stop',{'episode_id':status['episode_id'],'generation':status['generation']})
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        status=http('/api/session')
        if status['phase']=='stopped':return
        if status['phase']=='fault':
            # A failed file stays incomplete; stop the Session, never publish it.
            http('/api/session/stop',{'episode_id':status['episode_id'],'generation':status['generation']})
        time.sleep(.2)
    raise RuntimeError('Session finalization timeout; policy paused, model/data retained')

def recover_recorder():
    req=Request('http://127.0.0.1:8015/api/rlt/recover-recorder',data=b'{}',method='POST',
                headers={'Content-Type':'application/json','Connection':'close'})
    with build_opener(ProxyHandler({})).open(req,timeout=10) as response:return json.load(response)

def end_session():
    stop_session()
    if occupied(8026) and occupied(8015):recover_recorder()

def wait_conversions(timeout=180):
    deadline=time.monotonic()+timeout
    while True:
        active=[]
        for path in (BACKEND/'conversions').glob('*.json'):
            record=json.loads(path.read_text())
            if alive(record):active.append(record['pid'])
        if not active:return
        if time.monotonic()>deadline:
            raise RuntimeError('conversion still running; retain processes/data and wait before poweroff: '+str(active))
        time.sleep(.5)
def kill_registered(record):
    if not alive(record):return
    # Exact start ticks prevent PID reuse from affecting a different task.
    cmd=(Path('/proc')/str(record['pid'])/'cmdline').read_bytes()
    if b'methods.openpi_rlt.plug_v2' not in cmd and b'plug_v2/serve.py' not in cmd:
        raise RuntimeError('process command no longer belongs to plug_v2')
    if os.getpgid(record['pid'])!=record['pid']:raise RuntimeError('owned process group changed')
    os.killpg(record['pid'],signal.SIGTERM)
class Supervisor:
    def __init__(self,args):
        self.args=args;self.stop=False;self.children={};self.records={}
        self.generation=str(uuid.uuid4());self.started=identity(os.getpid())
        self.manifest=json.loads(MANIFEST.read_text())
        if self.manifest.get('cohort')!='plug_v2':raise ValueError('wrong deployment cohort')
        checkpoint=Path(self.manifest['checkpoint']).resolve(strict=True)
        checkpoint.relative_to((ROOT/'checkpoints/plug_v2').resolve())
        self.checkpoint=checkpoint
    def state(self,phase,error=None):
        atomic(BACKEND/'state.json',{'generation':self.generation,'phase':phase,
             'supervisor_pid':os.getpid(),'supervisor_start_ticks':self.started,
             'step':int(self.checkpoint.name),'mode':'explore' if self.args.explore else 'eval',
             'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'error_code':error,
             'children':{k:p.pid for k,p in self.children.items() if p.poll() is None}})
        atomic(BACKEND/'processes.json',{'supervisor':{'pid':os.getpid(),'start_ticks':self.started},
                                       'children':self.records})
    def launch(self,name,cmd,env=None):
        with (BACKEND/(name+'.log')).open('a') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.children[name]=p
        self.records[name]={'pid':p.pid,'start_ticks':identity(p.pid),'command':cmd}
        return p
    def stopping_requested(self):
        path=BACKEND/'request.json'
        if path.exists():
            try:
                command=json.loads(path.read_text()).get('command')
                if command=='down':
                    self.stop=True;path.unlink()
                elif command=='session-stop':
                    self.args.preload=True;path.unlink()
            except (OSError,ValueError):pass
        return self.stop
    def model(self):
        self.state('loading_machine_a')
        if occupied(8020):raise RuntimeError('port8020 occupied by an unregistered model')
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        if not free or int(free[0])<18000:raise RuntimeError('GPU0 has less than18GB free; do not disturb another task')
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.75',
             PYTHONPATH=':'.join([str(ROOT/'envs/machine-a-py311-overlay'),str(ROOT)]))
        cmd=[MACHINE_PY,'-u','-m','methods.openpi_rlt.plug_v2.serve','--project-root',str(ROOT),
             '--checkpoint',str(self.checkpoint),'--port','8020']
        p=self.launch('model',cmd,env);deadline=time.monotonic()+1800
        receipt=RUN/'model-server/validation.json'
        while not self.stopping_requested() and time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError('model exited; inspect '+str(BACKEND/'model.log'))
            if occupied(8020) and receipt.exists():
                data=json.loads(receipt.read_text())
                if data.get('pid')==p.pid and data.get('status')=='passed' and data['metadata']['checkpoint']==str(self.checkpoint):
                    self.state('loading_env');return
            time.sleep(1)
        if self.stop:return
        raise RuntimeError('model readiness timeout')
    def runtime(self):
        if occupied(8026):raise RuntimeError('port8026 occupied by an unregistered Session')
        if self.args.actor!='reference' and 'actor' not in self.children:
            if occupied(8021):raise RuntimeError('port8021 occupied')
            self.state('loading_actor')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
                  OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                  PYTHONPATH=':'.join([str(ROOT/'envs/machine-a-py311-overlay'),str(ROOT)]))
            p=self.launch('actor',[MACHINE_PY,'-u','-m','methods.openpi_rlt.plug_v2.actor_service'],env)
            deadline=time.monotonic()+120
            while not occupied(8021):
                if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError('actor service not ready')
                time.sleep(.2)
        self.state('loading_env')
        deadline=time.monotonic()+1800
        while not occupied(11311):
            if self.stopping_requested():return
            if self.args.preload:
                self.state('ready_disarmed');return
            if time.monotonic()>deadline:raise RuntimeError('ROS master not started; awaiting existing arm/camera launch')
            time.sleep(1)
        args=['--actor',self.args.actor]
        if self.args.shadow:args+=['--shadow']
        if self.args.explore:args+=['--explore']
        if self.args.no_record:args+=['--no-record']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
                                         PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        paths=[str(ROOT),str(OLD),str(ROOT/'code/openpi-rlt/rlt_online_rl/src')]
        command="source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash; "
        command+="export PYTHONPATH="+shlex.quote(':'.join(paths))+':"$PYTHONPATH"; '
        command+="exec "+shlex.join([CONTROL_PY,'-u','-m','methods.openpi_rlt.plug_v2.runtime',*args])
        p=self.launch('session',['bash','-c',command],env)
        deadline=time.monotonic()+120
        while not occupied(8026):
            if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Session not ready; inspect '+str(BACKEND/'session.log'))
            time.sleep(.2)
        if self.args.actor=='latest' and not self.args.no_record and not self.args.shadow and 'cycle' not in self.children:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
                  PYTHONPATH=':'.join([str(ROOT/'envs/machine-a-py311-overlay'),str(ROOT)]))
            self.launch('cycle',[MACHINE_PY,'-u','-m','methods.openpi_rlt.plug_v2.training_flow','cycle'],env)
        self.state('ready_disarmed')
    def run(self):
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        try:
            self.model()
            if self.stop:pass
            elif self.args.preload:
                self.state('ready_disarmed')
            else:self.runtime()
            while not self.stop:
                request=BACKEND/'request.json'
                if request.exists():
                    data=json.loads(request.read_text());request.unlink()
                    if data['command']=='down':self.stop=True;continue
                    if data['command']=='session-stop':
                        end_session();self.args.preload=True;self.state('ready_disarmed');continue
                    if data['command']=='up':
                        if 'session' in self.children and self.children['session'].poll() is None:
                            end_session();kill_registered(self.records['session']);self.children['session'].wait(timeout=20)
                        if 'cycle' in self.children:
                            kill_registered(self.records['cycle']);self.children['cycle'].wait(timeout=30)
                            del self.children['cycle'];del self.records['cycle']
                        self.args.preload=False
                        self.args.actor=data['actor'];self.args.explore=data['explore']
                        self.args.shadow=data['shadow'];self.args.no_record=data['no_record'];self.runtime()
                for name,p in self.children.items():
                    if p.poll() is not None:raise RuntimeError(name+'_exited_'+str(p.returncode))
                time.sleep(.2)
            stop_session()
            self.state('stopping')
            for name in ('session','cycle','actor','model'):
                if name in self.records:
                    kill_registered(self.records[name])
                    self.children[name].wait(timeout=30)
            self.state('offline')
        except Exception as error:
            self.state('fault',str(error))
            # Pause/finalize if possible; preserve unfinalized data on errors.
            try:stop_session(30)
            except Exception:pass
            for name in ('session','cycle','actor','model'):
                if name in self.records:
                    try:kill_registered(self.records[name]);self.children[name].wait(timeout=30)
                    except Exception:pass
            raise
def cleanup_rtc_actor():
    reg=BACKEND/'rtc-actor-process.json'
    if reg.exists():
        record=json.loads(reg.read_text())
        if alive(record):
            cmd=(Path('/proc')/str(record['pid'])/'cmdline').read_bytes()
            if b'methods.openpi_rlt.plug_v2.rtc_actor_service' not in cmd:raise RuntimeError('RTC actor identity mismatch')
            kill_registered(record)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['up','down','status','supervise','preload','session-stop'])
    p.add_argument('--actor',choices=['reference','warmup','latest','corrective'],default='reference')
    p.add_argument('--explore',action='store_true');p.add_argument('--shadow',action='store_true')
    p.add_argument('--no-record',action='store_true');p.add_argument('--preload',action='store_true')
    p.add_argument('--foreground',action='store_true');p.add_argument('--restart',action='store_true')
    args=p.parse_args();BACKEND.mkdir(parents=True,exist_ok=True)
    if args.command=='supervise':Supervisor(args).run();return
    if args.command=='status':
        print((BACKEND/'state.json').read_text() if (BACKEND/'state.json').exists() else '{"phase":"offline"}')
        return
    with (BACKEND/'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=BACKEND/'processes.json'
        state=json.loads(registry.read_text()) if registry.exists() else {}
        live=state.get('supervisor') and alive(state['supervisor'])
        previous_session=state.get('children',{}).get('session')
        if args.command=='session-stop':
            end_session()
            if live and not occupied(8026):atomic(BACKEND/'request.json',{'command':'session-stop'})
            print('Session已结束；预加载模型保留。再次rlt_up可开新Session。');return
        if args.command=='down':
            if not live:print('RLT已停止');return
            end_session()
            wait_conversions()
            atomic(BACKEND/'request.json',{'command':'down'})
            deadline=time.monotonic()+90
            while alive(state['supervisor']) and time.monotonic()<deadline:time.sleep(.2)
            if alive(state['supervisor']):raise RuntimeError('shutdown pending; inspect backend logs')
            print('RLT已停止，模型显存已释放');return
        manifest=json.loads(MANIFEST.read_text())
        if args.actor=='corrective':
            from methods.openpi_rlt.plug_v2.fixed_candidate import manifest as fixed_manifest
            fixed_manifest()
            if args.explore:raise ValueError('fixed corrective trial does not support exploration')
        elif args.actor!='reference':
            if (RUN/'learning/rtc-v5/current.json').exists():
                from .rtc_release import selected
                _,release=selected()
                print('RLT upstream RTC candidate='+release['name']+' step='+str(release['global_step'])+' fixed='+str(args.actor=='warmup'),flush=True)
            elif (RUN/'learning/v4/current.json').exists():
                from .online_release import selected
                _,release=selected()
                print('RLT v4 actor='+str(release['actor_version'])+' mode='+args.actor+'; fixed='+str(args.actor=='warmup'),flush=True)
            actor=RUN/'learning'/('warmup/actor.pt' if args.actor=='warmup' else 'online/actor.pt')
            if args.actor=='latest' and not actor.exists():actor=RUN/'learning/warmup/actor.pt'
            if not (RUN/'learning/rtc-v5/current.json').exists() and not actor.exists():raise RuntimeError('run manual warmup first; no legacy actor import')
        if args.command!='preload' and manifest.get('status')!='offline_validated':
            raise RuntimeError('plug_v2 model not validated; RLT unavailable')
        if live:
            if args.command=='preload':print('模型预加载进程已存在');return
            if args.restart:end_session()
            if occupied(8026) and http('/api/session')['phase'] not in ('disarmed','ready','stopped'):
                raise RuntimeError('先运行 ./scripts/rlt_stop.sh；或 rlt_up.sh --restart 结束旧Session再启动')
            atomic(BACKEND/'request.json',{'command':'up','actor':args.actor,'explore':args.explore,
                                         'shadow':args.shadow,'no_record':args.no_record})
        else:
            argv=[sys.executable,'-u','-m','methods.openpi_rlt.plug_v2.cli','supervise','--actor',args.actor]
            if args.command=='preload':argv+=['--preload']
            for name in ('explore','shadow','no_record'):
                if getattr(args,name):argv+=['--'+name.replace('_','-')]
            with (BACKEND/'supervisor.log').open('a') as f:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL,start_new_session=True)
            atomic(registry,{'supervisor':{'pid':child.pid,'start_ticks':identity(child.pid)},'children':{}})
        print('RLT正在准备；网页 http://127.0.0.1:8015/ 显示加载状态',flush=True)
    if args.foreground:
        from methods.openpi_rlt.plug_v2.terminal import watch
        watch(previous_session)
if __name__=='__main__':main()
