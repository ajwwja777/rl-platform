# New-cohort RTC rollout runtime. Only explicit onsite arm/start releases policy.
import argparse,json,os,signal,sys,time,threading,subprocess,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
os.environ['NO_PROXY']=','.join(filter(None,[os.environ.get('NO_PROXY',''),'127.0.0.1','localhost','::1']))
os.environ['no_proxy']=os.environ['NO_PROXY']
ROOT=Path(__file__).resolve().parents[3]
PLATFORM=ROOT.parent/'cobot-platform'
RUN=ROOT/'runs/plug_v2'
OLD=ROOT/'deployments/openpi-rlt/plug-insertion-stage1-v2/runtime-overlay'
sys.path[:0]=[str(OLD),str(ROOT),str(ROOT/'code/openpi-rlt/rlt_online_rl/src')]
# Control process has no Torch/PyArrow/GPU learner; legacy type dependencies use CPU JAX.
os.environ.setdefault('JAX_PLATFORMS','cpu')
from methods.openpi_rlt.plug_v2.rtc_queue import RTCQueue,CommandFilter,DeadlineRecoveryBudget,PassiveCommandLatch,QueueFault,PiperWorkspaceGuard,PIPER_WORKSPACE_MIN,PIPER_WORKSPACE_MAX,PIPER_WORKSPACE_RADIUS
from methods.openpi_rlt.plug_v2.ros_io import RTCIO
from methods.openpi_rlt.plug_v2.recorder_client import FlatRecorderClient
from methods.openpi_rlt.cobot_adapter.session import RltSessionController,SessionPhase
from methods.openpi_rlt.cobot_adapter.session_http import RltSessionApplication,RltSessionHttpServer,SessionHooks
from methods.openpi_rlt.cobot_adapter.task5_client import Task5EpisodeIdentity
from methods.openpi_rlt.plug_v2.rpc import ModelClient as WebsocketClientPolicy
from methods.openpi_rlt.plug_v2.storage import selected_root,history
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
class CohortSessionController(RltSessionController):
    def update_takeover(self,*,left,right):
        with self._lock:
            if self._phase is SessionPhase.PAUSED and (left or right):
                self._phase=SessionPhase.HIL
                self._expert_mask=(bool(left),bool(right))
                self._fresh_plan_required=False
                self._generation+=1
                return self.snapshot()
            return super().update_takeover(left=left,right=right)
class GuardedApplication(RltSessionApplication):
    def learning_busy(self):
        path=RUN/'learning/operation.json'
        try:
            value=json.loads(path.read_text())
            if value.get('phase') not in ('preparing','training'):return False
            pid=int(value.get('pid',-1))
            if pid<=0:return False
            proc=Path('/proc')/str(pid)
            cmd=(proc/'cmdline').read_bytes()
            stat=(proc/'stat').read_text()
            return b'methods.openpi_rlt.plug_v2.training_flow' in cmd and stat[stat.rfind(')')+2:].split()[0]!='Z'
        except (OSError,ValueError):return False
    def check_learning(self):
        if self.learning_busy():
            from methods.openpi_rlt.cobot_adapter.session import SessionConflict
            raise SessionConflict('learning_in_progress','RL preparing/updating; wait before next rollout')
    def motion(self,name,kwargs):
        folder=RUN/'learning';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'operation.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                from methods.openpi_rlt.cobot_adapter.session import SessionConflict
                raise SessionConflict('learning_in_progress','RL preparing/updating; wait')
            self.check_learning()
            return getattr(super(),name)(**kwargs)
    def _start_recorded_episode(self,starting):
        try:return super()._start_recorded_episode(starting)
        except Exception as error:
            self._controller.fail('task5_start_failed: '+str(error))
            raise
    def start(self,**kwargs):return self.motion('start',kwargs)
    def resume(self,**kwargs):return self.motion('resume',kwargs)
    def next_episode(self,**kwargs):return self.motion('next_episode',kwargs)
    def terminal(self,outcome,**kwargs):
        if kwargs.get('home_after_terminal',True) and self._home_after_terminal and not self._hooks.is_policy_mode():
            from methods.openpi_rlt.cobot_adapter.session import SessionConflict
            raise SessionConflict('release_hil_first','Release the teach button before terminal homing')
        return super().terminal(outcome,**kwargs)
    def status(self):
        result=super().status()
        provider=getattr(self,'_storage_provider',None)
        if provider is not None:result.update(provider(result))
        path=RUN/'learning/operation.json'
        try:
            result['online_update']=json.loads(path.read_text())
            result['learner_version']=result['online_update'].get('learner_final_step')
            result['next_actor_version']=result['online_update'].get('actor_version')
            if result['online_update'].get('phase') in ('preparing','training') and not self.learning_busy():
                result['online_update']['phase']='interrupted'
        except (OSError,ValueError):result['online_update']={'phase':'idle','automatic_warmup':False}
        return result
class Runtime:
    def __init__(self,args):
        self.args=args;self.tracking_bound=.04
        self.v5=(RUN/'learning/rtc-v5/current.json').exists();self.v5_worker=None;self.v5_release=None
        self.v4=(self.v5 or (RUN/'learning/v4/current.json').exists()) and args.actor!='corrective'
        self.frozen_release=None
        self.queue=RTCQueue(tracking_joints=np.arange(7,13));self.filter=(
            CommandFilter(velocity=.10,acceleration=.9) if args.actor=='corrective' or self.v4 else CommandFilter())
        self.passive_commands=PassiveCommandLatch()
        if args.actor=='corrective' or self.v4:
            print('[fixed-conditioning] control=30Hz inference=async-RTC velocity=0.10rad/s acceleration=0.9rad/s2 tracking=right-only bound=0.04rad limits=Piper',flush=True)
        self.end=threading.Event();self.error=None;self.rows=[];self.plans=[];self.frame_lock=threading.RLock()
        self.actor=None;self.actor_version=-1;self.future=None;self.chunks=0;self.next_health=time.monotonic()
        self.deadline_recoveries=0;self.episode_deadline_start=0
        self.deadline_budget=DeadlineRecoveryBudget(max_reanchors=3)
        self.deadline_reanchor_pending=False
        self.scene_saved=False;self.scene_snapshot=None
        self.workspace_guard=None
        self.max_policy_commands=360 if args.actor=='corrective' or self.v4 else None
        if self.v4:print('[v4] 30Hz async RTC; .10rad/s .9rad/s2; workspace guard; 360 autonomous commands (12s); HIL release stays paused',flush=True)
        self.episode_policy_commands=0
        self.health_future=None;self.health_started=None;self.health_timeout_reported=False
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='rtc-inference')
        self.health_worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='recorder-health')
        self.rpc=WebsocketClientPolicy(host='127.0.0.1',port=8020)
        meta=self.rpc.get_server_metadata()
        for key,value in {'cohort':'plug_v2','control_hz':30,'rtc_max_delay':6,'action_dim':14,'z_dim':2048}.items():
            if meta.get(key)!=value:raise ValueError('model contract mismatch '+key)
        self.io=RTCIO(self.queue,shadow_mode=args.shadow,recording_enabled=False,
               trace_dir=RUN/'traces',prompt='Insert the held plug into the socket.')
        self.data_phase='warmup' if args.actor=='reference' else 'online'
        self.data_root=selected_root(self.data_phase)
        self.data_root.mkdir(parents=True,exist_ok=True)
        self.model_meta=meta
        self.client=FlatRecorderClient('http://127.0.0.1:8015/api/rlt-recorder',timeout_sec=15,
                                      min_free_bytes=4*1024**3)
        self.app=GuardedApplication(CohortSessionController(),self.client,
               identity_factory=self.episode_identity,hooks=SessionHooks(
                 is_policy_mode=self.io.is_policy_mode,set_policy_paused=self.io.set_policy_paused,
                 submit_outcome=lambda _:None,signal_episode_ready=self.start_episode,
                 request_front_home=self.home,arm_policy=self.arm),
               home_after_terminal=not args.shadow,recording_enabled=not args.no_record and not args.shadow)
        self.app._storage_provider=self.storage_status
        self.app.update_metrics(shadow_mode=args.shadow,actor_version=-1)
        self.server=RltSessionHttpServer(self.app,host='127.0.0.1',port=8026)
        self.io.attach_session(self.app,self.server)
        if self.v5 and args.actor!='reference':
            from .rtc_release import selected as select_release
            from .rtc_actor_client import ensure_worker,select
            release,info=select_release();self.v5_worker=ensure_worker()
            select(release)  # Restore/compile before the UI can arm policy.
            print('[upstream-rlt-ready] candidate='+info['name']+' step='+str(info['global_step']),flush=True)
    def episode_identity(self,_):
        self.data_root=selected_root(self.data_phase)
        self.data_root.mkdir(parents=True,exist_ok=True)
        return Task5EpisodeIdentity(task_id='plug_insertion',model_id='plug_v2',
          checkpoint_id=Path(self.model_meta['checkpoint']).name,dataset_round='plug_v2',
          data_root=str(self.data_root),max_timesteps=3600)
    def storage_status(self,status):
        phase=status['phase']
        if phase in ('disarmed','ready','armed','waiting_scene','stopped'):
            self.data_root=selected_root(self.data_phase)
        kept=history(self.data_root)
        number=len(kept)
        if phase not in ('disarmed','ready','armed','waiting_scene','stopped','fault') and status.get('outcome')!='aborted':
            if status.get('task5_episode_uuid') not in {d['episode_uuid'] for d in kept}:number+=1
        return {'data_root':str(self.data_root),'data_phase':self.data_phase,
                'actor_mode':self.args.actor,'actor_candidate':('upstream-rlt-rtc-v5' if self.v5 and self.args.actor!='reference' else 'supported-iql-r1' if self.v4 and self.args.actor!='reference' else 'rtc-corrective-r1' if self.args.actor=='corrective' else None),'recorded_episode_count':len(kept),
                'episode_display_number':number,
                'rtc_deadline_recoveries':self.deadline_recoveries,
                'model_rpc_reconnects':self.rpc.reconnect_count,
                'runtime_profile':{'revision':'r12-bounded-deadline-reanchor-20260920',
                  'control_hz':30,'inference':'async_rtc','rtc_delay_steps':6,
                  'learning_algorithm':'pinned_upstream_rlt_rtc_smdp_v1' if self.v5 else 'supported_iql_residual_v1' if self.v4 else 'legacy',
                  'rtc_deadline_policy':'bounded_auto_reanchor_3_then_fail_closed',
                  'velocity_rad_per_sec':.10 if self.args.actor=='corrective' or self.v4 else .2,
                  'acceleration_rad_per_sec2':.9 if self.args.actor=='corrective' or self.v4 else None,
                  'tracking_bound_rad':self.tracking_bound,'joint_limits':'Piper',
                  'workspace_delta_min_m':PIPER_WORKSPACE_MIN.tolist(),
                  'workspace_delta_max_m':PIPER_WORKSPACE_MAX.tolist(),
                  'workspace_radius_m':float(PIPER_WORKSPACE_RADIUS),
                  'max_policy_commands':self.max_policy_commands}}
    def start_episode(self):
        with self.frame_lock:self.rows=[];self.plans=[]
        self.passive_commands.reset()
        self.deadline_budget.reset_episode()
        self.deadline_reanchor_pending=False
        self.episode_deadline_start=self.deadline_recoveries
        self.scene_saved=False;self.scene_snapshot=None
        self.workspace_guard=None
        self.episode_policy_commands=0
        if self.args.actor=='corrective':
            from methods.openpi_rlt.plug_v2.fixed_candidate import select_fixed_candidate
            self.actor,self.actor_version=select_fixed_candidate()
            from methods.openpi_rlt.plug_v2.fixed_candidate import manifest as fixed_manifest
            print('[fixed-candidate] '+json.dumps({k:fixed_manifest()[k] for k in ('name','sha256','version')}),flush=True)
        elif self.args.actor!='reference':
            if self.v5:
                from .rtc_release import selected as select_release
                from .rtc_actor_client import select
                if self.args.actor=='warmup' and self.v5_release is not None:release,info=self.v5_release
                else:release,info=select_release()
                if self.args.actor=='warmup':self.v5_release=(release,info)
                self.actor,self.actor_version=select(release)
                self.app.update_metrics(actor_version=self.actor_version)
                print('[upstream-rlt-episode] '+json.dumps({'candidate':info['name'],'global_step':self.actor_version,'checkpoint_sha256':info['checkpoint_sha256'],'fixed':self.args.actor=='warmup'}),flush=True)
                return
            from methods.openpi_rlt.plug_v2.actor_service import select_actor
            path=RUN/'learning'/('warmup/actor.pt' if self.args.actor=='warmup' else 'online/actor.pt')
            if self.args.actor=='latest' and not path.exists():path=RUN/'learning/warmup/actor.pt'
            if self.v4:
                from .online_release import selected
                if self.args.actor=='warmup':
                    if self.frozen_release is None:self.frozen_release=selected()
                    path,release=self.frozen_release
                else:path,release=selected()
                print('[v4-actor] '+json.dumps(release),flush=True)
            self.actor,self.actor_version=select_actor(path)
            if self.v4 and (self.actor.key!=release['sha256'] or self.actor_version!=release['actor_version']):
                raise RuntimeError('selected actor hash/version mismatch')
        else:self.actor=None;self.actor_version=-1
        self.app.update_metrics(actor_version=self.actor_version)
    def arm(self):
        # Public console exposes recorder/ROS readiness before any active policy publisher.
        from urllib.request import ProxyHandler,build_opener
        with build_opener(ProxyHandler({})).open('http://127.0.0.1:8015/api/console/status',timeout=3) as response:
            status=json.load(response)
        readiness=status.get('ros_readiness',{})
        if not self.args.shadow and readiness.get('status') != 'ok':
            raise RuntimeError('robot/recorder not ready: '+str(readiness.get('error_code')))
        self.io.arm_policy()
    def home(self):
        if self.args.shadow:raise RuntimeError('shadow cannot home')
        p=subprocess.run([str(PLATFORM/'scripts/home.sh'),'all','--pose','plug'],
                         text=True,capture_output=True,timeout=120,check=False)
        print(p.stdout,flush=True)
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    def prepare_request(self):
        obs=self.io.observation()
        if (self.args.actor=='corrective' or self.v4) and self.workspace_guard is None:
            self.workspace_guard=PiperWorkspaceGuard(obs['state'])
        req=self.queue.request(obs['state'])
        self.future=self.worker.submit(self.infer,req,obs)
        if not self.scene_saved:self.save_scene(obs)
    def save_scene(self,obs):
        self.scene_saved=True
        try:
            status=self.app.snapshot()
            folder=RUN/'scene-snapshots'/status.session_id;folder.mkdir(parents=True,exist_ok=True)
            path=folder/('episode_'+str(status.episode_id)+'.npz')
            if path.exists():raise FileExistsError('scene snapshot already exists')
            images=obs['images']
            with path.open('xb') as f:
                np.savez_compressed(f,state=np.asarray(obs['state'],np.float32),
                  camera_high=np.asarray(images['base_0_rgb'],np.uint8),
                  camera_left=np.asarray(images['left_wrist_0_rgb'],np.uint8),
                  camera_right=np.asarray(images['right_wrist_0_rgb'],np.uint8),
                  timestamp=np.asarray(time.time()))
                f.flush();os.fsync(f.fileno())
            self.scene_snapshot=str(path)
            print('[scene-snapshot] '+str(path),flush=True)
        except Exception as error:
            # Snapshot evidence is diagnostic; never delay or release motion
            # based on an optional evidence write.
            print('[scene-snapshot] unavailable: '+str(error),flush=True)
    def infer(self,req,obs):
        obs={**obs,'action_prefix':req.prefix.copy(),'prefix_length':req.prefix_length}
        start=time.perf_counter();result=self.rpc.infer(obs);model_done=time.perf_counter();d=req.prefix_length
        if not self.queue.is_current(req):
            elapsed=time.perf_counter()-start
            print('[rtc-timing] '+json.dumps({'tick':req.start_tick,'prefix':d,'model_ms':round(1000*(model_done-start),2),'total_ms':round(1000*elapsed,2),'accepted':False,'discarded':'stale_generation','server_timing':result.get('policy_timing'),'generation':req.generation}),flush=True)
            return False,elapsed
        raw=np.asarray(result['ref_chunk'],np.float32).copy()
        if raw.shape!=(50,14) or not np.array_equal(raw[:d],req.prefix[:d]):
            raise ValueError('server modified committed prefix')
        # RTC context makes the committed actions part of the actor/critic state.
        context=np.r_[req.state,(req.prefix[:6]-req.state).reshape(-1) if d else np.zeros(84),d/6].astype(np.float32)
        if self.actor is not None:
            future=self.actor.act(np.asarray(result['z_rl'],np.float32),context,raw[d:d+10])
            raw[d:d+10,7:13]=future[:,7:13]
        if self.args.explore:
            # One sampled perturbation per chunk; locked prefix and passive joints unchanged.
            raw[d:d+10,7:13]+=np.random.default_rng().normal(0,.001,(1,6)).astype(np.float32)
        actor_done=time.perf_counter()
        proposed=raw.copy()
        if self.v5 and self.actor is not None:
            from .chunk_projection import project_correction
            proposed[d:d+10,7:14]=project_correction(raw[d:d+10,7:14],np.asarray(result['ref_chunk'])[d:d+10,7:14],degree=1)
        passive_commands=self.passive_commands.update(proposed,d)
        conditioned=self.filter.plan(proposed,req.state,req.prefix,d,passive_commands=passive_commands)
        if self.workspace_guard is not None:
            self.workspace_guard.validate_plan(conditioned,d,d+self.queue.chunk)
        from .decision_trace import decision_evidence
        evidence=decision_evidence(req,np.asarray(result['ref_chunk']),raw,conditioned,
                                  actor_key=getattr(self.actor,'key',None))
        evidence['projected_plan']=proposed.copy();evidence['projection']='linear_actor_minus_reference' if self.v5 and self.actor is not None else 'none'
        ready=time.perf_counter()
        accepted=self.queue.complete(req,conditioned)
        elapsed=time.perf_counter()-start
        if elapsed>.15 or not accepted:
            print('[rtc-timing] '+json.dumps({'tick':req.start_tick,'prefix':d,'model_ms':round(1000*(model_done-start),2),'actor_ms':round(1000*(actor_done-model_done),2),'conditioning_guard_ms':round(1000*(ready-actor_done),2),'total_ms':round(1000*elapsed,2),'accepted':accepted,'server_timing':result.get('policy_timing'),'generation':req.generation}),flush=True)
        if accepted:
            record={'generation':req.generation,'sequence':req.sequence,'tick':req.start_tick,
                  'timestamp':time.time(),'latency_sec':elapsed,'state':req.state.copy(),'prefix':req.prefix[:6].copy(),
                  'prefix_length':d,'context':context,'ref':np.asarray(result['ref_chunk'])[d:d+10].copy(),
                  'z':np.asarray(result['z_rl'],np.float32),'actor_version':self.actor_version,**evidence}
            with self.frame_lock:self.plans.append(record)
        return accepted,time.perf_counter()-start
    def finalize(self):
        status=self.app.status()
        with self.frame_lock:rows=list(self.rows);plans=list(self.plans)
        metadata={'cohort':'plug_v2','session_id':status['session_id'],'episode_id':status['episode_id'],
                  'episode_uuid':status.get('task5_episode_uuid'),'episode_index':status.get('task5_episode_index'),
                  'outcome':status['outcome'],'actor_version':self.actor_version,
                  'actor_candidate':('upstream-rlt-rtc-v5' if self.v5 and self.args.actor!='reference' else 'supported-iql-r1' if self.v4 and self.args.actor!='reference' else 'rtc-corrective-r1' if self.args.actor=='corrective' else None),
                  'recording_enabled':status['recording_enabled'],'shadow':self.args.shadow,
                  'data_root':str(self.data_root),'data_phase':self.data_phase,'control_hz':30,'rtc_max_delay':6,
                  'runtime_revision':'r12-bounded-deadline-reanchor-20260920',
                  'decision_trace_schema':2,
                  'compact_trace_retained':True,
                  'scene_snapshot':self.scene_snapshot,
                  'rtc_deadline_misses':self.deadline_recoveries-self.episode_deadline_start,
                  'policy_command_count':self.episode_policy_commands,
                  'policy_frames':sum(x['valid_for_training'] and x['phase']=='rollout' for x in rows),
                  'hil_frames':sum(x['valid_for_training'] and x['phase']=='hil' for x in rows),
                  'inference_chunks':len(plans)}
        folder=RUN/'traces'/status['session_id'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/('episode_'+str(status['episode_id'])+'.npz')
        if path.exists():raise FileExistsError('trace already finalized')
        # Arrays are small; large images remain in recorder until validated conversion.
        serial=lambda x:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in x.items()}
        payload={'metadata':np.array(json.dumps(metadata)),
                 'frames_json':np.array(json.dumps([serial(x) for x in rows])),
                 'plans_json':np.array(json.dumps([serial(x) for x in plans]))}
        with path.open('xb') as f:
            np.savez_compressed(f,**payload);f.flush();os.fsync(f.fileno())
        if status['recording_enabled']:
            atomic(self.data_root/(f"episode_{status['task5_episode_index']:06d}.rlt.json"),
                   {**metadata,'trace':str(path)})
        self.app.mark_replay_finalized()
        if status['recording_enabled'] and status['outcome'] in ('success','failure','aborted'):
            # Background conversion/extraction is supervised independently of the control loop.
            cmd=[str(ROOT/'envs/rlt-online-py310/bin/python'),'-m',
                 'methods.openpi_rlt.plug_v2.data_worker','--root',str(self.data_root),
                 '--episode-index',str(status['task5_episode_index'])]
            if status['outcome']=='aborted':cmd+=['--discard']
            log=RUN/'conversion.log'
            with log.open('a') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(RUN/'backend/conversions'/(str(status.get('task5_episode_uuid'))+'.json'),{'pid':child.pid,'command':cmd,'start_ticks':int((Path('/proc')/str(child.pid)/'stat').read_text().split(') ')[1].split()[19])})
    def run(self):
        self.server.start()
        print('SESSION_READY disarmed http://127.0.0.1:8015/',flush=True)
        from .control_clock import next_deadline
        tick=time.monotonic();last_publish=None
        try:
            while not self.end.is_set() and not self.io.ros.is_shutdown():
                phase=self.app.snapshot().phase.value
                if phase=='stopped':
                    self.end.wait(.1);tick=time.monotonic();continue
                if phase=='replay_committing':self.finalize();continue
                if self.future is not None and self.future.done():
                    try:
                        accepted,latency=self.future.result()
                        if accepted:
                            self.chunks+=1
                            self.app.update_metrics(chunk_count=self.chunks,last_inference_latency_sec=latency,
                                                    actor_version=self.actor_version)
                            if self.chunks==1 or self.chunks%5==0:
                                print(f'[plug-v2] chunk={self.chunks} latency={latency:.3f}s actor={self.actor_version}',flush=True)
                    except Exception as error:
                        if self.deadline_reanchor_pending:
                            print('[rtc-deadline] stale late inference error discarded: '+str(error),flush=True)
                        elif phase=='rollout':
                            self.io.set_policy_paused(True)
                            self.app.mark_terminal_pending('inference_error: '+str(error))
                    self.future=None
                    if self.deadline_reanchor_pending and phase=='rollout':
                        self.queue.resume()
                        self.deadline_reanchor_pending=False
                        print(f'[rtc-deadline] fresh d0 replan started recovery={self.deadline_budget.count}/{self.deadline_budget.max_reanchors}',flush=True)
                if self.health_future is not None and self.health_future.done():
                    try:self.health_future.result()
                    except Exception as error:
                        if phase in ('rollout','hil'):
                            self.io.set_policy_paused(True)
                            self.app.mark_terminal_pending('recorder_health_error: '+str(error))
                    self.health_future=None;self.health_started=None;self.health_timeout_reported=False
                elif (self.health_future is not None and self.health_started is not None and
                      not self.health_timeout_reported and time.monotonic()-self.health_started>1.0 and
                      phase in ('rollout','hil')):
                    # Keep recorder integrity fail-closed without allowing a
                    # slow HTTP health request to stall the servo loop.
                    self.health_timeout_reported=True
                    self.io.set_policy_paused(True)
                    self.app.mark_terminal_pending('recorder_health_timeout')
                if phase in ('rollout','hil','paused','terminal_pending'):
                    state=None;command=None
                    try:
                        state=self.io.measured()
                        if phase=='rollout' and self.queue.active:
                            if self.workspace_guard is not None:self.workspace_guard.validate_measured(state)
                            if self.max_policy_commands is not None and self.episode_policy_commands>=self.max_policy_commands:
                                raise QueueFault(f'policy_command_budget_exceeded: commands={self.episode_policy_commands} limit={self.max_policy_commands}')
                            if self.future is None and self.queue.request_due:self.prepare_request()
                            try:
                                command=self.io.publish_next(tracking_bound=self.tracking_bound)
                                if command is not None:last_publish=time.monotonic()
                            except QueueFault as error:
                                reason=str(error)
                                if reason=='rtc_deadline_missed':
                                    self.deadline_recoveries+=1
                                    if self.deadline_budget.record_miss():
                                        # Keep the last published command while the stale
                                        # future drains, then restart RTC from a fresh d0
                                        # observation. Passive commands stay latched to the
                                        # operator-defined episode start.
                                        self.deadline_reanchor_pending=True
                                        print(f'[rtc-deadline] late plan discarded; holding command before fresh d0 replan recovery={self.deadline_budget.count}/{self.deadline_budget.max_reanchors}',flush=True)
                                    else:
                                        print(f'[rtc-deadline] recovery budget exhausted; pausing episode count={self.deadline_recoveries}',flush=True)
                                        self.io.set_policy_paused(True)
                                        self.app.mark_terminal_pending('rtc_deadline_missed: recovery budget exhausted')
                                elif reason!='waiting_initial_inference':raise
                            if command is not None:self.episode_policy_commands+=1
                    except Exception as error:
                        if phase=='rollout':
                            self.io.set_policy_paused(True)
                            self.app.mark_terminal_pending('command_paused: '+str(error))
                    if state is not None:
                        with self.frame_lock:self.rows.append({'timestamp':time.time(),
                          'ros_timestamp':self.io.ros.Time.now().to_sec(),'state':state.copy(),
                          'command':None if command is None else command.copy(),'mode':self.io._mode,
                          'phase':phase,'tick':self.queue.next_tick,'generation':self.queue.generation,
                          'valid_for_training':(phase=='hil' and self.io._mode.startswith('manual:')) or (phase=='rollout' and command is not None)})
                    if time.monotonic()>=self.next_health:
                        if self.health_future is None:
                            self.health_future=self.health_worker.submit(self.app.ensure_recorder_active)
                            self.health_started=time.monotonic();self.health_timeout_reported=False
                        self.next_health=time.monotonic()+1
                now=time.monotonic()
                tick=next_deadline(tick,now,last_publish)
                self.end.wait(max(0.,tick-time.monotonic()))
        finally:
            self.io.set_policy_paused(True)
            self.server.shutdown()
            self.rpc.close()
            if self.v5_worker is not None and self.v5_worker.poll() is None:
                self.v5_worker.terminate()
                try:self.v5_worker.wait(timeout=10)
                except subprocess.TimeoutExpired:self.v5_worker.kill();self.v5_worker.wait(timeout=5)
            self.worker.shutdown(wait=False,cancel_futures=True)
            self.health_worker.shutdown(wait=False,cancel_futures=True)
            print('SESSION_STOPPED; model remains loaded',flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--actor',choices=['reference','warmup','latest','corrective'],default='reference')
    p.add_argument('--explore',action='store_true');p.add_argument('--shadow',action='store_true')
    p.add_argument('--no-record',action='store_true')
    args=p.parse_args()
    if args.actor=='corrective' and args.explore:p.error('fixed corrective trial does not support exploration')
    runtime=Runtime(args)
    signal.signal(signal.SIGTERM,lambda *_:runtime.end.set())
    signal.signal(signal.SIGINT,lambda *_:runtime.end.set())
    runtime.run()
if __name__=='__main__':main()
