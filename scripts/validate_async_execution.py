#!/usr/bin/env python3
"""Actual frozen Stage1 + Actor, synthetic feedback, zero robot publishers.

Timing and command continuity audit only; this does not measure real insertion.
Does not initialize ROS, use hardware factories, bind servers or write Replay.
"""
import argparse,copy,hashlib,json,os,pickle,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'envs/machine-a-py311-overlay'),
              str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),
              str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np


def execute_probe_episode(env, feature_provider, actor, cfg):
    """Use the native episode planner for faithful execution; never write Replay."""
    from methods.openpi_rlt.cobot_adapter.online_runtime import install_bimanual_runtime_patch
    install_bimanual_runtime_patch()
    from rlt_online_rl.config import EnvDriverConfig
    from rlt_online_rl.inference import EnvDriver
    driver=EnvDriver(env,feature_provider,actor,None,cfg,
        EnvDriverConfig(control_frequency_hz=20,chunk_exec_horizon=10,
                        actor_deterministic=True,safe_fallback_to_ref=False,
                        enable_human_override=False),eval_actor_only=True)
    engine=getattr(env,'_execution',None)
    if engine is not None:
        from methods.openpi_rlt.cobot_adapter.execution_runtime import PlannerBackend
        engine.backend=PlannerBackend(driver)
        engine.set_episode(1)
    return driver.run_episode(1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--guard-web-url',default='http://127.0.0.1:8015')
    p.add_argument('--prompt', help='Explicit task prompt; omitted preserves the historical Stage1 default.')
    p.add_argument('--profiles', nargs='+', choices=('faithful20','async_rtc20','async_rtc30','async_rtc40','async_rtc50','async40_no_rtc','async40_no_smoothing'),
                   help='Select execution comparisons; omitted preserves all seven historical variants.')
    args=p.parse_args()
    if args.prompt is not None and not args.prompt.strip():
        p.error('--prompt must be nonempty')
    from audit_critic_guidance import check_idle
    check_idle(args.guard_web_url)
    from methods.openpi_rlt.plug_v3_yyshadow.serve_stage1 import load
    overlay=ROOT.parent/'vla-platform/integrations/cobot/pi05/dagger/common/rtc_overlay'
    policy_kwargs = {} if args.prompt is None else {'default_prompt': args.prompt}
    policy=load(ROOT,args.checkpoint,rtc_overlay=overlay,**policy_kwargs)
    import cv2,h5py,yaml,jax,jax.numpy as jnp
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from rlt_online_rl.inference import RLTPolicyInferenceWrapper,ActorResponse
    from methods.openpi_rlt.cobot_adapter.execution_profiles import ExecutionProfile
    from methods.openpi_rlt.cobot_adapter.async_execution import AsyncExecution
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv
    config_mapping=yaml.safe_load(args.config.read_text())
    cfg=RLTOnlineRLConfig(**config_mapping['experiment']['rl'])
    adapter=ActionRepresentationAdapter.from_config(cfg)
    snapshot=args.actor.read_bytes();payload=pickle.loads(snapshot)
    params=jax.tree_util.tree_map(jnp.asarray,payload['actor_params'])
    wrapper=RLTPolicyInferenceWrapper(cfg)
    with h5py.File(args.recording) as f:
        state=np.asarray(f['observations/qpos'][0],np.float32)[-7:]
        images={}
        for dst,src in [('base_0_rgb','cam_high'),('left_wrist_0_rgb','cam_left_wrist'),('right_wrist_0_rgb','cam_right_wrist')]:
            img=np.asarray(f['observations/images'][src][0])
            if img.ndim==1:img=cv2.cvtColor(cv2.imdecode(img,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            images[dst]=img
    observation=dict(state=state,images=images)
    class Features:
        def get_features(self,obs):return policy.infer(obs)
    class Actor:
        def infer(self,request):
            ref=adapter.normalize_ref_chunk(request.ref_chunk,request.proprio)
            result=wrapper.infer(params,request.z_rl,request.proprio,ref,deterministic=True)
            result=adapter.denormalize_to_abs_chunk(result,request.proprio)
            return ActorResponse(result,int(payload['version']),request.request_id)
    actor=Actor()
    prewarm=[]
    for _ in range(3):
        before=policy.infer(observation)
        prewarm.append(before['policy_timing']['infer_ms'])
        request=dict(observation,rtc=dict(previous_actions=before['ref_chunk'][:5].copy(),delay_steps=4,execution_horizon=5))
        guided=policy.infer(request)
        assert guided['rtc_used']
        from rlt_online_rl.inference import ActorRequest
        actor.infer(ActorRequest(guided['z_rl'],guided['proprio'],guided['ref_chunk'],'warmup',-1,0,True))
    report=dict(schema=1,checkpoint=str(args.checkpoint),actor=str(args.actor),
        actor_sha256=hashlib.sha256(snapshot).hexdigest(),actor_version=int(payload['version']),
        recording=str(args.recording),robot_publishers=0,learner_updates=0,
        baseline_prewarm_ms=prewarm,profiles={},policy_metadata=policy.metadata,
        devices=[str(d) for d in jax.devices()],backend=jax.default_backend(),
        native_driver_contract=dict(control_frequency_hz=20,chunk_exec_horizon=10,eval_actor_only=True,enable_human_override=False,collection_phase="online",adapter_runtime_patch=True),
        config_path=str(args.config),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        algorithm_config=config_mapping['experiment']['rl'],requested_profiles=args.profiles,
        requested_prompt=args.prompt,limitations=[
        'Perfect synthetic joint tracking and one recorded camera frame; no real dynamics or insertion success.',
        'Native episode success is only a synthetic terminal placeholder; not a task outcome. RTC stats flags do not prove data/training.',
        'In-process Stage1/Actor calls exclude production RPC overhead and online Learner contention.',
        'RTC-conditioned online Replay training does not mean Stage1 training-time RTC fine-tuning.'])
    class IO:
        shadow_mode=False
        def __init__(self):self.state=state.copy();self.published=[];self.records=[];self.latencies=[]
        def sample(self):
            return SimpleNamespace(observation=dict(state=self.state.copy(),images=images),
                mode='policy',paused=False,outcome=None,timestamp=time.monotonic())
        def publish_policy_action(self,action):
            self.state=np.asarray(action).copy();self.published.append((time.monotonic(),self.state.copy()));return True
        def wait_armed(self):pass
        def mark_replay_finalized(self):pass
        def mark_terminal_pending(self,reason):pass
        def wait_terminal_outcome(self):return "failure"  # Synthetic budget stop, not task evidence.
        def set_chunk_ready(self,ready):pass
        def set_policy_paused(self,paused):pass
        def record_raw_step(self,record):self.records.append(record)
        def report_chunk(self,latency_sec,actor_version):self.latencies.append(latency_sec)
    os.environ['COBOT_RLT_EXECUTION_PROFILE']='faithful20'
    variants=[("faithful20",20,None)]+[(f"async_rtc{hz}",hz,ExecutionProfile(publish_hz=hz)) for hz in (20,30,40,50)]
    variants += [("async40_no_rtc",40,ExecutionProfile(publish_hz=40,rtc=False)),
                 ("async40_no_smoothing",40,ExecutionProfile(publish_hz=40,smoothing_tau_sec=0.))]
    if args.profiles is not None:
        variants=[v for v in variants if v[0] in args.profiles]
    for name,hz,profile in variants:
        io=IO()
        env=RightArmCobotOnlineEnv(io,chunk_exec_horizon=10,control_frequency_hz=20,max_episode_steps=30,collection_phase="online",
            joint_step_limit=.03,gripper_step_limit=.004,sleep=time.sleep)
        engine=AsyncExecution(env,name,profile) if profile else None
        env._execution=engine
        try:
            episode=execute_probe_episode(env,Features(),actor,cfg)
            assert episode['eval_actor_only'] and episode['transitions_written']==0
            assert episode['actor_version_start']==episode['actor_version_end']==int(payload['version']), 'Probe must exercise the supplied Actor'
            assert len(io.records)==30, 'Synthetic episode must contain exactly30 logical steps'
            commands=np.stack([a for _,a in io.published]);times=np.array([t for t,_ in io.published])
            velocities=np.diff(commands[:,:6],axis=0)/np.diff(times)[:,None]
            row=dict(status='passed',logical_steps=len(io.records),commands=len(io.published),
                native_episode_summary=episode,synthetic_outcome=True,task_outcome=None,
                publication_elapsed_sec=(times-times[0]).tolist(),publication_intervals_ms=(np.diff(times)*1000).tolist(),
                commanded_right_arm=commands.tolist(),
                measured_hz=float(1/np.diff(times).mean()),interval_p95_ms=float(np.quantile(np.diff(times),.95)*1000),
                inference_ms=[1000*x for x in io.latencies],velocity_rms=float(np.sqrt(np.mean(velocities**2))),
                command_step_max=float(np.max(np.abs(np.diff(commands[:,:6],axis=0)))),
                execution=dict(engine.stats) if engine else {"profile":"faithful20"})
        except Exception as exc:
            row=dict(status='failed',error=str(exc),logical_steps=len(io.records),commands=len(io.published),
                     inference_ms=[1000*x for x in io.latencies],execution=dict(engine.stats) if engine else {"profile":"faithful20"})
        finally:
            if engine:engine.close()
        report['profiles'][name]=row
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    if hashlib.sha256(args.actor.read_bytes()).hexdigest()!=report['actor_sha256']:
        raise RuntimeError('Actor source changed during audit')
    report['finished_at']=time.time();args.output.write_text(json.dumps(report,indent=2)+'\n')
    if any(x['status']!='passed' for x in report['profiles'].values()):raise SystemExit(1)
if __name__=='__main__':main()
