#!/usr/bin/env python3
"""CPU native HIL/input -> private Replay RPC -> real 5k learner -> staged Actor.

All experience and features are synthetic. Verifies data/update identities, not
robot execution, Stage1 representations or improved task ability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def load_rows(path):
    rows = []
    with Path(path).open('rb') as stream:
        while True:
            try:
                rows.append(pickle.load(stream))
            except EOFError:
                return rows


def configure():
    os.environ.update(COBOT_RLT_REPLAY_ACTION_PRECISION='float32',
                      COBOT_RLT_HIL_TARGET='coordinator_command',
                      COBOT_RLT_HIL_SAMPLING='logical20',
                      COBOT_RLT_RAW_OBSERVATION_CONTRACT='trace',
                      COBOT_RLT_INPUT_AUDIT='strict',
                      COBOT_RLT_EXECUTION_PROFILE='faithful20')
    for key in ['COBOT_RLT_EXPERIMENT_PROFILE', 'COBOT_RLT_EXECUTION_OPTIONS',
                'COBOT_EXECUTION_OPTIONS', 'RLT_DISABLE_LEARNER']:
        os.environ.pop(key, None)


def learner(config_path, output):
    import numpy as np
    import jax
    import yaml
    from methods.openpi_rlt.cobot_adapter.process_bootstrap import initialize_process
    from methods.openpi_rlt.cobot_adapter.input_audit import array_receipt
    initialize_process(str(config_path))
    from rlt_online_rl import trainer
    from rlt_online_rl.config import system_config_from_mapping
    from rlt_online_rl.replay import ReplayClient
    config = system_config_from_mapping(yaml.safe_load(config_path.read_text()))
    original = trainer.train_step
    def audited(state, batch, **kwargs):
        # Actual normalized JAX arrays at the real train_step boundary.
        rows = []
        for i, phase in enumerate(np.asarray(batch['collection_phase_id'])):
            if int(phase) == 2:
                rows.append({'episode_id': int(batch['episode_id'][i]),
                             'step_id': int(batch['step_id'][i]),
                             'arrays': {k: array_receipt(np.asarray(batch[k][i])) for k in
                                        ['z_rl','proprio','action_chunk','ref_chunk','rewards',
                                         'next_z_rl','next_proprio','next_ref_chunk','source_chunk','done']}})
        new, metrics = original(state, batch, **kwargs)
        with (output/'actual_train_inputs.jsonl').open('a') as stream:
            stream.write(json.dumps({'global_step': int(new.global_step),
                                     'actor_version': int(new.actor_version),
                                     'actor_updated': int(new.actor_version) > int(state.actor_version),
                                     'sample_count': len(batch['episode_id']),
                                     'new_experience_draws': rows,
                                     'finite_metrics': all(np.isfinite(float(v)) for v in jax.device_get(metrics).values())})+'\n')
        return new, metrics
    trainer.train_step = audited
    replay = ReplayClient(config.learner_service.replay_url, timeout_sec=3.)
    service = trainer.LearnerService(config.rl, config.learner_service, replay,
                                    metrics_path=str(output/'metrics/learner_metrics.jsonl'))
    steps = []
    while len(steps) < 16:
        result = service.train_once()
        if result is None:
            break
        steps.append(result)
    assert len(steps) == 15 and int(service.state.global_step) == 5015
    assert int(service.state.actor_version) == 2507
    before_flush = pickle.loads(Path(config.learner_service.actor_snapshot_path).read_bytes())
    assert int(before_flush['version']) == 2500
    service.flush_artifacts()
    restart = trainer.LearnerService(config.rl, config.learner_service, replay,
                                    metrics_path=str(output/'restart_metrics.jsonl'))
    assert restart.train_once() is None
    for left, right in zip(jax.tree_util.tree_leaves(service.state), jax.tree_util.tree_leaves(restart.state)):
        np.testing.assert_array_equal(left, right)
    dump(output/'learner_report.json', {'updates': len(steps), 'step':5015, 'actor':2507,
                                      'candidate_export_before_flush':int(before_flush['version']),
                                      'candidate_export_after_flush':2507,
                                      'periodic_export_interval_updates':config.learner_service.push_actor_interval_steps,
                                      'restart_exact':True, 'pending_updates':0,
                                      'metrics_finite':all(np.isfinite(float(v)) for row in steps for v in row.values())})


class SyntheticIO:
    shadow_mode = False
    def __init__(self, root):
        self._trace_writer = SimpleNamespace(_root=root/'trace')
        self.count = 0
        self.records = []
        self.publications = []
        self.finalized = False
    def sample(self):
        import numpy as np
        from methods.openpi_rlt.cobot_adapter.cobot_ros1 import CobotIOSample
        from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome
        n = self.count; self.count += 1
        state = np.full(7, .0001*n, np.float32); state[6] = .004
        observation = {'state':state, 'prompt':'Insert the held plug into the socket.',
                       'images': {key:np.full((2,2,3), n, np.uint8) for key in
                                  ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']}}
        now = time.perf_counter()
        evidence = {'mode':'manual:right', 'captured_ros_time':float(n),
                    'captured_monotonic':now, 'right_takeover_started_monotonic':0.,
                    'coordinator_commands':{'right':{'target':(state+np.asarray([.000123]*6+[.000001],np.float32)).tolist(),
                        'valid':True,'topic':'/master/joint_right','ros_stamp':float(n),
                        'arrival_monotonic':now},'left':{'valid':False}}}
        return CobotIOSample(observation=observation, mode='manual:right',
                             outcome=EpisodeOutcome.SUCCESS if n == 60 else None,
                             paused=True, timestamp=float(n), io_evidence=evidence)
    def wait_armed(self): pass
    def wait_episode_ready(self): pass
    def set_chunk_ready(self, _): pass
    def report_chunk(self, *_): pass
    def mark_replay_finalized(self): self.finalized=True
    def record_raw_step(self, row): self.records.append(row)
    def publish_policy_action(self, row):
        self.publications.append(row)
        raise AssertionError('Synthetic HIL must not publish policy commands')


class SyntheticFeatures:
    def get_features(self, observation):
        import numpy as np
        return {'z_rl':np.full(2048, observation['state'][0], np.float32),
                'proprio':observation['state'],
                'ref_chunk':np.tile(observation['state']+np.asarray([.0002]*6+[0.],np.float32), (10,1)).astype(np.float32)}


def main(args):
    import numpy as np
    import jax
    import yaml
    configure()
    if args.output.exists():
        raise ValueError('Use a fresh isolated output directory')
    args.output.mkdir(parents=True)
    out = args.output.resolve()
    sources = {'checkpoint':args.checkpoint.resolve(),'norm':args.norm.resolve()}
    hashes = {key:digest(path) for key,path in sources.items()}
    from methods.openpi_rlt.cobot_adapter.process_bootstrap import initialize_process
    initialize_process()
    from rlt_online_rl.config import system_config_from_mapping
    from rlt_online_rl.replay import RLTTransition, ReplayClient
    from rlt_online_rl.inference import ActorClient, EnvDriver, ActorRequest
    from methods.openpi_rlt.plug_v3_yyshadow.right_arm_env import RightArmCobotOnlineEnv
    initial = pickle.loads(args.checkpoint.read_bytes())
    assert int(initial['state']['global_step']) == 5000 and int(initial['state']['actor_version']) == 2500
    run = out/'run'; (run/'checkpoints').mkdir(parents=True)
    # Private synthetic Replay has 600 starting records; change only its arrival
    # counter fixture. All real 5k network/optimizer/RNG values remain identical.
    fixture = dict(initial, progress={'warmup_ready_adds_total':600})
    with (run/'checkpoints/latest.pkl').open('wb') as stream:pickle.dump(fixture,stream)
    with (run/'frozen_actor.pkl').open('wb') as stream:
        pickle.dump({'version':2500,'global_step':5000,'actor_params':initial['state']['actor_params']},stream)
    frozen_hash = digest(run/'frozen_actor.pkl')
    (run/'action_norm_stats.json').write_bytes(args.norm.read_bytes())
    with (run/'replay.pkl').open('wb') as stream:
        for i in range(600):
            p=np.zeros(7,np.float32);p[6]=.004;ref=np.tile(p,(10,1))
            row=RLTTransition(z_rl=np.zeros(2048,np.float32),proprio=p,ref_chunk=ref,action_chunk=ref,
                              rewards=np.zeros(10,np.float32),done=False,next_z_rl=np.zeros(2048,np.float32),
                              next_proprio=p,next_ref_chunk=ref,source=0,source_chunk=np.zeros(10,np.uint8),
                              success=0,intervention_flag=False,
                              collection_phase='warmup',episode_id=10000+i,step_id=0)
            pickle.dump(row.to_journal_record(),stream)
    config=yaml.safe_load((ROOT/'configs/rlt/plug_v3_yyshadow/online_rl.yaml').read_text())
    config['experiment']['rl']['action_norm_stats_path']=str(run/'action_norm_stats.json')
    runtime=config['runtime'];runtime['local_debug_mode']=True;runtime['monitoring']['enable_wandb']=False
    sockets=[]
    for _ in range(2):
        s=socket.socket();s.bind(('127.0.0.1',0));sockets.append(s)
    ports=[s.getsockname()[1] for s in sockets]
    runtime['actor_service'].update(port=ports[0],bind_host='127.0.0.1',snapshot_path=str(run/'frozen_actor.pkl'))
    runtime['replay'].update(port=ports[1],bind_host='127.0.0.1',journal_path=str(run/'replay.pkl'))
    runtime['learner_service'].update(replay_url='http://127.0.0.1:'+str(ports[1]),
                                     checkpoint_dir=str(run/'checkpoints'),actor_snapshot_path=str(run/'candidate_actor.pkl'))
    runtime['env_driver'].update(actor_service_url='http://127.0.0.1:'+str(ports[0]),
                               replay_service_url='http://127.0.0.1:'+str(ports[1]),actor_deterministic=True)
    config_path=out/'online.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    cfg=system_config_from_mapping(config)
    base=[sys.executable,str(ROOT/'methods/openpi_rlt/scripts/online_role.py'),
          '--upstream-root',str(ROOT/'third_party/openpi-rlt'),'--config',str(config_path)]
    for s in sockets:s.close()
    processes=[];logs=[]
    try:
        for role in ['replay_manager','actor_service']:
            log=(out/(role+'.log')).open('w');logs.append(log)
            processes.append(subprocess.Popen(base+['--system.role',role],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        replay=ReplayClient(cfg.env_driver.replay_service_url,timeout_sec=2.)
        actor=ActorClient(cfg.env_driver.actor_service_url,timeout_sec=5.)
        deadline=time.monotonic()+40
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Private service startup timed out')
            if any(p.poll() is not None for p in processes):raise RuntimeError('Private service stopped; inspect logs')
            try:
                if replay.stats()['size']==600 and actor.get_actor_param_version()==2500:break
            except Exception:
                if time.monotonic()>deadline:raise
            time.sleep(.1)
        io=SyntheticIO(out)
        env=RightArmCobotOnlineEnv(io,chunk_exec_horizon=10,control_frequency_hz=20.,max_episode_steps=None,
                                  joint_step_limit=.03,gripper_step_limit=.004,collection_phase='online',sleep=lambda _:None)
        class AcknowledgedReplay:
            def __getattr__(self,name):return getattr(replay,name)
            def add_transitions(self, rows):
                before=replay.stats();replay.add_transitions(rows);after=replay.stats()
                dump(out/'submission_ack.json',{'before':before,'after':after,'submitted':len(rows),
                                                'boundary':'Replay server RPC returned and journal read back below; not power-loss durability or physical execution'})
        driver=EnvDriver(env,SyntheticFeatures(),actor,AcknowledgedReplay(),cfg.rl,cfg.env_driver,
                         metrics_path=str(out/'env_metrics.jsonl'))
        summary=driver.run_episode(90001);driver.close();dump(out/'episode_summary.json',summary)
        assert io.finalized and not io.publications
        journal=load_rows(run/'replay.pkl');new=journal[600:]
        assert len(new)==3 and all(row['action_chunk'].dtype==np.float32 for row in new)
        receipts=[json.loads(line) for path in (out/'trace/replay_inputs').glob('*.jsonl') for line in path.read_text().splitlines()]
        assert len(receipts)==3 and all(row['checks_passed'] for row in receipts)
        from methods.openpi_rlt.cobot_adapter.input_audit import array_receipt
        for row,receipt in zip(new,receipts):
            for key,value in row.items():
                if key=='collection_phase':
                    assert value=='online'
                else:
                    assert array_receipt(value)==receipt['serialized_replay_arrays'][key]
        probe=SyntheticFeatures().get_features(io.records[0]['observation'])
        request=ActorRequest(**probe,deterministic=True,request_id='private-fixed-probe',episode_id=90001,step_id=0)
        before_actor=actor.infer(request)
        log=(out/'learner.log').open('w');logs.append(log)
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--learner',str(config_path),'--output',str(out)],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        if process.wait(timeout=150):raise RuntimeError('Private learner failed; inspect learner.log')
        after_actor=actor.infer(request)
        assert after_actor.actor_param_version==before_actor.actor_param_version==2500
        np.testing.assert_array_equal(before_actor.refined_chunk,after_actor.refined_chunk)
        assert digest(run/'frozen_actor.pkl')==frozen_hash
        trained=pickle.loads((run/'checkpoints/latest.pkl').read_bytes())
        candidate=pickle.loads((run/'candidate_actor.pkl').read_bytes())
        assert int(trained['state']['global_step'])==candidate['global_step']==5015
        assert int(trained['state']['actor_version'])==candidate['version']==2507
        for left,right in zip(jax.tree_util.tree_leaves(trained['state']['actor_params']),jax.tree_util.tree_leaves(candidate['actor_params'])):
            np.testing.assert_array_equal(left,right)
        assert any(not np.array_equal(left,right) for left,right in zip(jax.tree_util.tree_leaves(initial['state']['actor_params']),jax.tree_util.tree_leaves(candidate['actor_params'])))
        # Verify actual normalized training rows against their serialized sources.
        from rlt_online_rl.action_representation import ActionRepresentationAdapter
        adapter=ActionRepresentationAdapter.from_config(cfg.rl)
        expected={}
        for row in new:
            # Online Replay preserves its stored dtypes; a Warmup loader's
            # float32 stacking helper is not the Online sampling contract.
            batch={key:np.asarray(value)[None] for key,value in row.items() if key!='collection_phase'}
            norm=adapter.prepare_training_batch(batch)
            expected[(int(row['episode_id']),int(row['step_id']))]={key:array_receipt(value[0]) for key,value in norm.items()}
        batches=[json.loads(line) for line in (out/'actual_train_inputs.jsonl').read_text().splitlines()]
        counts={};actor_counts={}
        for batch in batches:
            assert batch['finite_metrics']
            for draw in batch['new_experience_draws']:
                key=(draw['episode_id'],draw['step_id'])
                for field,value in draw['arrays'].items():assert value==expected[key][field],(key,field)
                counts[str(key)]=counts.get(str(key),0)+1
                if batch['actor_updated']:actor_counts[str(key)]=actor_counts.get(str(key),0)+1
        assert len(batches)==15
        assert hashes=={key:digest(path) for key,path in sources.items()}
        report = {'status':'passed','source_sha256':hashes,'private_synthetic_start_records':600,
             'new_records':3,'input_receipts_match_journal':True,'actual_training_batches':15,'batch_size':128,
             'new_record_draws':counts,'new_record_actor_draws':actor_counts,'critic_updates':15,'actor_updates':7,
             'final_learner_step':5015,'candidate_actor_version':2507,'served_actor_version':2500,
             'served_action_unchanged':True,'exact_restart':True,'sources_unchanged':True,
             'robot_publishers':0,'stage1_loads':0,'GPU':False,
             'boundary':'Synthetic observations/features/commands and 600 synthetic starting records; real immutable5k Actor/Critic/optimizer/RNG, native Replay RPC and native training. No model improvement or field timing claim.'}
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for log in logs:log.close()
        for port in ports:
            with socket.socket() as s:
                assert s.connect_ex(('127.0.0.1',port))!=0,'Owned private port remained live'
    report['owned_private_processes_stopped'] = all(p.poll() is not None for p in processes)
    report['owned_private_ports_closed'] = True
    dump(out/'report.json', report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--norm',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--learner',type=Path)
    args=parser.parse_args()
    configure()
    if args.learner:learner(args.learner,args.output)
    else:
        if args.checkpoint is None or args.norm is None:parser.error('--checkpoint and --norm are required')
        main(args)
