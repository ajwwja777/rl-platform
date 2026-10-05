#!/usr/bin/env python3
"""Exercise the real project launcher on private assets and CPU dummy I/O.

Starts only isolated A6000 loopback services. No ROS/Stage1/robot connection,
no production Replay writes. Synthetic data verify plumbing, not task ability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--norm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-profile', choices=['mc_30'])
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh isolated output directory')
    import numpy as np
    import yaml
    from rlt_online_rl.config import system_config_from_mapping
    from rlt_online_rl.replay import ReplayManager
    from rlt_online_rl.trainer import LearnerService
    import jax
    args.output.mkdir(parents=True)
    out = args.output.resolve()
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {name: getattr(args,name).resolve() for name in ['checkpoint','replay','norm']}
    hashes = {name:digest(path) for name,path in sources.items()}
    run = out/'candidates/run' if args.experiment_profile else out/'run'
    (run/'checkpoints').mkdir(parents=True)
    shutil.copyfile(args.checkpoint, run/'checkpoints/latest.pkl')
    shutil.copyfile(args.norm, run/'action_norm_stats.json')
    shutil.copyfile(args.replay, run/'replay.pkl')
    config = yaml.safe_load((ROOT/'configs/rlt/plug_v3_yyshadow/online_rl.yaml').read_text())
    config['experiment']['rl']['action_norm_stats_path'] = str(run/'action_norm_stats.json')
    runtime = config['runtime']; runtime['local_debug_mode'] = True
    # Reserve then release three distinct loopback ports for the private service run.
    sockets=[]
    for _ in range(3):
        s=socket.socket();s.bind(('127.0.0.1',0));sockets.append(s)
    ports=[s.getsockname()[1] for s in sockets]
    runtime['actor_service'].update(port=ports[0],bind_host='127.0.0.1',snapshot_path=str(run/'actor_snapshot.pkl'))
    runtime['replay'].update(port=ports[1],bind_host='127.0.0.1',journal_path=str(run/'replay.pkl'))
    runtime['learner_service'].update(replay_url='http://127.0.0.1:'+str(ports[1]),
        checkpoint_dir=str(run/'checkpoints'),actor_snapshot_path=str(run/'actor_snapshot.pkl'))
    runtime['env_driver'].update(actor_service_url='http://127.0.0.1:'+str(ports[0]),
        replay_service_url='http://127.0.0.1:'+str(ports[1]),actor_deterministic=True)
    # DummyFeatureProvider is selected by upstream local_debug_mode. No Stage1 port.
    runtime['monitoring']['enable_wandb']=False
    cfg_path=out/'online.yaml';cfg_path.write_text(yaml.safe_dump(config,sort_keys=False))
    initial=pickle.loads(args.checkpoint.read_bytes())['state']
    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',COBOT_RLT_REPLAY_ACTION_PRECISION='float32',
        RLT_OUTPUT_DIR=str(run),XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for key in ['COBOT_RLT_EXPERIMENT_PROFILE','COBOT_RLT_EXECUTION_OPTIONS','COBOT_EXECUTION_OPTIONS','RLT_DISABLE_LEARNER']:
        env.pop(key,None)
    if args.experiment_profile:
        env['COBOT_RLT_EXPERIMENT_PROFILE']=args.experiment_profile
    cmd=[sys.executable,str(ROOT/'methods/openpi_rlt/scripts/online_role.py'),'--upstream-root',str(ROOT/'third_party/openpi-rlt'),
        '--config',str(cfg_path),'--num-episodes','1']
    for s in sockets:s.close()
    started=time.time()
    with (out/'supervisor.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise RuntimeError('Private spawned runtime timed out; owned process group stopped')
    if code: raise RuntimeError('Private runtime failed; inspect '+str(out/'supervisor.log'))
    system=system_config_from_mapping(config)
    replay=ReplayManager(system.replay.capacity,journal_path=str(run/'replay.pkl'),seed=42)
    final=pickle.loads((run/'checkpoints/latest.pkl').read_bytes())
    snapshot=pickle.loads((run/'actor_snapshot.pkl').read_bytes())
    rows=[]
    with (run/'replay.pkl').open('rb') as stream:
        while True:
            try:rows.append(pickle.load(stream))
            except EOFError:break
    before=0
    with args.replay.open('rb') as stream:
        while True:
            try:pickle.load(stream);before+=1
            except EOFError:break
    added=len(rows)-before
    expected=int(initial['global_step'])+added*system.rl.grad_updates_per_cycle
    # A bounded operator stop may leave earned budget pending. Restore must
    # preserve it exactly, rather than discard it or repeat consumed updates.
    step=int(final['state']['global_step'])
    actor_version=int(final['state']['actor_version'])
    assert added>0 and int(initial['global_step'])<=step<=expected,(added,step,expected)
    assert actor_version==int(initial['actor_version'])+(step//2-int(initial['global_step'])//2)
    assert snapshot['global_step']==step and snapshot['version']==actor_version
    for x,y in zip(jax.tree_util.tree_leaves(snapshot['actor_params']),jax.tree_util.tree_leaves(final['state']['actor_params'])):
        np.testing.assert_array_equal(x,y)
    assert all(r['action_chunk'].dtype==np.float32 for r in rows[before:])
    audit=run/'metrics/batch_composition.jsonl'
    batches=[json.loads(line) for line in audit.read_text().splitlines() if line.strip()] if audit.exists() else []
    assert len(batches)==step-int(initial['global_step'])
    if args.experiment_profile:
        learner_metrics=[json.loads(line) for line in (run/'metrics/learner_metrics.jsonl').read_text().splitlines() if line.strip()]
        assert learner_metrics and all(abs(row['mc_effective_weight']-.3)<1e-5 for row in learner_metrics)
        from methods.openpi_rlt.experiments.runtime import install_learner, CreditIndex
        install_learner({'mc_weight':.3},str(run/'replay.pkl'),system.rl.gamma)
        index=CreditIndex(run/'replay.pkl',system.rl.gamma)
        original_replay=replay
        class CreditSource:
            def stats(self):return original_replay.stats()
            def sample_batch(self,*a,**kw):return index.attach(original_replay.sample_batch(*a,**kw))
        replay=CreditSource()
    restored=LearnerService(system.rl,system.learner_service,replay,metrics_path=str(out/'restart/metrics.jsonl'))
    for key in initial:
        for x,y in zip(jax.tree_util.tree_leaves(getattr(restored.state,key)),jax.tree_util.tree_leaves(final['state'][key])):
            np.testing.assert_array_equal(x,y)
    pending=expected-step
    for _ in range(pending):
        metrics=restored.train_once()
        assert metrics is not None and all(np.isfinite(v) for v in metrics.values())
    assert restored.train_once() is None,'Restore repeated already consumed update budget'
    assert int(restored.state.global_step)==expected
    restored.flush_artifacts()
    restarted=LearnerService(system.rl,system.learner_service,replay,metrics_path=str(out/'restart_second/metrics.jsonl'))
    assert restarted.train_once() is None
    for x,y in zip(jax.tree_util.tree_leaves(restored.state),jax.tree_util.tree_leaves(restarted.state)):
        np.testing.assert_array_equal(x,y)
    assert hashes=={name:digest(path) for name,path in sources.items()}
    for port in ports[:2]:
        with socket.socket() as s: assert s.connect_ex(('127.0.0.1',port))!=0,'Owned service port remained live'
    result=dict(status='passed',command=cmd,config=str(cfg_path),config_sha256=digest(cfg_path),
        elapsed_sec=time.time()-started,initial_step=int(initial['global_step']),final_step=step,actor_version=actor_version,
        synthetic_added_transitions=added,spawned_updates=len(batches),pending_updates_at_stop=pending,
        resumed_updates=pending,total_updates=expected-int(initial['global_step']),final_resumed_step=expected,
        spawned_batch_audit_verified=bool(batches),published_step=snapshot['global_step'],
        published_actor=snapshot['version'],full_restart_state_exact=True,source_sha256=hashes,sources_unchanged=True,
        spawned_precision_verified=True,owned_ports_closed=True,
        experiment_profile=args.experiment_profile,spawned_mc_weight_verified=.3 if args.experiment_profile else None,
        robot_publishers=0,stage1_loads=0,field_operations=0,
        boundary='Actual fixed network/checkpoint and project spawn entry; dummy features/dynamics only. Not timing, HIL, insertion success or autonomous learning acceptance.')
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
