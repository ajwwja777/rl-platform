#!/usr/bin/env python3
"""Build an isolated resumable candidate; never publish or touch production assets."""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]
import numpy as np
import yaml


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_rows(p):
    rows=[]
    with Path(p).open('rb')as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    return rows



def canonicalize_training_row(row):
    """Match the offline study's explicit phase and expert-ID contract."""
    row = dict(row)
    online = bool(row.pop('phase_online'))
    row['collection_phase_id'] = 2 if online else 1
    row['collection_phase'] = 'online' if online else 'warmup'
    episode = int(row['episode_id'])
    if not online and episode >= 100000:
        row['episode_id'] = -(episode - 100000 + 1)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();o=a.output.resolve()
    if o.exists()or'candidates'not in o.parts:raise ValueError('Fresh isolated candidates directory required')
    study=json.loads(a.study.read_text())
    if study['status']!='completed':raise ValueError('Complete matched study first')
    run=next(r for r in study['runs']if r['seed']==a.seed and r['variant']==a.variant)
    for file,digest in study['source_sha256'].items():
        if sha(file)!=digest:raise ValueError('Study asset changed: '+file)
    source=Path(run['checkpoint']);initial=a.assets_root/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
    if sha(source)!=run['checkpoint_sha256']:raise ValueError('Selected state changed')
    payload=pickle.loads(source.read_bytes());start=pickle.loads(initial.read_bytes())
    state=payload['state'];exp=payload['experiment'];cfg=dict(payload['rl_config'])
    # Runtime scheduling starts at this complete inherited state, not old20k
    # warmup metadata. The loss/optimizer/network contract remains unchanged.
    cfg.update(warmup_post_collect_updates=int(state['global_step']),grad_updates_per_cycle=1,
               warmup_bc_weight=cfg['online_bc_weight'],warmup_q_weight=exp['actor_q_weight'])
    journal=initial.parent.parent/'replay/replay_journal.pkl';old=a.assets_root/'outputs/model-repair-20261005'
    raw=dict(np.load(old/'precision_inputs.npz'));rows=read_rows(journal)
    lookup={(int(raw['episode_id'][i]),int(raw['step_id'][i])):i for i in range(len(raw['episode_id']))if not raw['phase_online'][i]}
    full=[]
    for row in rows:
        row=dict(row);match=lookup.get((int(row['episode_id']),int(row['step_id'])))
        if match is not None:row['action_chunk']=raw['original_action'][match].copy()
        row.update(phase_online=False);full.append(row)
    for i in np.flatnonzero(raw['phase_online']):
        row={k:raw[k][i]for k in raw if k not in ['original_action','raw_verified','replay_index']}
        row['action_chunk']=raw['original_action'][i].copy();row['collection_phase']='online';full.append(row)
    trained={tuple(k)for k in study['train_episodes']};dev={tuple(k)for k in study['dev_episodes']}
    selected=[canonicalize_training_row(r) for r in full if(bool(r['phase_online']),int(r['episode_id']))in trained]
    assert len(selected)==study['training_records'] and not trained&dev
    retained=[];groups={}
    for row in selected:
        key=(int(row['collection_phase_id']),int(row['episode_id']));groups.setdefault(key,[]).append(row)
    for key,rs in groups.items():
        expert=all(int(r['episode_id'])<0 or int(r['episode_id'])>=100000 for r in rs)
        human=any(np.isin(r['source_chunk'],[2,3]).any()for r in rs)
        if key[0]==1 and(expert or(not human and any(r['success']for r in rs))):
            retained.append(list(key))
    # Verify the deployable Replay/masks against the actual numeric training
    # contract, rather than relying on metadata or successful restoration.
    identity=dict(np.load(a.study.parent/'data_identity.npz'))
    expected={}
    for i in range(len(identity['episode_id'])):
        key=(bool(identity['phase_online'][i]),int(identity['episode_id'][i]))
        if key not in trained:continue
        eid=-(key[1]-100000+1)if not key[0]and key[1]>=100000 else key[1]
        expected[(2 if key[0]else 1,eid,int(identity['step_id'][i]))]=i
    seen=set()
    for row in selected:
        key=tuple(int(row[k])for k in ['collection_phase_id','episode_id','step_id'])
        if key in seen:raise ValueError('Duplicate packaged training identity')
        seen.add(key);i=expected[key]
        for field,values in [('action_chunk',identity['action']),('ref_chunk',identity['reference']),('source_chunk',identity['source_chunk'])]:
            np.testing.assert_array_equal(row[field],values[i])
    assert seen==set(expected)
    retained_set={tuple(k)for k in retained}
    assert sum(tuple(int(r[k])for k in ['collection_phase_id','episode_id'])in retained_set for r in selected)==study['retention_rows']
    o.mkdir(parents=True)
    norm=o/'action_norm_stats.json';shutil.copyfile(cfg['action_norm_stats_path'],norm)
    teacher=o/'teacher.pkl'
    with teacher.open('wb')as f:pickle.dump(dict(actor_params=start['state']['actor_params'],source_checkpoint_sha256=sha(initial)),f,pickle.HIGHEST_PROTOCOL)
    jpath=o/'replay/replay_journal.pkl';jpath.parent.mkdir()
    with jpath.open('wb')as f:
        for row in selected:
            row=dict(row)
            pickle.dump(row,f,pickle.HIGHEST_PROTOCOL)
    critical=[k for k in cfg if not k.startswith('warmup_')and k not in ['freeze_after_warmup','grad_updates_per_cycle','action_norm_stats_path']]
    profile=dict(schema='held-gripper-retention-v1',held_gripper=bool(exp.get('held_gripper',False)),mc_weight=exp['mc_weight'],retention_weight=exp['retention_weight'],actor_q_weight=exp['actor_q_weight'],training_config={k:cfg[k]for k in critical},retention_episodes=retained,
      teacher=dict(file=teacher.name,sha256=sha(teacher)),normalization=dict(file=norm.name,sha256=sha(norm)),publication_policy='staged',episode_id_floor=10000,
      runtime_budget=dict(base_step=int(state['global_step']),updates_per_new_transition=1),runtime_environment=dict(COBOT_RLT_EXECUTION_PROFILE='async_rtc50',COBOT_RLT_HOLD_RIGHT_GRIPPER='1'if exp.get('held_gripper',False)else'0',COBOT_RLT_REPLAY_ACTION_PRECISION='float32',COBOT_RLT_HIL_SAMPLING='logical20',COBOT_RLT_HIL_TARGET='coordinator_command',COBOT_RLT_RAW_OBSERVATION_CONTRACT='trace',COBOT_RLT_INPUT_AUDIT='strict',COBOT_RLT_DIAGNOSTIC_METRICS='0'),
      source_study=str(a.study.resolve()),source_state_sha256=sha(source),source_initial_sha256=sha(initial),selection='Seed42 declared before candidate inspection; all three seeds evaluated. Reused DEV, no independent TEST.',readiness=dict(native_export_and_resume='pending',frozen_robot='pending',autonomous_online_gain='evidence_insufficient'))
    source_files=['methods/openpi_rlt/experiments/held_gripper.py','methods/openpi_rlt/experiments/retained_actor.py','methods/openpi_rlt/experiments/credit.py','methods/openpi_rlt/experiments/target_attribution.py','methods/openpi_rlt/experiments/supported_runtime.py','third_party/openpi-rlt/rlt_online_rl/src/rlt_online_rl/trainer.py','third_party/openpi-rlt/rlt_online_rl/src/rlt_online_rl/networks.py']
    profile['training_source_sha256']={f:sha(ROOT/f)for f in source_files}
    path=o/'profile.json';path.write_text(json.dumps(profile,indent=2))
    cfg['action_norm_stats_path']=str(norm)
    from rlt_online_rl.config import RLTOnlineRLConfig,relativize_rl_config_paths
    import dataclasses
    for folder,file in [('checkpoints','latest.pkl'),('actor_snapshot','actor_snapshot.pkl')]:
        target=o/folder/file;target.parent.mkdir()
        portable=dataclasses.asdict(relativize_rl_config_paths(RLTOnlineRLConfig(**cfg),str(target)))
        out=dict(rl_config=portable,state=state,progress=dict(warmup_ready_adds_total=len(selected)),candidate_profile_sha256=sha(path))if folder=='checkpoints'else dict(rl_config=portable,actor_params=state['actor_params'],version=int(state['actor_version']),global_step=int(state['global_step']))
        with target.open('wb')as f:pickle.dump(out,f,pickle.HIGHEST_PROTOCOL)
    config=yaml.safe_load((a.assets_root/'configs/rlt/plug_v3_yyshadow/online_rl.yaml').read_text());config['experiment']['rl']=cfg
    runtime=config['runtime'];runtime['monitoring'].update(enable_wandb=False,wandb_dir=str(o/'metrics/wandb'),wandb_run_name='supported-online-staged')
    runtime['replay'].update(journal_path=str(jpath),sample_strategy='stratified',recent_episode_window=20,recent_online_ratio=.4,warmup_demo_ratio=.3,human_intervention_ratio=.2,port=9142)
    runtime['actor_service'].update(snapshot_path=str(o/'actor_snapshot/actor_snapshot.pkl'),port=9141)
    runtime['learner_service'].update(checkpoint_dir=str(o/'checkpoints'),actor_snapshot_path=str(o/'pending_actor/actor_snapshot.pkl'),replay_url='http://127.0.0.1:9142',push_actor_interval_steps=50,checkpoint_interval_steps=50)
    runtime['env_driver'].update(actor_service_url='http://127.0.0.1:9141',replay_service_url='http://127.0.0.1:9142',actor_deterministic=False)
    (o/'online.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (o/'provenance.json').write_text(json.dumps(dict(research_selected_state=str(source),selected_seed=a.seed,selected_variant=a.variant,state_step=int(state['global_step']),actor_version=int(state['actor_version']),replay_rows=len(selected),replay_episodes=len(groups),source_journal_sha256=sha(journal),candidate_journal_sha256=sha(jpath),initial_warmup_seen_development=True,independent_test=None,actual_execution_profile='operator50Hz RTC+filter; new physical acceptance pending',field_deployed=False,production_unchanged=True),indent=2))
    print(str(o))

if __name__=='__main__':main()
