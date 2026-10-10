from pathlib import Path
import sys,json,pickle,hashlib,shutil,dataclasses
import numpy as np,yaml
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'third_party/openpi-rlt/rlt_online_rl/src')]
from methods.openpi_rlt.experiments.supported_runtime import SupportedIndex
from methods.openpi_rlt.experiments.supported_sampler import SupportedSampler,FIELDS
from rlt_online_rl.config import RLTOnlineRLConfig,relativize_rl_config_paths
from rlt_online_rl.action_representation import ActionRepresentationAdapter
R=Path('/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform');O=R/'outputs/online-credit-repair-20261010';A=R/'models/rlt/plug_v3_yyshadow/history/candidates/supported_online_20261006_v4';P=R/'models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010'
assert not P.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 rows=[]
 with p.open('rb') as f:
  while True:
   try:rows.append(pickle.load(f))
   except EOFError:break
 return rows
study=json.loads((O/'followup/comparison.json').read_text());gates=[]
for seed in [41,42,43]:
 base=study['evaluations'][f'baseline_seed{seed}'][-1];candidate=study['evaluations'][f'hil8_clip_seed{seed}'][-1]
 assert candidate['groups']['new_HIL_TRAIN']['BC6_mse']<base['groups']['new_HIL_TRAIN']['BC6_mse']
 for group in ['autonomous','assisted','failure']:
  q=candidate['repeated_DEV']['groups'][group];b=base['repeated_DEV']['groups'][group]
  assert q['bc6_abs_p95_mrad']-b['bc6_abs_p95_mrad']<=.05
  if group!='failure':assert q['q1_behavior_mse']<=b['q1_behavior_mse']*1.05
 gates.append(dict(seed=seed,passed=True))
rows=read(A/'replay/replay_journal.pkl')
D=R/'outputs/online-rollout-strategy-20261007';rows+=read(D/'new204-delta.pkl')+read(D/'new77-delta.pkl');assert len(rows)==2777
baseprofile=json.loads((A/'profile.json').read_text());initial=pickle.loads((A/'checkpoints/latest.pkl').read_bytes())
cfg=dict(initial['rl_config']);cfg['action_norm_stats_path']=str(A/'action_norm_stats.json');adapter=ActionRepresentationAdapter.from_config(RLTOnlineRLConfig(**cfg))
# Reconstruct exactly the sampling states from the verified arrival replay; no RNG guess.
tmp=O/'sampler-verification';tmp.mkdir();journal=tmp/'replay_journal.pkl'
index=SupportedIndex(journal,.99,baseprofile['retention_episodes']);sampler=SupportedSampler(index,adapter,state=dict(base_rng=np.random.default_rng(42).bit_generator.state,actor_rng=np.random.default_rng(1042).bit_generator.state,global_step=7000))
ci=np.load(O/'followup/critic_sample_indices.npz')['hil8_clip_seed42'];ai=np.load(O/'followup/sample_indices.npz')['hil8_clip_seed42']
arrivals=json.loads((O/'followup/arrival_schedule.json').read_text());arrays={k:np.stack([r[k]for r in rows])for k in FIELDS}
for update in range(1,282):
 for entry in arrivals:
  if entry['first_update']==update:
   with journal.open('wb')as f:
    for row in rows[:entry['available']]:pickle.dump(row,f,pickle.HIGHEST_PROTOCOL)
 batch=sampler.sample_batch(128);actual=sampler.last_receipt
 for tag,indices in [('critic',ci[update-1]),('actor',ai[update-1])]:
  expected=[[int(arrays[k][i])for k in ['collection_phase_id','episode_id','step_id']]for i in indices]
  assert actual[tag+'_identities']==expected,(tag,update)
 if update in [1,225,281]:
  expected=adapter.prepare_training_batch(index.attach({k:v[ai[update-1]]for k,v in arrays.items()}))
  for k in expected:np.testing.assert_array_equal(batch['actor__'+k],expected[k])
P.mkdir()
for name in ['action_norm_stats.json','teacher.pkl']:shutil.copyfile(A/name,P/name)
(P/'replay').mkdir();shutil.copyfile(journal,P/'replay/replay_journal.pkl')
selected=O/'followup/hil8_clip_seed42/step_7281.pkl';payload=pickle.loads(selected.read_bytes());state=payload['state'];assert state['global_step']==7281 and state['actor_version']==3640
cfg=dict(payload['rl_config']);cfg.update(action_norm_stats_path=str(P/'action_norm_stats.json'),warmup_post_collect_updates=7281,grad_updates_per_cycle=1)
profile=dict(baseprofile,schema='held-gripper-hil8-clip-v2',target_policy='clip',actor_hil_slots=8,runtime_budget=dict(base_step=7281,updates_per_new_transition=1),source_study=str(O/'followup/comparison.json'),source_state_sha256=sha(selected),selection='Seed42 predeclared, 3-seed paired gates; TRAIN/reused DEV only; no independent TEST',readiness=dict(offline_sampling_gates='passed',native_resume='pending',frozen_robot='pending',autonomous_online_gain='evidence_insufficient'))
source=list(baseprofile['training_source_sha256'])+['methods/openpi_rlt/experiments/guarded_credit.py','methods/openpi_rlt/experiments/guarded_retained_actor.py','methods/openpi_rlt/experiments/separate_actor_batch.py','methods/openpi_rlt/experiments/supported_sampler.py','methods/openpi_rlt/experiments/supported_hil_runtime.py']
profile['training_source_sha256']={p:sha(W/p) for p in source};(P/'profile.json').write_text(json.dumps(profile,indent=2))
for folder,name in [('checkpoints','latest.pkl'),('actor_snapshot','actor_snapshot.pkl')]:
 p=P/folder/name;p.parent.mkdir();portable=dataclasses.asdict(relativize_rl_config_paths(RLTOnlineRLConfig(**cfg),str(p)))
 data=dict(rl_config=portable,state=state,progress=dict(warmup_ready_adds_total=len(rows)),candidate_profile_sha256=sha(P/'profile.json'),sampling_state=sampler.state()) if folder=='checkpoints' else dict(rl_config=portable,actor_params=state['actor_params'],version=int(state['actor_version']),global_step=int(state['global_step']))
 p.write_bytes(pickle.dumps(data,pickle.HIGHEST_PROTOCOL))
config=yaml.safe_load((A/'online.yaml').read_text());config['experiment']['rl']=cfg
runtime=config['runtime'];runtime['replay']['journal_path']=str(P/'replay/replay_journal.pkl');runtime['actor_service']['snapshot_path']=str(P/'actor_snapshot/actor_snapshot.pkl');runtime['learner_service'].update(checkpoint_dir=str(P/'checkpoints'),actor_snapshot_path=str(P/'pending_actor/actor_snapshot.pkl'));runtime['monitoring'].update(wandb_dir=str(P/'metrics/wandb'),wandb_run_name='hil8-clip-staged')
(P/'online.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
report=dict(package=str(P),selected_state=str(selected),selected_sha=sha(selected),gates=gates,sampling_identity_all_281_batches=True,actor_normalization_exact=True,sampling_checkpoint_step=sampler.step,source_protected_sha={str(p):sha(p) for p in [A/'checkpoints/latest.pkl',A/'replay/replay_journal.pkl',A/'teacher.pkl',A/'action_norm_stats.json']},replay_rows=2777,production_release=False,autonomous_gain='not established',failed_terminal_Q='still high; not certified')
(P/'provenance.json').write_text(json.dumps(report,indent=2));(O/'package-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
