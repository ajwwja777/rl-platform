#!/usr/bin/env python3
"""Prepare a new isolated teacher5 candidate; no registration, service or publication."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import argparse,sys,json,pickle,hashlib,shutil,dataclasses
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src'),str(ROOT/'third_party/openpi-rlt/packages/openpi-client/src')]
from run_teacher_continuation import read,key,arrays
from methods.openpi_rlt.experiments.supported_hil_runtime import SupportedIndex,load_profile
from methods.openpi_rlt.experiments.supported_sampler import SupportedSampler
from rlt_online_rl.config import RLTOnlineRLConfig,relativize_rl_config_paths
from rlt_online_rl.action_representation import ActionRepresentationAdapter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--study',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 p,o,study=a.assets,a.output,a.study
 if o.exists():raise ValueError('Fresh candidate directory required')
 if 'candidates' not in o.parts:raise ValueError('Isolated candidates path required')
 # Selection is fixed seed42, not the best single run. Prior teacher50/5 gates remain necessary.
 prior=json.loads((p/'outputs/teacher-continuation-20261013/diagnostics.json').read_text())
 assert prior['gates']['passed'] and prior['gates']['all_paired_seeds_better']
 assert prior['gates']['new_dev_mean_improvement_pct']>=.5
 assert json.loads((study/'study.json').read_text())['status']=='complete'
 source=p/'models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010'
 profile=load_profile(source/'profile.json')
 chosen=study/'teacher5_seed42_u846.pkl';payload=pickle.loads(chosen.read_bytes())
 assert payload['retention_weight']==5 and int(payload['state']['global_step'])==8127 and int(payload['state']['actor_version'])==4063
 assert payload['experimental_options']==dict(teacher=5.,clip=None,minq=False)
 conf=dict(payload['rl_config'],action_norm_stats_path=str(source/'action_norm_stats.json'))
 adapter=ActionRepresentationAdapter.from_config(RLTOnlineRLConfig(**conf))
 rows=read(study/'training.pkl');assert len(rows)==3623
 sched=np.load(study/'teacher5_seed42_batches.npz');ident=sched['identities'];splits=sched['splits'];assert (splits[:len(rows)]!='new_dev').all() and (splits[:len(rows)]!='old_dev').all()
 assert [key(r) for r in rows]==[tuple(int(x) for x in k) for k in ident[:len(rows)]]
 o.mkdir(parents=True);journal=o/'replay/replay_journal.pkl';journal.parent.mkdir()
 with journal.open('xb') as f:
  for row in rows[:2777]:pickle.dump(row,f,pickle.HIGHEST_PROTOCOL)
 index=SupportedIndex(journal,.99,profile['retention_episodes'])
 sampler=SupportedSampler(index,adapter,state=dict(base_rng=np.random.default_rng(42).bit_generator.state,actor_rng=np.random.default_rng(100042).bit_generator.state,global_step=7281))
 pos=0
 for ep in dict.fromkeys(r['episode_id'] for r in rows[2777:]):
  chunk=[r for r in rows[2777:] if r['episode_id']==ep]
  with journal.open('ab') as f:
   for row in chunk:pickle.dump(row,f,pickle.HIGHEST_PROTOCOL)
  for _ in chunk:
   sampler.sample_batch(128)
   for role in ['critic','actor']:
    assert sampler.last_receipt[role+'_identities']==ident[sched[role][pos]].tolist(),(role,pos)
   pos+=1
 assert pos==846 and sampler.step==8127
 for name in ['teacher.pkl','action_norm_stats.json']:shutil.copyfile(source/name,o/name)
 cfg=dict(conf,action_norm_stats_path=str(o/'action_norm_stats.json'),warmup_post_collect_updates=8127,grad_updates_per_cycle=1)
 # Loaded profile contains machine-local resolved helper paths; exclude them from portable identity.
 for name in ['teacher','normalization']:profile[name].pop('path',None)
 profile.update(retention_weight=5.,runtime_budget=dict(base_step=8127,updates_per_new_transition=1),source_study=str(study/'study.json'),source_state_sha256=sha(chosen),selection='Predeclared seed42; teacher5 passes repeated DEV engineering gates; no independent autonomous success proof',readiness=dict(offline_selection='passed_teacher5_vs50',native_resume='pending_verification',frozen_robot='pending',autonomous_online_gain='evidence_insufficient'))
 profile['training_source_sha256']={f:sha(ROOT/f) for f in profile['training_source_sha256']}
 (o/'profile.json').write_text(json.dumps(profile,indent=2))
 for folder,name in [('checkpoints','latest.pkl'),('actor_snapshot','actor_snapshot.pkl'),('pending_actor','actor_snapshot.pkl')]:
  f=o/folder/name;f.parent.mkdir();portable=dataclasses.asdict(relativize_rl_config_paths(RLTOnlineRLConfig(**cfg),str(f)))
  obj=dict(rl_config=portable,state=payload['state'],progress=dict(warmup_ready_adds_total=len(rows)),sampling_state=sampler.state(),candidate_profile_sha256=sha(o/'profile.json')) if folder=='checkpoints' else dict(rl_config=portable,actor_params=payload['state']['actor_params'],version=4063,global_step=8127)
  f.write_bytes(pickle.dumps(obj,pickle.HIGHEST_PROTOCOL))
 config=yaml.safe_load((source/'online.yaml').read_text());config['experiment']['rl']=cfg
 def paths(v):
  if isinstance(v,dict):return {k:paths(x) for k,x in v.items()}
  if isinstance(v,list):return [paths(x) for x in v]
  return v.replace(str(source),str(o)) if isinstance(v,str) else v
 config=paths(config);config['runtime']['monitoring']['wandb_run_name']='teacher5-8127-staged'
 (o/'online.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 provenance=dict(selected=str(chosen),selected_sha256=sha(chosen),initial_checkpoint_sha256=payload['initial_checkpoint_sha256'],teacher_sha256=sha(o/'teacher.pkl'),all_846_sampling_receipts_exact=True,sampling_global_step=8127,train_windows=3623,train_episodes=len(set((int(r['collection_phase_id']),int(r['episode_id'])) for r in rows)),dev_draws=0,seed=42,diagnostic_gates_source=str(p/'outputs/teacher-continuation-20261013/diagnostics.json'),prior_diagnostics=prior,production_release=False,field_registration=False,autonomous_gain='not established',runtime_policy='staged; served actor stays fixed until explicit promotion',checkpoint_interval_updates=50)
 (o/'provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps({k:v for k,v in provenance.items() if k!='prior_diagnostics'}))
if __name__=='__main__':main()
