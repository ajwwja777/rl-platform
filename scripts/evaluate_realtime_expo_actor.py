#!/usr/bin/env python3
"""Evaluate exact deployed float32 single-input Actor, distinct from Replay float16 ref."""
import argparse,json,pickle,sys,shutil
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('project',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--run-name',default='warmup-native-10k');a=ap.parse_args()
root=Path(__file__).resolve().parents[1];up=a.project/'third_party/openpi-rlt';sys.path[:0]=[str(root),str(up/'rlt_online_rl/src'),str(up/'packages/openpi-client/src')]
import numpy as np,jax
from rlt_online_rl.inference import RLTPolicyInferenceWrapper
from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.config import RLTOnlineRLConfig
run=a.output/a.run_name;original=run/'evaluations-replay-input.json'
if not original.exists():shutil.copyfile(run/'evaluations.json',original)
evaluations=json.loads(original.read_text());cfg=json.loads((run/'config.json').read_text())['rl'];config=RLTOnlineRLConfig(**cfg)
wrapper=RLTPolicyInferenceWrapper(config);adapter=ActionRepresentationAdapter.from_config(config)
rows={}
for split in ['train','dev']:
 rows[split]=[]
 with (a.output/f'warmup-seed/{split}_expert_journal.pkl').open('rb') as f:
  while True:
   try:rows[split].append(pickle.load(f))
   except EOFError:break
for item in evaluations:
 step=item['step']
 if step==0:continue
 payload=pickle.loads((run/f'actor_snapshot/step_{step}.pkl').read_bytes());params=jax.tree.map(jax.numpy.asarray,payload['actor_params'])
 for split,records in rows.items():
  actions=[]
  for row in records:
   ref=adapter.normalize_ref_chunk(row['ref_chunk'],row['proprio'])
   pred=wrapper.infer(params,np.asarray(row['z_rl'],np.float32),row['proprio'],ref,deterministic=True)
   actions.append(adapter.denormalize_to_abs_chunk(pred,row['proprio']))
  actions=np.stack(actions);old=np.load(run/f'{split}-step-{step}.npz');quant_error=float(np.max(np.abs(old['actions']-actions)))
  uuids=np.array([r['episode_uuid'] for r in records]);target=np.stack([r['action_chunk'] for r in records]);err=np.abs(actions-target)
  episode=item['cohorts'][split]['episodes']
  for e in episode:
   selected=err[uuids==e['uuid']];e['mae_per_dim']=selected.mean((0,1)).tolist();e['p95_per_dim']=np.percentile(selected.reshape(-1,7),95,axis=0).tolist()
  means=np.mean([e['mae_per_dim'] for e in episode],0);tails=np.mean([e['p95_per_dim'] for e in episode],0)
  item['cohorts'][split].update(joint_mae_rad=float(means[:6].mean()),gripper_mae_m=float(means[6]),joint_p95_rad=float(tails[:6].mean()),replay_vs_deployed_max_abs=quant_error,action_evaluation='native single-input float32 deployed wrapper; Q fields retain replay-input diagnostic')
  np.save(run/f'{split}-deployed-actions-{step}.npy',actions)
 print(step,{k:v['joint_mae_rad'] for k,v in item['cohorts'].items()},flush=True)
(run/'evaluations.json').write_text(json.dumps(evaluations,indent=2))
