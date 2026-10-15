#!/usr/bin/env python3
"""Select by repeated-DEV action error; report episode uncertainty and Q limits."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
if not (a.run/'completed.json').exists():raise RuntimeError('training incomplete')
rows=json.loads((a.run/'evaluations.json').read_text());candidates=[r for r in rows if r['step']>=1000]
# Task units are kept separate; gripper and joint errors cannot be added directly.
# Lowest DEV joint MAE among checkpoints within 5% of best gripper and p95 joint.
best_grip=min(r['cohorts']['dev']['gripper_mae_m'] for r in candidates)
best_tail=min(r['cohorts']['dev']['joint_p95_rad'] for r in candidates)
eligible=[r for r in candidates if r['cohorts']['dev']['gripper_mae_m']<=1.05*best_grip and r['cohorts']['dev']['joint_p95_rad']<=1.05*best_tail]
if not eligible:raise RuntimeError('joint/gripper/tail tradeoff requires explicit selection review')
selected=min(eligible,key=lambda r:(r['cohorts']['dev']['joint_mae_rad'],r['step']))
rng=np.random.default_rng(41)
def interval(ep):
 values=np.array([e['mae_per_dim'] for e in ep]);ix=rng.integers(len(values),size=(5000,len(values)));boot=values[ix].mean(1)
 return dict(joint_mae_rad_95ci=np.percentile(boot[:,:6].mean(1),[2.5,97.5]).tolist(),gripper_mae_m_95ci=np.percentile(boot[:,6],[2.5,97.5]).tolist())
receipt=dict(selected_step=selected['step'],rule='post-run declared descriptive selection: minimum repeated DEV joint MAE subject to gripper and joint-p95 <= 1.05 times best; no independent-test claim',selected=selected,uncertainty={k:interval(v['episodes']) for k,v in selected['cohorts'].items()},limitations=['All-success human experts; cannot validate failure ranking or autonomous success','DEV 5 Episodes reused for Stage1 and Warmup selection; not TEST','No semantic stage labels; no full-task effectiveness claim','Q real-return bound assumes terminal-only reward1, C10/gamma.99 => [0,.99**9]; TD bootstrap may violate it'])
(a.run/'selection.json').write_text(json.dumps(receipt,indent=2))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for split in ['train','dev']:
 for ax,key,label in zip(axes,['joint_mae_rad','gripper_mae_m','joint_p95_rad'],['6-joint MAE (mrad)','Gripper MAE (mm)','Mean per-episode joint p95 (mrad)']):
  ax.plot([r['step'] for r in rows[1:]],[1000*r['cohorts'][split][key] for r in rows[1:]],marker='.',label=split.upper()+(' (repeated selection)' if split=='dev' else ''))
  limits=[]
  for row in rows[1:]:
   values=np.array([e['p95_per_dim' if key=='joint_p95_rad' else 'mae_per_dim'] for e in row['cohorts'][split]['episodes']])
   values=values[:,6] if key=='gripper_mae_m' else values[:,:6].mean(1)
   indices=rng.integers(len(values),size=(5000,len(values)))
   limits.append(np.percentile(values[indices].mean(1),[2.5,97.5])*1000)
  limits=np.array(limits);ax.fill_between([r['step'] for r in rows[1:]],limits[:,0],limits[:,1],alpha=.12)
  ax.axvline(selected['step'],color='gray',ls=':');ax.set(xlabel='Learner updates',ylabel=label);ax.grid(alpha=.2);ax.legend()
fig.suptitle('Native RLT Warmup | 44 TRAIN / 5 reused DEV Episodes | no independent TEST\n95% complete-Episode bootstrap bands; exploratory selection');fig.tight_layout();fig.savefig(a.run/'warmup_comparison.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for split in ['train','dev']:
 q=np.load(a.run/f'{split}-step-{selected["step"]}.npz');axes[0].hist(q['q_actual'].ravel(),bins=35,alpha=.4,label=split.upper());axes[1].hist((q['q_actor'][:,0]-q['q_ref'][:,0]),bins=35,alpha=.4,label=split.upper())
axes[0].axvline(0,color='red',ls=':');axes[0].axvline(.99**9,color='red',ls=':');axes[0].set_title('Q actual human action (both heads)');axes[1].set_title('Paired Q1(Actor)-Q1(Reference); no causal claim')
for ax in axes:ax.legend()
axes[0].set_ylabel('Q values (2 heads per transition)')
axes[1].set_ylabel('Paired transitions (not independent Episodes)')
fig.tight_layout();fig.savefig(a.run/'warmup_q.png',dpi=160)
print(json.dumps(dict(selected_step=selected['step'],train=selected['cohorts']['train']['joint_mae_rad'],dev=selected['cohorts']['dev']['joint_mae_rad'],grip=selected['cohorts']['dev']['gripper_mae_m'])))
