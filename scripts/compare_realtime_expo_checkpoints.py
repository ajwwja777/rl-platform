#!/usr/bin/env python3
"""Compare complete checkpoint evaluations; figures never mix TRAIN and DEV."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from methods.openpi_rlt.realtime_expo.selection import select_checkpoint
ap=argparse.ArgumentParser();ap.add_argument('--reports',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
reports=[json.loads(p.read_text()) for p in a.reports]
result=select_checkpoint(reports)
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'selection.json').exists():raise FileExistsError('completed comparison exists')
reports.sort(key=lambda r:int(Path(r['checkpoint']).name))
fig,axes=plt.subplots(1,2,figsize=(11,4))
tags=[int(Path(r['checkpoint']).name) for r in reports]
for split,style in [('train','--o'),('dev','-s')]:
    for ax,key,scale in [(axes[0],'c10_joint_mae_rad',1000),(axes[1],'c10_gripper_mae_m',1000)]:
        means=[r['cohorts'][split]['summary']['episode_equal_mean'][key]*scale for r in reports]
        ax.plot(tags,means,style,label=split.upper())
        ci=np.asarray([r['cohorts'][split]['summary']['episode_bootstrap_95'][key] for r in reports])*scale
        ax.fill_between(tags,ci[:,0],ci[:,1],alpha=.12)
axes[0].set_ylabel('Joint MAE (mrad)');axes[1].set_ylabel('Gripper MAE (mm)')
for ax in axes:ax.set_xlabel('Checkpoint tag');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Complete-Episode equal means / bootstrap 95%; reused DEV; no TEST')
fig.tight_layout();fig.savefig(a.output/'checkpoint_comparison.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(10,7))
for row,split in enumerate(('train','dev')):
    values=np.asarray([r['cohorts'][split]['summary']['episode_equal_mean']['c10_mae_per_dim'] for r in reports])*1000
    # Joints mrad and gripper mm explicitly identified, never averaged together.
    im=axes[row].imshow(values,aspect='auto')
    axes[row].set_xticks(range(7),['J1','J2','J3','J4','J5','J6','Gripper'])
    axes[row].set_yticks(range(len(tags)),tags);axes[row].set_title(split.upper()+' — joints mrad / gripper mm')
    fig.colorbar(im,ax=axes[row])
fig.tight_layout();fig.savefig(a.output/'per_dimension_errors.png',dpi=170);plt.close(fig)
result['reports']=list(map(str,a.reports))
(a.output/'selection.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
