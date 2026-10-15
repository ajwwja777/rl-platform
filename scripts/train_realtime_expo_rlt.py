#!/usr/bin/env python3
"""Validate/stats/train new left-arm cohort; scoped data-loader hook, fixed RLT trainer."""
import argparse,dataclasses,os,sys,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','train']);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--fsdp-devices',type=int,default=2);ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
project=Path(__file__).resolve().parents[1];upstream=a.assets/'third_party/openpi-rlt'
if a.command=='train':
 import subprocess
 selected=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
 if len(selected)!=a.fsdp_devices or not all(x.isdigit() for x in selected):raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES and matching FSDP count required')
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 used={x.split(',')[0].strip():int(x.split(',')[1]) for x in raw.strip().splitlines()}
 if any(used.get(x,10**9)>1024 for x in selected):raise RuntimeError('Selected GPU is occupied; never interrupt another task')
for f in [project,upstream/'src',upstream/'scripts',upstream/'packages/openpi-client/src']:sys.path.insert(0,str(f))
if a.command=='prepare':os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENPI_DATA_HOME']=str(a.output/'cache/openpi');os.environ['HF_HOME']=str(a.assets/'cache/huggingface');os.environ['WANDB_MODE']='disabled'
# Load C libraries in the known working order before model/config imports.
import datasets,torch
import openpi.training.data_loader as dl
import numpy as np
from methods.openpi_rlt.realtime_expo.dataset import ExpertDataset
from methods.openpi_rlt.realtime_expo.configuration import build_config,WristInputs
from openpi import transforms
from openpi.shared import normalize
from methods.openpi_rlt.plug_v3_yyshadow.train_stage1 import verify_upstream
from methods.openpi_rlt.stage1_entry import run_upstream_train
verify_upstream(upstream)
a.output.mkdir(parents=True,exist_ok=True)
c=build_config(a.output,a.base,a.batch_size,a.fsdp_devices,a.steps)
train=ExpertDataset(a.data,'train');dev=ExpertDataset(a.data,'dev')
assert not {r['uuid'] for r in train.identities}&{r['uuid'] for r in dev.identities}
original=dl.create_torch_dataset
def loader(data_config,horizon,model):
 if data_config.repo_id=='realtime_expo_left_train':
  assert horizon==train.horizon
  return train
 return original(data_config,horizon,model)
dl.create_torch_dataset=loader
if a.command=='prepare':
 stats={'state':normalize.RunningStats(),'actions':normalize.RunningStats()};delta=transforms.DeltaActions(transforms.make_bool_mask(6,-1));absolute=transforms.AbsoluteActions(transforms.make_bool_mask(6,-1));maxerr=0.
 for i in range(len(train)):
  raw=train[i];x=delta(WristInputs()(raw));restored=absolute({**x,'actions':x['actions'].copy()})
  maxerr=max(maxerr,float(np.max(np.abs(restored['actions']-raw['actions']))))
  for key in stats:stats[key].update(x[key][None])
 assert maxerr<1e-6
 normalize.save(c.assets_dirs/'realtime_expo_left_train',{k:v.get_statistics() for k,v in stats.items()})
 config=c.data.create(c.assets_dirs,c.model);transformed=dl.transform_dataset(train,config)
 probes=[0,len(train)//2,len(train)-1];shapes=[]
 for i in probes:
  x=transformed[i];assert x['actions'].shape==(50,32);assert np.isfinite(x['actions']).all();assert not x['image_mask']['base_0_rgb'];shapes.append({k:list(v.shape) for k,v in x['image'].items()})
 batch=next(iter(dl.create_data_loader(c,num_batches=1,shuffle=False)))
 observation,actions=batch
 assert actions.shape==(c.batch_size,50,32) and np.isfinite(np.asarray(actions)).all()
 assert not np.asarray(observation.image_masks['base_0_rgb']).any()
 result=dict(actual_batch_action_shape=list(actions.shape),actual_batch_image_shapes={k:list(v.shape) for k,v in observation.images.items()},status='cpu_input_verified',train_episodes=train.identities,dev_episodes=dev.identities,train_frames=len(train),dev_frames=len(dev),action_roundtrip_max_abs=maxerr,transformed_shapes=shapes,actual_loader_train_only=True,fps=10,horizon=50,active_arm='left',gripper='absolute_variable',model_params_precision='fixed upstream, no precision overrides',batch_size=c.batch_size,fsdp_devices=c.fsdp_devices,steps=c.num_train_steps,base=str(a.base),norm=str(c.assets_dirs/'realtime_expo_left_train/norm_stats.json'),gpu_train_started=False)
 (a.output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
else:
 if not (a.output/'preflight.json').exists():raise RuntimeError('CPU prepare required')
 checked=json.loads((a.output/'preflight.json').read_text())
 assert checked['train_episodes']==train.identities and checked['dev_episodes']==dev.identities and checked['base']==str(a.base)
 (a.output/'resolved_config.txt').write_text(repr(c)+'\n')
 import train_rlt
 run_upstream_train(train_rlt,dataclasses.replace(c,resume=a.resume),a.output/'jax-cache')
