#!/usr/bin/env python3
"""Full-Episode checkpoint comparison with physical errors and preserved predictions."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tokenizer-cache',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--splits',nargs='+',choices=['train','dev'],default=['train','dev']);a=ap.parse_args()
project=Path(__file__).resolve().parents[1];upstream=a.assets/'third_party/openpi-rlt'
for p in [project,upstream/'src',upstream/'scripts',upstream/'packages/openpi-client/src']:sys.path.insert(0,str(p))
os.environ['OPENPI_DATA_HOME']=str(a.tokenizer_cache);os.environ['WANDB_MODE']='disabled'
os.environ['JAX_COMPILATION_CACHE_DIR']=str(a.output.parent/'eval-jax-cache')
import datasets,torch,numpy as np
from methods.openpi_rlt.realtime_expo.dataset import ExpertDataset
from methods.openpi_rlt.realtime_expo.policy import load
from methods.openpi_rlt.realtime_expo.metrics import episode_metrics,complete_episode_summary
from methods.openpi_rlt.plug_v3_yyshadow.train_stage1 import verify_upstream
verify_upstream(upstream)
if (a.output/'metrics.json').exists():raise FileExistsError('complete evaluation already exists')
a.output.mkdir(parents=True,exist_ok=True)
policy=load(a.checkpoint,a.output)
result=dict(checkpoint=str(a.checkpoint),metadata=policy.metadata,split_note='TRAIN was used for Stage1; DEV reused in EXPO and RLT selection; no independent TEST',dataset_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),cohorts={})
first=True
effective_batch_size=a.batch_size
for split in a.splits:
    dataset=ExpertDataset(a.data,split)
    if first:
        raw0=dataset[0]
        single,zsingle=policy.predict_batch([raw0])
        many,zmany=policy.predict_batch([raw0]*a.batch_size)
        diff=np.max(np.abs(single[0]-many[0]),axis=0)
        passed=bool(diff[:6].max()<=1e-3 and diff[6]<=1e-4)
        result['batch_consistency']=dict(requested_batch_size=a.batch_size,
            physical_max_abs_per_dim=diff.tolist(),token_max_abs=float(np.max(np.abs(zsingle[0]-zmany[0]))),
            passed=passed,action='requested_batch' if passed else 'fallback_to_deployed_single_observation')
        effective_batch_size=a.batch_size if passed else 1
        result['effective_batch_size']=effective_batch_size
        (a.output/'batch-consistency.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result['batch_consistency']),flush=True)
        first=False
    for identity in dataset.identities:
        uuid=identity['uuid'];indices=[i for i,row in enumerate(dataset.rows) if row[0]==uuid]
        predictions=[];tokens=[];targets=[];valid=[];timings=[];frame_indices=[];states=[]
        for offset in range(0,len(indices),effective_batch_size):
            selected=indices[offset:offset+effective_batch_size];raw=[dataset[i] for i in selected]
            batch=raw+[raw[-1]]*(effective_batch_size-len(raw))
            start=time.perf_counter();pred,z=policy.predict_batch(batch);timings.append(time.perf_counter()-start)
            predictions.append(pred[:len(raw)]);tokens.append(z[:len(raw)])
            for i,row in zip(selected,raw):
                _,t,end=dataset.rows[i];targets.append(row['actions']);valid.append(np.arange(50)<end-t);frame_indices.append(t);states.append(row['state'])
        predicted=np.concatenate(predictions);target=np.asarray(targets);mask=np.asarray(valid)
        metrics=episode_metrics(predicted,target,mask)
        times=np.asarray(frame_indices);adjacent=np.diff(times)==1
        if adjacent.any():
            replan=np.abs(predicted[1:,:9]-predicted[:-1,1:10]);overlap=mask[1:,:9]&mask[:-1,1:10]&adjacent[:,None]
            metrics['one_frame_replan_mae_per_dim']=replan[overlap].mean(0).tolist()
        record=dict(uuid=uuid,native_id=identity['native_id'],frames=len(indices),evaluated_frames=len(predicted),metrics=metrics,seconds=sum(timings))
        result['cohorts'].setdefault(split,{'per_episode':[]})['per_episode'].append(record)
        np.savez_compressed(a.output/f"{split}_{identity['native_id']:06d}.npz",predicted=predicted,z_rl=np.concatenate(tokens),target=target,future_valid=mask,source_frame_indices=times,state=np.asarray(states))
        (a.output/'progress.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(split=split,episode=identity['native_id'],frames=len(predicted),joint_mae=metrics['c10_joint_mae_rad'],gripper_mae=metrics['c10_gripper_mae_m'],seconds=sum(timings))),flush=True)
    result['cohorts'][split]['summary']=complete_episode_summary(result['cohorts'][split]['per_episode'])
(a.output/'metrics.json').write_text(json.dumps(result,indent=2))
print('evaluation complete',flush=True)
