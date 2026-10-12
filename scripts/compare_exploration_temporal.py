#!/usr/bin/env python3
"""Open-loop recorded-target counterfactual, not a robot success prediction."""
import argparse,json,hashlib,sys,importlib.util,math
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();o=a.output;o.mkdir(exist_ok=False,parents=True)
src=a.assets/'outputs/chunk-continuity-20261011/commands.json';data=json.loads(src.read_text())
path=a.assets.parent/'vla-platform/integrations/cobot/pi05/dagger/common/runtime_lib/execution_methods/execution_timing.py'
spec=importlib.util.spec_from_file_location('timing',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
result=[];variants=['original_plan_ar','episode_constant','no_exploration'];curves={}
for ep in data:
 rows=ep['rows'];first=next((r['exploration']['offset_rad'] for r in rows if r.get('exploration') and r['exploration'].get('offset_rad') is not None),None)
 if first is None:continue
 constant=np.array(first);state={};seq={k:[] for k in variants};targets={k:[] for k in variants};actual=[];t=[];epoch=None;logical=0;start=None;previous={};epoch_time=0.
 for r in rows:
  pubs=r['publications'] or [];exp=r.get('exploration')
  if r.get('planned_action') is None or not pubs:continue
  target=np.array(r['planned_action'][:6]);oldoffset=np.array(exp['offset_rad']) if exp and exp.get('offset_rad') is not None else np.zeros(6)
  new_epoch=pubs[0]['execution_epoch']
  if new_epoch!=epoch:
   epoch=new_epoch;logical=0;epoch_time=pubs[0]['monotonic_timestamp']
   start=(np.array(pubs[0]['ref_action'][:6])-.4*np.array(r['logical_ref_action'][:6]))/.6
   state={k:mod.CausalJointFilter(.08) for k in variants}
   previous={k:start.copy() for k in variants}
   for f in state.values():f.reset(start)
  x=np.clip(pubs[0]['monotonic_timestamp']-epoch_time,0.,1.);gain=x**3*(10-15*x+6*x*x)
  candidate=target-oldoffset+(gain*constant if exp else 0.)
  goal={'original_plan_ar':target,'episode_constant':candidate,'no_exploration':target-oldoffset}
  n0=int(math.floor(logical*2.5+1e-9));n1=int(math.floor((logical+1)*2.5+1e-9));alphas=[k/2.5-logical for k in range(n0+1,n1+1)]
  for pub,alpha in zip(pubs,alphas):
   actual.append(pub['action'][:6]);t.append(pub['monotonic_timestamp'])
   for name in variants:
    req=previous[name]+alpha*(goal[name]-previous[name]);targets[name].append(req)
    seq[name].append(state[name].apply(req,state[name].previous,.02).copy())
  previous=goal;logical+=1
 if len(t)<10:continue
 times=np.array(t);valid=(np.diff(times)>0)&(np.diff(times)<.08);metrics={}
 for name in variants:
  y=np.array(seq[name]);v=np.diff(y,axis=0)/.02;acc=np.diff(v,axis=0)/.02;mask=valid[:-1]&valid[1:]
  tail=times>times[-1]-3
  metrics[name]=dict(acc_rms=float(np.sqrt(np.mean(acc[mask]**2))),acc_joint_rms=np.sqrt(np.mean(acc[mask]**2,axis=0)).tolist(),path_rad=float(np.abs(np.diff(y,axis=0)[valid]).sum()),tail3s_path_rad=float(np.abs(np.diff(y[tail],axis=0)).sum()),target_rmse_mrad=float(np.sqrt(np.mean((y-np.array(targets[name]))**2))*1000))
 result.append(dict(path=ep['path'],outcome=rows[-1]['outcome'],metrics=metrics,baseline_actual_rmse_mrad=float(np.sqrt(np.mean((np.array(seq['original_plan_ar'])-np.array(actual))**2))*1000)))
 if len(curves)<3:curves[ep['path']]=dict(time=(times-times[0]).tolist(),**{k:np.array(seq[k]).tolist() for k in variants})
summary=dict(input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),episodes=len(result),baseline_actual_rmse_mrad_median=float(np.median([r['baseline_actual_rmse_mrad'] for r in result])),results=result,limits=['All trajectories are reused DEV','Open-loop fixed Actor/Reference target counterfactual; observations and contact do not react','Constant candidate uses first recorded offset, not cherry-picked direction','Ramp timing approximated on recorded logical rows; implementation unit tests use exact per-plan time','Same existing EMA08/50Hz kept; neither closed-loop smoothness nor success verified'])
for k in variants[1:]:summary[k+'_median_ratios']={m:float(np.median([r['metrics'][k][m]/r['metrics']['original_plan_ar'][m] for r in result])) for m in ['acc_rms','path_rad','tail3s_path_rad']}
(o/'summary.json').write_text(json.dumps(summary,indent=2));(o/'curves.json').write_text(json.dumps(curves))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for ax,m,label in zip(axes,['acc_rms','path_rad','tail3s_path_rad'],['Command acceleration RMS','Command path length','Last 3s command path']):
 for i,k in enumerate(variants[1:]):
  y=[r['metrics'][k][m]/r['metrics']['original_plan_ar'][m] for r in result];ax.scatter(np.full(len(y),i),y,alpha=.45);ax.plot([i-.18,i+.18],[np.median(y)]*2,color='black')
 ax.axhline(1,color='gray',linestyle='--');ax.set_xticks([0,1]);ax.set_xticklabels(['Constant/1s ramp','Exploration off']);ax.set_title(label);ax.set_ylabel('Ratio to recorded per-plan exploration')
fig.suptitle('29 reused DEV Episodes; open-loop command replay, not robot results');fig.tight_layout();fig.savefig(o/'temporal-comparison.png',dpi=150)
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
