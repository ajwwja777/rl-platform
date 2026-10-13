#!/usr/bin/env python3
"""Episode-equal single-factor diagnostics; reused DEV, no autonomous-release claim."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.titlesize':10})

o=Path(sys.argv[1]);r=json.loads((o/'study.json').read_text());assert r['status']=='complete'
arms=list(r['arms']);splits=['new_train','new_dev','old_dev'];colors=dict(zip(arms,['#3168a8','#bd7c17','#359867']))
def evrows(ev,split):return [e for e in ev if e['split']==split]
def avg(ev,key):
 vals=[np.mean(e[key]) for e in ev if e[key] is not None]
 return float(np.mean(vals)) if vals else None
keys=['human_mae_mrad','human_p95_mrad','reference_mae_mrad','actor_change_mrad','qrecord_mse']
result=dict(limits=r['limits'],initial={},arms={},gates={},gradient={})
for split in splits:result['initial'][split]={k:avg(evrows(r['baseline'],split),k) for k in keys}
for arm in arms:
 result['arms'][arm]={}
 for split in splits:
  vals={k:[avg(evrows(run['curve'][-1]['evaluation'],split),k) for run in r['runs'] if run['arm']==arm] for k in keys}
  result['arms'][arm][split]={k:dict(mean=float(np.mean(v)),seeds=v) for k,v in vals.items()}
 logs=[json.loads(l) for seed in [41,42,43] for l in (o/('%s_seed%d_metrics.jsonl'%(arm,seed))).read_text().splitlines()]
 norms=np.array([x['critic_grad_norm'] for x in logs]);result['gradient'][arm]=dict(raw_max=float(norms.max()),raw_p99=float(np.quantile(norms,.99)),clipped_updates=sum(int(x['critic_grad_clipped']) for x in logs),total_updates=len(logs))
 base=result['arms'].get('teacher5')
 if arm!='teacher5':
  vals=result['arms'][arm];new=np.array(vals['new_dev']['human_mae_mrad']['seeds']);old=np.array(base['new_dev']['human_mae_mrad']['seeds'])
  result['gates'][arm]=dict(new_hil_improvement=float(1-new.mean()/old.mean()),all_seeds_no_worse=bool((new<=old).all()),old_hil_change=vals['old_dev']['human_mae_mrad']['mean']/base['old_dev']['human_mae_mrad']['mean']-1,old_ref_change=vals['old_dev']['reference_mae_mrad']['mean']/base['old_dev']['reference_mae_mrad']['mean']-1,old_hil_p95_change=vals['old_dev']['human_p95_mrad']['mean']/base['old_dev']['human_p95_mrad']['mean']-1)
  g=result['gates'][arm];g['engineering_pass']=bool(g['new_hil_improvement']>=.005 and g['all_seeds_no_worse'] and g['old_hil_change']<=.02 and g['old_ref_change']<=.02 and g['old_hil_p95_change']<=.05)
# Paired uncertainty resamples complete old DEV Episodes after seed averaging.
rng=np.random.default_rng(20261013);result['episode_bootstrap']={}
for arm in arms:
 result['episode_bootstrap'][arm]={}
 for k in ['human_mae_mrad','reference_mae_mrad']:
  base={(e['phase'],e['episode']):np.mean(e[k]) for e in evrows(r['baseline'],'old_dev') if e[k] is not None}
  means=[]
  for run in r['runs']:
   if run['arm']==arm:means.append({(e['phase'],e['episode']):np.mean(e[k]) for e in evrows(run['curve'][-1]['evaluation'],'old_dev') if e[k] is not None})
  d=np.array([np.mean([m[ep] for m in means])-val for ep,val in base.items()]);samples=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
  result['episode_bootstrap'][arm][k]=dict(n=len(d),mean=float(d.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
fig,axs=plt.subplots(2,3,figsize=(13,7))
for j,split in enumerate(splits):
 for i,k in enumerate(['human_mae_mrad','reference_mae_mrad']):
  ax=axs[i,j]
  for arm in arms:
   runs=[run for run in r['runs'] if run['arm']==arm];xx=[0]+[x['update'] for x in runs[0]['curve']];yy=np.array([[result['initial'][split][k]]+[avg(evrows(x['evaluation'],split),k) for x in run['curve']] for run in runs]);ax.plot(xx,yy.mean(0),label=arm,color=colors[arm]);ax.fill_between(xx,yy.min(0),yy.max(0),color=colors[arm],alpha=.12)
  ax.set_title(split+' / '+k);ax.set_ylabel('Episode-mean MAE, mrad');ax.set_xlabel('Critic updates');ax.legend()
fig.suptitle('Same 7281 state / teacher5 / paired batches; band = 3-seed range. DEV reused, no TEST.');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(o/'learning-curves.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4));result['paired_q']={}
for ax,split in zip(axs,['new_train','new_dev']):
 result['paired_q'][split]={};ax.axhline(0,color='grey')
 for i,arm in enumerate(arms):
  qq=np.array([np.mean([e['pure_hil_delta_q'] for e in evrows(run['curve'][-1]['evaluation'],split) if e['pure_hil_delta_q'] is not None],axis=0) for run in r['runs'] if run['arm']==arm]);result['paired_q'][split][arm]=qq.tolist();ax.scatter(np.full(3,i)-.06,qq[:,0],color=colors[arm],label=arm+' Q1');ax.scatter(np.full(3,i)+.06,qq[:,2],color=colors[arm],marker='x')
 ax.set_xticks(range(len(arms)),arms);ax.set_title(split+' / 1 HIL Episode');ax.set_ylabel('Q(HIL) - Q(Actor)');ax.legend(fontsize=7)
fig.suptitle('Paired current-checkpoint actions; dots Q1, crosses min-Q. Not historical counterfactual success.');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(o/'paired-q.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for arm in arms:
 vals=np.array([json.loads(l)['critic_grad_norm'] for l in (o/('%s_seed42_metrics.jsonl'%arm)).read_text().splitlines()]);ax.plot(np.arange(1,len(vals)+1),vals,label=arm,alpha=.7,color=colors[arm])
ax.axhline(10,color='black',linestyle='--',label='clip threshold 10');ax.set_xlabel('Critic updates / seed42');ax.set_ylabel('Raw global gradient norm');ax.set_yscale('log');ax.legend();fig.tight_layout(rect=(0,0,1,.92));fig.savefig(o/'gradient-norm.png',dpi=150);plt.close(fig)
(o/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(gates=result['gates'],gradient=result['gradient'])))

# Actual Actor/Critic exposure uses update parity, never final-buffer ratios.
first=np.load(o/'teacher5_seed42_batches.npz');ident=first['identities'];split=first['splits']
lookup={(e['split'],e['phase'],e['episode']):e['outcome'] for e in r['baseline']}
category=np.array(['old_expert' if sp=='old_train' and ep<0 else 'old_rollout' if sp=='old_train' else 'new_'+lookup[(str(sp),int(ph),int(ep))] if sp=='new_train' else 'DEV' for sp,(ph,ep,st) in zip(split,ident)])
comp={}
for role in ['critic','actor']:
 vals=[]
 for seed in [41,42,43]:
  ix=np.load(o/('teacher5_seed%d_batches.npz'%seed))[role]
  if role=='actor':ix=ix[0::2]
  vals.append({k:float(np.mean(category[ix]==k)) for k in sorted(set(category))})
 comp[role]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,role in zip(axs,['critic','actor']):
 cats=[k for k in comp[role] if k!='DEV'];vv=[100*comp[role][k] for k in cats]
 ax.barh(cats,vv,color='#3168a8');ax.set_xlabel('% actual draws');ax.set_title(role+' actual training batches')
 for i,v in enumerate(vv):ax.text(v+.2,i,'%.2f%%'%v,va='center',fontsize=9)
 ax.set_xlim(0,max(vv)*1.25)
fig.suptitle('Matched 3-seed batch exposure; DEV draws = 0\nAutonomous and assisted success are separate; no independent TEST');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(o/'actual-sampling.png',dpi=150);plt.close(fig)
(o/'actual-sampling.json').write_text(json.dumps(comp,indent=2))
# Fixed seed42 diagnostic Q and target distributions: each Episode receives equal mass.
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,sp in zip(axs,splits):
 mask=split==sp;idx=np.flatnonzero(mask);groups={}
 for i in idx:groups.setdefault(tuple(ident[i,:2]),[]).append(i)
 weights=np.zeros(len(ident))
 for ix in groups.values():weights[ix]=1./len(ix)/len(groups)
 for arm in arms:
  z=np.load(o/('%s_seed42_u846.npz'%arm));ax.hist(z['qrecord'][mask,0],bins=30,weights=weights[mask],histtype='step',label=arm+' Q1',color=colors[arm])
 z=np.load(o/'teacher5_seed42_u846.npz');ax.hist(z['mixed_target'][mask],bins=30,weights=weights[mask],histtype='step',linestyle='--',color='black',label='teacher5 mixed target')
 ax.set_title(sp);ax.set_xlabel('Q / diagnostic target');ax.set_ylabel('Episode-equal mass');ax.legend(fontsize=7)
fig.suptitle('Reused TRAIN/DEV; fixed diagnostic target RNG, not actual batch targets. No independent TEST.');fig.tight_layout(rect=(0,0,1,.92));fig.savefig(o/'q-target-distributions.png',dpi=150);plt.close(fig)
review=json.loads((o.parent/'awbc-retention-review.json').read_text())
fig,ax=plt.subplots(figsize=(8,4))
for seed in [41,42,43]:
 rr=[x for x in review['comparisons'] if x['seed']==seed];ax.plot([x['updates'] for x in rr],[100*x['relative_change']['autonomous']['bc6_abs_p95_mrad'] for x in rr],marker='o',label='seed%d'%seed)
ax.axhline(0,color='gray');ax.set_xlabel('Additional updates from Supported7000');ax.set_ylabel('Old autonomous DEV p95 fit error change vs paired A0 (%)');ax.set_title('W2 AWBC retention cost; reused DEV, not task success');ax.legend();fig.tight_layout(rect=(0,0,1,.92));fig.savefig(o/'awbc-retention.png',dpi=150);plt.close(fig)
