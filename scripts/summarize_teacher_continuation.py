#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();o=a.root;r=json.loads((o/'study.json').read_text());assert r['status']=='complete'
metrics=['human_mae_mrad','reference_mae_mrad','actor_change_mrad','qrecord_mse']
def rows(es,split):return [e for e in es if e['split']==split]
def mean(es,k):
 v=[np.mean(e[k]) for e in es if e[k] is not None];return float(np.mean(v)) if v else None
summary={'baseline':{},'arms':{},'limits':r['limits']}
for split in ['new_train','new_dev','old_dev']:
 summary['baseline'][split]={k:mean(rows(r['baseline'],split),k) for k in metrics}
for arm in ['teacher50','teacher5']:
 summary['arms'][arm]={}
 for split in summary['baseline']:
  values={k:[mean(rows(run['curve'][-1]['evaluation'],split),k) for run in r['runs'] if run['arm']==arm] for k in metrics}
  summary['arms'][arm][split]={k:dict(mean=float(np.mean(v)),seed_min=min(v),seed_max=max(v),change_pct=(float(np.mean(v))/summary['baseline'][split][k]-1)*100 if summary['baseline'][split][k] else None) for k,v in values.items()}
# Paired old DEV confidence: resample complete Episodes after averaging paired seeds.
rng=np.random.default_rng(20261012);ci={}
for arm in summary['arms']:
 ci[arm]={}
 for k in ['human_mae_mrad','reference_mae_mrad']:
  base={e['episode']:np.mean(e[k]) for e in rows(r['baseline'],'old_dev') if e[k] is not None}
  predictions=[]
  for run in r['runs']:
   if run['arm']!=arm:continue
   predictions.append({e['episode']:np.mean(e[k]) for e in rows(run['curve'][-1]['evaluation'],'old_dev') if e[k] is not None})
  ids=sorted(base);d=np.array([np.mean([v[e] for v in predictions])-base[e] for e in ids]);boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
  ci[arm][k]=dict(episodes=len(d),mean_change_mrad=float(d.mean()),ci95_mrad=np.quantile(boot,[.025,.975]).tolist())
summary['old_dev_paired_episode_bootstrap']=ci
(o/'summary.json').write_text(json.dumps(summary,indent=2))
colors={'teacher50':'#3268ac','teacher5':'#cf7136'}
fig,axs=plt.subplots(2,3,figsize=(13,7))
for j,split in enumerate(['new_train','new_dev','old_dev']):
 for i,k in enumerate(['human_mae_mrad','reference_mae_mrad']):
  ax=axs[i,j];b=summary['baseline'][split][k]
  for arm in colors:
   selected=[run for run in r['runs'] if run['arm']==arm];xx=[0]+[c['update'] for c in selected[0]['curve']]
   yy=np.array([[b]+[mean(rows(c['evaluation'],split),k) for c in run['curve']] for run in selected])
   ax.plot(xx,yy.mean(0),label=arm,color=colors[arm]);ax.fill_between(xx,yy.min(0),yy.max(0),alpha=.15,color=colors[arm])
  ax.set_title(split+' / '+k.replace('_mae_mrad',''));ax.set_xlabel('Critic updates');ax.set_ylabel('Episode-mean MAE (mrad)');ax.legend()
fig.suptitle('Same HIL8 7281 start, teacher and batch schedule; 3 seeds. Bands = seed range. DEV reused, no TEST.');fig.tight_layout();fig.savefig(o/'learning-curves.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,3.8));sampling=r['sampling']
for i,k in enumerate(['new_ratio','new_human_ratio','human_ratio']):
 ax=axs[i]
 for j,arm in enumerate(colors):
  for off,role in [(-.16,'critic'),(.16,'actor')]:
   vs=[100*x[role][k] for x in sampling if x['arm']==arm];ax.bar(j+off,np.mean(vs),.28,label=role if j==0 else None,color='#3268ac' if role=='critic' else '#cf7136');ax.scatter(np.full(3,j+off),vs,s=14,color='black')
 ax.set_xticks([0,1]);ax.set_xticklabels(['Teacher50','Teacher5']);ax.set_title(k);ax.set_ylabel('% actual draws');ax.legend()
fig.suptitle('Actual scheduled batches; sources overlap. human includes expert-coded HUMAN. DEV draws = 0');fig.tight_layout();fig.savefig(o/'sampling.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for ax,split in zip(axs,['new_train','new_dev','old_dev']):
 base=rows(r['baseline'],split);ids=[e['episode'] for e in base if e['human_mae_mrad'] is not None]
 b=np.mean([e['human_mae_mrad'] for e in base if e['human_mae_mrad'] is not None],axis=0);ax.plot(range(1,7),b,'k--',label='7281 start')
 for arm,col in colors.items():
  vals=[np.mean([e['human_mae_mrad'] for e in rows(run['curve'][-1]['evaluation'],split) if e['human_mae_mrad'] is not None],axis=0) for run in r['runs'] if run['arm']==arm];ax.plot(range(1,7),np.mean(vals,axis=0),label=arm,color=col)
 ax.set_title(split+' HIL (%d Episodes)'%len(ids));ax.set_xlabel('Active joint');ax.set_ylabel('MAE mrad');ax.legend()
fig.suptitle('Observed HIL action fit, not optimality or autonomous success');fig.tight_layout();fig.savefig(o/'hil-joint-errors.png',dpi=150);plt.close(fig)
print(json.dumps(summary,indent=2))

fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,split in zip(axs,['new_train','new_dev']):
 es=[e for e in r['baseline'] if e['split']==split and e['pure_hil_delta_q'] is not None];assert len(es)==1
 ax.axhline(0,color='gray');ax.axhline(es[0]['pure_hil_delta_q'][0],color='black',linestyle='--',label='7281 Q1 delta')
 for i,arm in enumerate(colors):
  vals=[next(e['pure_hil_delta_q'] for e in run['curve'][-1]['evaluation'] if e['split']==split and e['pure_hil_delta_q'] is not None) for run in r['runs'] if run['arm']==arm]
  ax.scatter(np.full(3,i),np.array(vals)[:,0],label=arm+' Q1',color=colors[arm]);ax.scatter(np.full(3,i+.15),np.array(vals)[:,2],marker='x',color=colors[arm])
 ax.set_xticks([0,1]);ax.set_xticklabels(['Teacher50','Teacher5']);ax.set_ylabel('Q(HIL) - Q(Actor)');ax.set_title(split+': 1 HIL Episode');ax.legend(fontsize=7)
fig.suptitle('Same-state pure-HIL chunks; dots=Q1, x=min-Q. Episode means, not success proof.');fig.tight_layout();fig.savefig(o/'paired-q.png',dpi=150);plt.close(fig)
first=np.load(o/'teacher50_seed41_batches.npz');identities=first['identities'];splits=first['splits']
lookup={(e['split'],e['phase'],e['episode']):e['outcome'] for e in r['baseline']}
category=[]
for split,(phase,ep,step) in zip(splits,identities):
 if split=='old_train':category.append('old_expert' if ep<0 else 'old_rollout')
 elif split=='new_train':category.append('new_'+lookup[(str(split),int(phase),int(ep))])
 else:category.append('DEV')
category=np.array(category);composition={}
for arm in colors:
 composition[arm]={}
 for role in ['critic','actor']:
  vs=[]
  for seed in [41,42,43]:
   z=np.load(o/('%s_seed%d_batches.npz'%(arm,seed)))[role]
   if role=='actor':z=z[0::2]
   vs.append({c:float(np.mean(category[z]==c)) for c in sorted(set(category))})
  composition[arm][role]={c:float(np.mean([v[c] for v in vs])) for c in vs[0]}
(o/'sampling-composition.json').write_text(json.dumps(composition,indent=2))
