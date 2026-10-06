#!/usr/bin/env python3
"""Whole-Episode paired summaries of the cached CPU training study."""
import argparse
import json
from pathlib import Path
import numpy as np


def interval(values):
    a=np.asarray(values,float)
    if not len(a): return dict(episodes=0,mean=None,ci95=None)
    ci=None
    if len(a)>1:
        boot=np.random.default_rng(42).choice(a,(10000,len(a)),replace=True).mean(1)
        ci=np.quantile(boot,[.025,.975]).tolist()
    return dict(episodes=len(a),mean=float(a.mean()),ci95=ci)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    r=json.loads(a.study.read_text())
    def key(e):return (e['phase'],e['episode_id'])
    baseline={key(e):e for e in r['baseline']}
    selectors={
        'train_expert':lambda e:e['split']=='training' and e['outcome']=='expert',
        'train_autonomous_success':lambda e:e['split']=='training' and e['outcome']=='autonomous_success',
        'train_hil':lambda e:e['split']=='training' and e['outcome']!='expert' and e['human_mae']is not None,
        'dev_hil':lambda e:e['split']=='development' and e['outcome']!='expert' and e['human_mae']is not None,
        'historical_dev_hil':lambda e:e['split']=='historical_online_development' and e['outcome']=='assisted_success',
        'dev_autonomous_success':lambda e:e['split']=='development' and e['outcome']=='autonomous_success',
        'dev_failure':lambda e:e['split']=='development' and e['outcome']=='failure',
    }
    def metrics(e):
        out={}
        for field in ['human_mae','policy_reference_mae','recorded_mae','change_from_initial']:
            if e[field] is not None:
                out[field+'_joints_mrad']=float(np.mean(e[field][:6])*1000)
                out[field+'_grip_mm']=float(e[field][6]*1000)
        for field in ['hil_endpoint_minus_actor','pure_hil_record_minus_actor','qrecord_minus_actor']:
            if e[field] is not None:
                out[field+'_q1']=e[field][0];out[field+'_minq']=e[field][2]
        out['qactor_q1']=e['qactor'][0];out['qrecord_q1']=e['qrecord'][0]
        if e['observed_return_mean'] is not None:
            out['qrecord_minus_observed']=e['qrecord'][0]-e['observed_return_mean']
            out['td_minus_observed']=e['td_mean']-e['observed_return_mean']
        return out
    def summarize(episodes,ref=None):
        out={}
        for label,select in selectors.items():
            by={}
            for k,e in episodes.items():
                if not select(e):continue
                for name,value in metrics(e).items():
                    if ref is not None:
                        if name not in metrics(ref[k]):continue
                        value-=metrics(ref[k])[name]
                    by.setdefault(name,[]).append(value)
            out[label]={name:interval(v)for name,v in by.items()}
        return out
    runs=[]
    for run in r['runs']:
        es={key(e):e for e in run['evaluation']}
        runs.append(dict(seed=run['seed'],variant=run['variant'],mc_weight=run['mc_weight'],actor_q_weight=run['actor_q_weight'],summary=summarize(es),delta_from_initial=summarize(es,baseline),actual_online_ratio=run['actual_online_ratio']))
    contrasts={}
    for label,left,right in [('mc_with_q','mc03_q01','mc00_q01'),('mc_without_q','mc03_q00','mc00_q00'),('q_off_native','mc00_q00','mc00_q01'),('q_off_mc','mc03_q00','mc03_q01')]:
        all_d={}
        seed_summaries=[]
        for seed in sorted({x['seed']for x in r['runs']}):
            found={x['variant']:x for x in r['runs']if x['seed']==seed}
            if left not in found or right not in found:continue
            le={key(e):e for e in found[left]['evaluation']};ri={key(e):e for e in found[right]['evaluation']}
            seed_summaries.append(dict(seed=seed,summary=summarize(le,ri)))
            for k,e in le.items():
                for name,v in metrics(e).items():
                    if name in metrics(ri[k]):all_d.setdefault((k,name),[]).append(v-metrics(ri[k])[name])
        grouped={}
        for label2,select in selectors.items():
            by={}
            for (k,name),values in all_d.items():
                if select(baseline[k]):by.setdefault(name,[]).append(float(np.mean(values)))
            grouped[label2]={name:interval(v)for name,v in by.items()}
        contrasts[label]=dict(left=left,right=right,seeds=seed_summaries,seed_averaged_episode_deltas=grouped)
    result=dict(status=r['status'],source=str(a.study),baseline=summarize(baseline),runs=runs,contrasts=contrasts,independent_test=None,active_control='User confirms right_j1..j6 only; right gripper and left arm held throughout. Gripper output discrepancy is diagnostic only, not a task failure or candidate rejection gate.',
        uncertainty='Complete Episode resampling of per-Episode window-slot averages. Repeated windows never count as independent samples. Paired seeds averaged within an Episode before bootstrap; per-seed results also retained.',
        limitations=r['limitations'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    identity=np.load(a.study.parent/'data_identity.npz')
    train_keys={tuple(k) for k in r['train_episodes']}
    groups={}
    for i,(online,ep) in enumerate(zip(identity['phase_online'],identity['episode_id'])):
        groups.setdefault((bool(online),int(ep)),[]).append(i)
    train=np.asarray([i for k in sorted(train_keys) for i in groups[k]])
    classes={k:('expert' if identity['expert'][ids].all() else 'assisted_success' if identity['human'][ids].any() and identity['success'][ids].any() else 'autonomous_success' if identity['success'][ids].any() else 'failure')for k,ids in groups.items()}
    names=['expert','autonomous_success','assisted_success','failure']
    sampling=dict(episode_proportions={},window_proportions={},actual_batch_proportions={})
    for name in names:
        selected=np.asarray([classes[(bool(identity['phase_online'][i]),int(identity['episode_id'][i]))]==name for i in train])
        sampling['episode_proportions'][name]=sum(classes[k]==name for k in train_keys)/len(train_keys)
        sampling['window_proportions'][name]=float(selected.mean())
        sampled=[]
        for seed in sorted({x['seed']for x in r['runs']}):
            indices=np.load(a.study.parent/('indices_seed%d.npy'%seed))
            sampled.append(float(selected[indices].mean()))
        sampling['actual_batch_proportions'][name]=sampled
    result['sampling']=sampling
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if not a.plots:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=sorted({x['variant']for x in runs});colors=dict(zip(variants,['#365aab','#bd6837','#2a927d','#a35198']))
    fig,ax=plt.subplots(figsize=(8,4))
    xs=np.arange(len(names));width=.25
    for j,(label,values)in enumerate([('TRAIN Episodes',[sampling['episode_proportions'][n]for n in names]),('TRAIN windows',[sampling['window_proportions'][n]for n in names]),('Actual batch',[np.mean(sampling['actual_batch_proportions'][n])for n in names])]):
        ax.bar(xs+(j-1)*width,values,width,label=label)
    ax.set_xticks(xs);ax.set_xticklabels(names,rotation=10);ax.set_ylabel('Fraction');ax.legend(fontsize=8)
    ax.set_title('Disjoint outcome groups; assisted success is separate from autonomous success')
    fig.tight_layout();fig.savefig(a.output/'sampling.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    panels=[('historical_dev_hil','human_mae_joints_mrad','Assisted DEV: joint error (mrad)'),('historical_dev_hil','pure_hil_record_minus_actor_q1','Assisted DEV: pure HIL - Actor Q1'),('historical_dev_hil','hil_endpoint_minus_actor_q1','Assisted DEV: hybrid HIL - Actor Q1'),('train_expert','recorded_mae_joints_mrad','TRAIN expert: joint error (mrad)'),('dev_autonomous_success','policy_reference_mae_joints_mrad','Reused DEV autonomous: ref fit (mrad)'),('train_autonomous_success','policy_reference_mae_joints_mrad','TRAIN autonomous: reference fit (mrad)')]
    for ax,(group,metric,title)in zip(axes.flat,panels):
        base=result['baseline'].get(group,{}).get(metric,{}).get('mean')
        if base is not None:ax.axhline(base,color='black',ls='--',label='initial5k')
        for variant in variants:
            subset=[x for x in runs if x['variant']==variant]
            y=[x['summary'][group][metric]['mean']for x in subset if metric in x['summary'].get(group,{})]
            if y:ax.scatter(np.full(len(y),variants.index(variant)),y,color=colors[variant]);ax.plot(variants.index(variant),np.mean(y),'_',color=colors[variant],ms=22,mew=3)
        ax.set_xticks(range(len(variants)));ax.set_xticklabels(variants,rotation=20,fontsize=8);ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
    fig.suptitle('Cached CPU final2000 comparison: dots = seeds; no independent TEST')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(a.output/'final_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,group,metric,title in [(axes[0],'historical_dev_hil','human_mae_joints_mrad','Assisted DEV active6 joint fit (mrad)'),(axes[1],'historical_dev_hil','hil_endpoint_minus_actor_q1','Assisted DEV hybrid paired Q1'),(axes[2],'train_autonomous_success','policy_reference_mae_joints_mrad','TRAIN autonomous reference fit')]:
        for variant in variants:
            curves=[]
            for run in r['runs']:
                if run['variant']!=variant:continue
                points={0:result['baseline'][group].get(metric,{}).get('mean')}
                for point in run['curve']:
                    values=[metrics(e)[metric]for e in point['evaluation']if selectors[group](e) and metric in metrics(e)]
                    points[point['updates']]=float(np.mean(values))if values else None
                curves.append(points)
            if curves:
                xs=sorted(set.intersection(*(set(c)for c in curves)))
                ys=np.array([[c[x]for x in xs]for c in curves],float)
                ax.plot(xs,ys.mean(0),label=variant,color=colors[variant]);ax.fill_between(xs,ys.min(0),ys.max(0),color=colors[variant],alpha=.12)
        ax.set_title(title,fontsize=10);ax.set_xlabel('New Critic updates (Actor every2)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Reused DEV / TRAIN learning curves; shading = seed range, not task CI')
    fig.tight_layout();fig.savefig(a.output/'learning_curves.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,len(variants),figsize=(4*len(variants),4),squeeze=False)
    for ax,variant in zip(axes.flat,variants):
        for label,color in [('autonomous_success','#286daf'),('assisted_success','#dc893c'),('failure','#888888')]:
            es=[e for run in r['runs']if run['variant']==variant for e in run['evaluation']if e['split']=='development' and e['outcome']==label]
            xs=[e['observed_return_mean']for e in es if e['observed_return_mean']is not None]
            ys=[e['qrecord'][0]for e in es if e['observed_return_mean']is not None]
            ax.scatter(xs,ys,s=18,alpha=.6,label=label)
        ax.plot([0,1],[0,1],'k--',lw=.8);ax.set_xlim(-.03,1);ax.set_ylim(-.1,1.05);ax.set_title(variant);ax.set_xlabel('Episode mean observed return');ax.grid(alpha=.15)
    axes[0,0].set_ylabel('Episode mean recorded Q1');axes[0,0].legend(fontsize=7)
    fig.suptitle('Reused DEV calibration; dots repeat seeds, not independent episodes / autonomous values')
    fig.tight_layout();fig.savefig(a.output/'critic_calibration.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
