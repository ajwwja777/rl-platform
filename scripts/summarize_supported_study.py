#!/usr/bin/env python3
"""Episode-cluster summaries/figures from recorded CPU studies; no model loads."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();o=a.output_root
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    def read(path):return json.loads((o/path).read_text())
    studies={k:read(k+'/study.json')for k in ['retained_native','held_contract','small_lr']}
    first=read('external20/external20_active6.json');second=read('external20-small-lr/external20_active6.json')
    base=second['baseline'];runs=first['runs']+second['runs']
    selected=next(r for r in runs if r['study']=='small_lr'and r['variant']=='mc03_q01'and r['seed']==42)
    def field(e,name):
        v=e[name]
        if v is None:return None
        return float(np.mean(v))if name.endswith('mae_mrad')else float(v[0])
    def paired(name,outcome=None):
        bs={e['episode_id']:e for e in base}
        values=[];initial=[];final=[]
        for e in selected['evaluation']:
            b=bs[e['episode_id']]
            if outcome and e['outcome']!=outcome:continue
            x,y=field(b,name),field(e,name)
            if x is not None and y is not None:initial.append(x);final.append(y);values.append(y-x)
        if not values:return dict(episodes=0,value=None,reason='No eligible complete Episodes')
        values=np.asarray(values);rng=np.random.default_rng(20261006);boot=np.mean(values[rng.integers(len(values),size=(10000,len(values)))],axis=1)
        return dict(episodes=len(values),initial=float(np.mean(initial)),candidate=float(np.mean(final)),paired_delta=float(values.mean()),paired_delta_episode_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),direction='decrease is better'if'mae'in name or'mse'in name else'context dependent')
    metrics=dict(auto_reference_fit=paired('reference_mae_mrad','autonomous_success'),assisted_recorded_fit=paired('human_mae_mrad','assisted_success'),observed_return_q1_mse=paired('recorded_q_mse'),failed_recorded_q1=paired('qrecord','failure'),pure_hil_minus_actor_q1=paired('pure_hil_record_minus_actor_q','assisted_success'))
    figdir=o/'figures';figdir.mkdir(exist_ok=True)
    labels=[];groups=[]
    for study,variant,label in [('retained_native','mc03_q01','native MC30\nLR1e-4'),('held_contract','mc03_q01','held MC30\nLR1e-4'),('held_contract','mc10_q01','held MC100\nLR1e-4'),('held_contract','mc10_q001','held MC100 Q.01\nLR1e-4'),('small_lr','mc03_q01','held MC30\nLR1e-5'),('small_lr','mc10_q01','held MC100\nLR1e-5')]:
        labels.append(label);groups.append([r for r in runs if r['study']==study and r['variant']==variant])
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for ax,name,outcome,title in zip(axes,['reference_mae_mrad','human_mae_mrad','recorded_q_mse'],['autonomous_success','assisted_success',None],['Auto DEV: active6 Reference fit (mrad)','Assisted DEV: active6 logged HIL fit (mrad)','All DEV: observed behavior-return Q1 MSE']):
        bs=[field(e,name)for e in base if(not outcome or e['outcome']==outcome)and field(e,name)is not None];ax.axhline(np.mean(bs),color='black',linestyle='--',label='initial5k')
        for j,rs in enumerate(groups):
            for r in rs:
                vals=[field(e,name)for e in r['evaluation']if(not outcome or e['outcome']==outcome)and field(e,name)is not None]
                ax.scatter(j+(r['seed']-42)*.12,np.mean(vals),s=32,color={41:'#386cb0',42:'#f0027f',43:'#7fc97f'}[r['seed']])
        ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=40,ha='right',fontsize=8);ax.set_title(title,fontsize=10)
    
    for seed,color in [(41,'#386cb0'),(42,'#f0027f'),(43,'#7fc97f')]:axes[0].scatter([],[],color=color,label='seed%d'%seed)
    axes[0].legend(loc='upper left',fontsize=8);fig.suptitle('External20 reused DEV (5 autonomous / 6 assisted / 9 failed); points = seeds, not confidence intervals')
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(figdir/'seed_stability.png');plt.close(fig)
    outcomes=['autonomous_success','assisted_success','failure'];colors=['#1b9e77','#d95f02','#7570b3']
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4))
    for j,(kind,color)in enumerate(zip(outcomes,colors)):
        b=[e for e in base if e['outcome']==kind];c=[e for e in selected['evaluation']if e['outcome']==kind]
        for offset,es in [(-.14,b),(.14,c)]:
            axes[0].scatter(np.full(len(es),j+offset),[e['recorded_q_mse'][0]for e in es],color=color,alpha=.8,marker='o'if offset<0 else's')
        axes[1].scatter([e['observed_return_mean']for e in c],[e['qrecord'][0]for e in c],color=color,label=kind.replace('_',' '))
    axes[0].set_xticks(range(3));axes[0].set_xticklabels(['Auto (5)','Assisted (6)','Failed (9)']);axes[0].set_title('Q1 behavior-return MSE per Episode\ncircle initial5k / square candidate')
    axes[1].plot([0,1],[0,1],color='black',linestyle='--');axes[1].set_xlabel('Mean observed return (help included)');axes[1].set_ylabel('Candidate mean recorded-action Q1');axes[1].legend(fontsize=8)
    bmap={e['episode_id']:e for e in base};pairs=[e for e in selected['evaluation']if e['pure_hil_record_minus_actor_q']is not None]
    for i,e in enumerate(pairs):axes[2].plot([0,1],[bmap[e['episode_id']]['pure_hil_record_minus_actor_q'][0],e['pure_hil_record_minus_actor_q'][0]],marker='o',alpha=.8,label=str(e['episode_id']))
    axes[2].axhline(0,color='black',linestyle='--');axes[2].set_xticks([0,1]);axes[2].set_xticklabels(['Initial5k','Candidate']);axes[2].set_title('Pure-HIL minus reconstructed Actor Q1\nactive6 alternatives; no counterfactual outcome')
    fig.suptitle('Critic diagnostics: external20 reused DEV, seed42 MC30 LR1e-5');fig.tight_layout(rect=(0,0,1,.88));fig.savefig(figdir/'critic_calibration.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4));internal=studies['small_lr'];chosen=next(r for r in internal['runs']if r['seed']==42 and r['variant']=='mc03_q01')
    scopes=[('training','autonomous_success','policy_reference_mae','TRAIN auto Reference'),('historical_online_development','assisted_success','human_mae','Old6 reused DEV HIL')]
    for split,outcome,name,label in scopes:
        points=[(0,internal['baseline'])]+[(c['updates'],c['evaluation'])for c in chosen['curve']]
        xs=[];ys=[]
        for x,es in points:
            vals=[np.mean(e[name][:6])*1000 for e in es if e['split']==split and e['outcome']==outcome and e[name]is not None]
            xs.append(x);ys.append(np.mean(vals))
        axes[0].plot(xs,ys,marker='o',label=label)
    axes[0].set_xlabel('Additional Critic updates (start full5k)');axes[0].set_ylabel('Episode-equal active6 MAE (mrad)');axes[0].legend();axes[0].set_title('Fixed-budget action-fitting proxies')
    auto=[e for e in selected['evaluation']if e['outcome']=='autonomous_success'];assist=[e for e in selected['evaluation']if e['outcome']=='assisted_success']
    for es,name,label in [(auto,'reference_mae_mrad','Auto DEV Reference'),(assist,'human_mae_mrad','Assisted DEV logged HIL')]:
        ids={e['episode_id']for e in es};initial=np.mean([e[name]for e in base if e['episode_id']in ids],axis=0);final=np.mean([e[name]for e in es],axis=0)
        axes[1].plot(range(1,7),initial,linestyle='--',label=label+' initial');axes[1].plot(range(1,7),final,marker='o',label=label+' candidate')
    axes[1].set_xlabel('Right joint');axes[1].set_ylabel('MAE (mrad)');axes[1].set_title('External20 DEV per active joint');axes[1].legend(fontsize=8)
    fig.suptitle('Inactive gripper excluded from capability gates; fit does not prove autonomous success');fig.tight_layout(rect=(0,0,1,.88));fig.savefig(figdir/'action_fit.png');plt.close(fig)
    ident=dict(np.load(o/'small_lr/data_identity.npz'));groups2={}
    for i in range(len(ident['episode_id'])):groups2.setdefault((bool(ident['phase_online'][i]),int(ident['episode_id'][i])),[]).append(i)
    keys={tuple(k)for k in internal['train_episodes']};train=np.asarray([i for k in sorted(keys)for i in groups2[k]])
    categories=[];episode_counts=np.zeros(4);window_counts=np.zeros(4)
    for k in sorted(keys):
        ids=np.asarray(groups2[k]);cat=0 if ident['expert'][ids].all()else 2 if ident['human'][ids].any()and ident['success'][ids].any()else 1 if ident['success'][ids].any()else 3
        episode_counts[cat]+=1;window_counts[cat]+=len(ids);categories.extend([cat]*len(ids))
    categories=np.asarray(categories);sample_counts=[]
    for seed in [41,42,43]:
        draws=np.load(o/('small_lr/indices_seed%d.npy'%seed));sample_counts.append(np.bincount(categories[draws].ravel(),minlength=4))
    proportions=dict(episode=(episode_counts/episode_counts.sum()).tolist(),windows=(window_counts/window_counts.sum()).tolist(),actual_batch_seed_mean=np.mean([x/x.sum()for x in sample_counts],axis=0).tolist(),episode_counts=episode_counts.astype(int).tolist(),window_counts=window_counts.astype(int).tolist(),categories=['expert','autonomous_success','assisted_success','failure'])
    fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(3)
    vals=np.array([proportions['episode'],proportions['windows'],proportions['actual_batch_seed_mean']])*100
    for i,label in enumerate(proportions['categories']):
        ax.bar(range(3),vals[:,i],bottom=bottom,label=label.replace('_',' '))
        for j in range(3):ax.text(j,bottom[j]+vals[j,i]/2,'%.1f%%'%vals[j,i],ha='center',va='center',fontsize=9)
        bottom+=vals[:,i]
    ax.set_xticks(range(3));ax.set_xticklabels(['Complete TRAIN Episodes','Stored TRAIN windows','Actual sampled batches\n3 seeds x 2000 x 128']);ax.set_ylabel('Percent');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left');ax.set_ylim(0,100)
    ax.set_title('Disjoint outcomes; HIL is overlapping metadata, not a fifth category\nNative .4 recent / .3 Warmup / .2 HIL / .1 uniform pools overlap')
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(figdir/'sampling.png');plt.close(fig)
    internal_pairs={}
    for split,outcome,name in [('training','autonomous_success','policy_reference_mae'),('historical_online_development','assisted_success','human_mae')]:
        bs={(e['phase'],e['episode_id']):e for e in internal['baseline']};values=[];initial=[];final=[]
        for e in chosen['evaluation']:
            if e['split']!=split or e['outcome']!=outcome or e[name]is None:continue
            b=bs[(e['phase'],e['episode_id'])]
            x=float(np.mean(b[name][:6])*1000);y=float(np.mean(e[name][:6])*1000)
            initial.append(x);final.append(y);values.append(y-x)
        rng=np.random.default_rng(20261006);v=np.asarray(values)
        boot=np.mean(v[rng.integers(len(v),size=(10000,len(v)))],axis=1)
        internal_pairs[split+'_'+outcome]=dict(episodes=len(v),initial_mrad=float(np.mean(initial)),candidate_mrad=float(np.mean(final)),paired_delta_mrad=float(v.mean()),episode_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
    result=dict(status='verified',candidate=dict(study='small_lr',variant='mc03_q01',seed=42),external20_paired_episode_metrics=metrics,internal_paired_episode_metrics=internal_pairs,sampling=proportions,bootstrap=dict(resamples=10000,seed=20261006,unit='complete Episode, paired candidate-initial; never overlapping windows'),independent_test=None,limits=['Repeated DEV; no independent capability test.','Observed returns include future HIL; positive preference is not proof of optimality.','Actor actions are reconstructed from current candidate, not historical pre-intervention proposals.','Q1 used by Actor; TD uses min-Q. Plot/calibration summary uses Q1, arrays retain Q1/Q2/min-Q.','Old cached feature inputs cannot establish the magnitude of the newly found RTC identity bug.'])
    (o/'episode_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
