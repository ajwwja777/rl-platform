"""Render the 2026-10-07 matched sampling study; uncertainty units are Episodes."""
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();o=a.output
    load=lambda f:json.loads((o/f).read_text())
    study=load('episode-sampling-study/comparison.json');period=load('period-metrics.json');grad=load('gradient-sensitivity.json')
    rows=[json.loads(x)for x in (o/'episode-sampling-study/metrics.jsonl').read_text().splitlines()]
    groups=collections.defaultdict(list)
    for row in rows:
        groups[row['variant']].append(row)
        assert all(np.isfinite(v)for v in row.values()if isinstance(v,(float,int)))
    assert len(groups)==6
    samples=np.load(o/'episode-sampling-study/sample_indices.npz')
    arrivals=load('episode-sampling-study/arrival_schedule.json')
    verification={}
    for name,rs in groups.items():
        assert [r['update_index']for r in rs]==list(range(1,282))
        assert sum(r['did_actor_update']for r in rs)==140
        ix=samples[name];assert ix.shape==(281,128)
        for n in range(1,282):
            available=max(x['available']for x in arrivals if x['first_update']<=n)
            assert ix[n-1].min()>=0 and ix[n-1].max()<available
        checkpoint=o/'episode-sampling-study'/name/'step_7281.pkl'
        verification[name]={'critic_updates':281,'actor_updates':140,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (o/'verification.json').write_text(json.dumps(verification,indent=2))
    rng=np.random.default_rng(20261007);ci={}
    for group in ['autonomous','assisted','failure']:
        ci[group]={}
        for metric in ['bc6_rmse_mrad','bc6_abs_p95_mrad','q1_behavior_mse']:
            byep=collections.defaultdict(list)
            for seed in [41,42,43]:
                x=study['evaluations'][f'window_recent_seed{seed}'][-1]['repeated_DEV']['episodes'];y=study['evaluations'][f'episode_recent_seed{seed}'][-1]['repeated_DEV']['episodes']
                lookup={(v['phase'],v['episode']):v for v in y}
                for v in x:
                    if v['group']==group:byep[(v['phase'],v['episode'])].append(lookup[(v['phase'],v['episode'])][metric]-v[metric])
            assert all(len(v)==3 for v in byep.values())
            d=np.array([np.mean(v)for v in byep.values()]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
            ci[group][metric]={'episodes':len(d),'mean_episode_difference':float(d.mean()),'paired_episode_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'boundary':'Average seeds within each Episode before bootstrap. Repeated DEV only; no independent TEST or seed uncertainty.'}
    (o/'DEV-paired-episode-intervals.json').write_text(json.dumps(ci,indent=2))
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    fig,ax=plt.subplots(1,2,figsize=(12,4));xx=np.arange(5);fail=[100*x['sampling']['outcome']['failure']['ratio']for x in period['periods']]
    ax[0].bar(xx,fail,color='#dc7958');ax[0].set_xticks(xx);ax[0].set_xticklabels(['3 auto S','+ failure A','+ failure B','+ assisted S','+ failure C'],rotation=15);ax[0].set_ylabel('Actual failure draws (%)');ax[0].set_ylim(0,27)
    for i,v in enumerate(fail):ax[0].text(i,v+.5,f'{v:.1f}',ha='center')
    ax[0].set_title('TRAIN: batch mix changes with arrivals')
    labels=['Expert','Autonomous S','Assisted S','Failure'];buffer=np.array([1186,497,428,666])/2777*100;draw=np.array([12927,14890,3715,4436])/35968*100
    x=np.arange(4);ax[1].bar(x-.18,buffer,.36,label='Current buffer windows');ax[1].bar(x+.18,draw,.36,label='All 281 Critic batches');ax[1].set_xticks(x);ax[1].set_xticklabels(labels,rotation=15);ax[1].set_ylabel('Share (%)');ax[1].legend(fontsize=8);ax[1].set_title('Window composition != sampled composition')
    fig.tight_layout();fig.savefig(o/'01-actual-sampling.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4));norm=np.array([x['gradient_norms']for x in grad['batches']]);cos=np.array([x['gradient_cosine_with_new_HIL_BC']for x in grad['batches']]);labels=['BC','negative Q','delta','retention']
    for j,label in enumerate(labels):
        ax[0].plot(range(6),norm[:,j],marker='o',label=label);ax[1].plot(range(6),cos[:,j],marker='o',label=label)
    for a in ax:a.set_xticks(range(6));a.set_xticklabels([x['recorded_batch_step']for x in grad['batches']],rotation=25);a.set_xlabel('Recorded batch identity; recomputed at 7281')
    ax[0].set_yscale('log');ax[0].set_ylabel('Weighted gradient norm');ax[0].legend();ax[1].axhline(0,color='gray',lw=1);ax[1].set_ylabel('Cosine with HIL gradient (+ helpful)');ax[1].legend();fig.suptitle('One HIL TRAIN Episode, active6: local sensitivity, no Episode CI');fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(o/'02-gradient-direction.png');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4));colors=['#168a9f','#bf812d','#7961ad']
    for i,seed in enumerate([41,42,43]):
        aa=study['evaluations'][f'window_recent_seed{seed}'][-1];bb=study['evaluations'][f'episode_recent_seed{seed}'][-1]
        for j,(group,metric)in enumerate([('new_HIL_TRAIN','BC6_mse'),('new_failure_TRAIN','Q1_vs_behavior_return_mse')]):
            ax[j].plot([0,1],[aa['groups'][group][metric],bb['groups'][group][metric]],marker='o',color=colors[i],label=f'Sampling seed {seed}')
    ax[0].set_title('New HIL fit: 1 TRAIN Episode');ax[0].set_ylabel('Active6 BC MSE (normalized)');ax[1].set_title('New failures: 3 TRAIN Episodes');ax[1].set_ylabel('Q1 vs observed-return MSE')
    for a in ax[:2]:a.set_xticks([0,1]);a.set_xticklabels(['Window recent','Episode recent']);a.legend(fontsize=8)
    for i,g in enumerate(['autonomous','assisted','failure']):
        d=ci[g]['bc6_rmse_mrad'];lo,hi=d['paired_episode_bootstrap_95'];mu=d['mean_episode_difference'];ax[2].errorbar(i,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',capsize=4)
    ax[2].axhline(0,color='gray',lw=1);ax[2].set_xticks(range(3));ax[2].set_xticklabels(['Auto (5)','Assisted (6)','Failure (9)']);ax[2].set_ylabel('Episode - window BC6 RMSE (mrad)');ax[2].set_title('Repeated DEV, paired Episode 95% CI');fig.tight_layout();fig.savefig(o/'03-matched-sampling-study.png');plt.close(fig)
    print(json.dumps({'verification':verification,'DEV_intervals':ci},indent=2))
if __name__=='__main__':main()
