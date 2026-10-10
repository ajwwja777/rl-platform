#!/usr/bin/env python3
"""Render existing paired studies; no models, training, services or GPU."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(root):
    root=Path(root);out=root/"figures";out.mkdir(exist_ok=True)
    single=json.loads((root/"single-factor/comparison.json").read_text())
    follow=json.loads((root/"followup/comparison.json").read_text())
    terminal=json.loads((root/"terminal1-corrected/comparison.json").read_text())
    variants=[("clip_target",single),("takeover_credit",single),("critic_terminal2",single),("critic_terminal1",terminal),("actor_hil8",single),("hil8_clip",follow)]
    seeds=[41,42,43];groups=["autonomous","assisted","failure"]
    summary=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for k,(name,d) in enumerate(variants):
        for j,seed in enumerate(seeds):
            b=d["evaluations"][f"baseline_seed{seed}"][-1];c=d["evaluations"][f"{name}_seed{seed}"][-1]
            h=100*(c["groups"]["new_HIL_TRAIN"]["BC6_mse"]/b["groups"]["new_HIL_TRAIN"]["BC6_mse"]-1)
            q=max(100*(c["repeated_DEV"]["groups"][g]["q1_behavior_mse"]/b["repeated_DEV"]["groups"][g]["q1_behavior_mse"]-1) for g in groups[:2])
            tail=max(c["repeated_DEV"]["groups"][g]["bc6_abs_p95_mrad"]-b["repeated_DEV"]["groups"][g]["bc6_abs_p95_mrad"]for g in groups)
            summary.append(dict(variant=name,seed=seed,HIL_TRAIN_MSE_pct=h,worst_success_DEV_Q_MSE_pct=q,worst_DEV_tail_delta_mrad=tail,proxy_gate=q<=5 and tail<=.05 and h<0))
            axes[0].scatter(k+(j-1)*.1,h,color=["#2878b5","#e58b2a","#439b64"][j],label=str(seed)if k==0 else None)
            axes[1].scatter(k+(j-1)*.1,q,color=["#2878b5","#e58b2a","#439b64"][j])
    for ax in axes:ax.set_xticks(range(len(variants)));ax.set_xticklabels([x[0].replace("_","\n")for x in variants],fontsize=8);ax.axhline(0,color="gray",lw=.8);ax.grid(axis="y",alpha=.2)
    axes[0].set_title("TRAIN: new HIL fit (one Episode, no CI)");axes[0].set_ylabel("MSE change vs paired baseline (%)");axes[0].legend(title="sampling seed")
    axes[1].set_title("Reused DEV: worst autonomous/assisted Q proxy");axes[1].set_ylabel("Behavior-return MSE change (%)");axes[1].axhline(5,color="red",ls="--",label="predeclared 5% rejection gate");axes[1].legend(fontsize=8)
    fig.suptitle("Single-factor contrasts then combination; no independent TEST");fig.savefig(out/"01_ablation.png",dpi=160);plt.close(fig)
    b=follow["evaluations"]["baseline_seed42"][-1];c=follow["evaluations"]["hil8_clip_seed42"][-1]
    rng=np.random.default_rng(441);ci=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,g in enumerate(groups):
        br={x["episode"]:x for x in b["repeated_DEV"]["episodes"]if x["group"]==g};cr={x["episode"]:x for x in c["repeated_DEV"]["episodes"]if x["group"]==g}
        for metric in ["bc6_rmse_mrad","q1_behavior_mse"]:
            dif=np.array([cr[k][metric]-br[k][metric]for k in br]);boots=dif[rng.integers(0,len(dif),(5000,len(dif)))].mean(1);lo,hi=np.quantile(boots,[.025,.975]);ci.append(dict(group=g,metric=metric,n_episodes=len(dif),delta=float(dif.mean()),paired_episode_bootstrap95=[float(lo),float(hi)]))
        for tag,data,style in [("base",br,"--"),("HIL8+clip",cr,"-")]:
            joint=np.mean([x["bc6_joint_rmse_mrad"]for x in data.values()],axis=0)
            axes[0].plot(range(1,7),joint,style,color=["#2878b5","#439b64","#d54f42"][i],label=f"{g} {tag} (n={len(data)})")
        rr=ci[-2];axes[1].errorbar(i,rr["delta"],yerr=[[rr["delta"]-rr["paired_episode_bootstrap95"][0]],[rr["paired_episode_bootstrap95"][1]-rr["delta"]]],fmt="o",capsize=4)
    axes[0].set(xlabel="Active joint (gripper held; excluded)",ylabel="Mean Episode RMSE (mrad)",title="Reused DEV: joint fit");axes[0].legend(fontsize=7)
    axes[1].set(xticks=range(3),xticklabels=groups,ylabel="Candidate minus baseline RMSE (mrad)",title="Paired whole-Episode bootstrap 95%");axes[1].axhline(0,color="gray");fig.suptitle("Seed42, 20 repeatedly used DEV Episodes; no independent TEST");fig.savefig(out/"02_dev_retention.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,ep in zip(axes,["10004","10005","10010"]):
        for name,style in [("baseline","--"),("hil8_clip","-")]:
            series=[follow["initial"]]+follow["evaluations"][name+"_seed42"]
            for head in [0,1]:ax.plot([x["step"]for x in series],[x["all_failed_terminal_q"][ep][head]for x in series],style,color=["#2878b5","#e58b2a"][head],marker="o",label=f"{name} Q{head+1}")
        ax.axhline(0,color="black",lw=1);ax.set(title="TRAIN failure Episode "+ep,xlabel="Critic update",ylabel="Terminal Q (target = 0)");ax.tick_params(axis='x',labelrotation=30)
    axes[0].legend(fontsize=7);fig.suptitle("Residual high terminal Q is NOT solved; later arrivals are diagnostic probes before admission");fig.savefig(out/"03_failed_terminal_q.png",dpi=160);plt.close(fig)
    # Decode actual draws using the exact private journal ordering from the study.
    project=root.parents[1];asset=project/"models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010/replay/replay_journal.pkl"
    rows=[]
    with asset.open("rb")as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    episode={}
    for row in rows:
        key=(int(row['collection_phase_id']),int(row['episode_id']));x=episode.setdefault(key,dict(hil=False,expert=key[1]<0,outcome=None));x['hil']|=bool(np.isin(row['source_chunk'],[2,3]).any())
        if row['done']:x['outcome']=int(row['success'])
    labels=[]
    for row in rows:
        x=episode[(int(row['collection_phase_id']),int(row['episode_id']))]
        labels.append('expert' if x['expert']else 'assisted success' if x['outcome']==1 and x['hil']else 'autonomous success' if x['outcome']==1 else 'failure' if x['outcome']==0 else 'unknown')
    assert 'unknown'not in labels;labels=np.array(labels)
    ai=np.load(root/'followup/sample_indices.npz');ci_indices=np.load(root/'followup/critic_sample_indices.npz');arr=json.loads((root/'followup/arrival_schedule.json').read_text());sampling=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for name,style in [('baseline','--'),('hil8_clip','-')]:
        for role,indices in [('Actor',ai[name+'_seed42']),('Critic',ci_indices[name+'_seed42'])]:
            xs=[];ys=[]
            for e in arr:
                lo=e['first_update']-1;hi=lo+e['windows'];selected=indices[lo:hi]
                if role=='Actor':selected=selected[(7001+np.arange(lo,hi))%2==0]
                flat=selected.reshape(-1);n=len(flat)
                h=np.array([int(rows[i]['episode_id'])==10009 and bool(np.isin(rows[i]['source_chunk'],[2,3]).any())for i in flat]);pct=100*float(h.mean())if n else None
                item=dict(variant=name,role=role,arrival=e,draws=n,new_HIL_draws=int(h.sum()),new_HIL_pct=pct,composition={tag:float(np.mean(labels[flat]==tag)) for tag in sorted(set(labels))});sampling.append(item);xs.append(hi+7000);ys.append(pct)
            axes[0].plot(xs,ys,style,marker='o',label=name+' '+role)
    axes[0].set(xlabel='End update of admitted Episode',ylabel='Actual draws containing new HIL (%)',title='TRAIN: real HIL exposure per arrival');axes[0].legend(fontsize=8)
    categories=['expert','autonomous success','assisted success','failure'];names=['Replay windows','baseline Actor','HIL8 Actor','Critic']
    pools=[np.arange(len(rows)),ai['baseline_seed42'][1::2].reshape(-1),ai['hil8_clip_seed42'][1::2].reshape(-1),ci_indices['hil8_clip_seed42'].reshape(-1)];bottom=np.zeros(4)
    for tag in categories:
        vals=np.array([np.mean(labels[idx]==tag)*100 for idx in pools]);axes[1].bar(names,vals,bottom=bottom,label=tag);bottom+=vals
    axes[1].set(ylabel='Window/draw proportion (%)',title='TRAIN composition; not fixed success:failure quota');axes[1].tick_params(axis='x',labelrotation=12);axes[1].legend(fontsize=7)
    fig.suptitle('Source labels retained; overlaps resolved by Episode outcome for this plot');fig.savefig(out/'04_actual_sampling.png',dpi=160);plt.close(fig)
    (root/'analysis.json').write_text(json.dumps(dict(contrasts=summary,paired_episode_bootstrap=ci,actual_sampling=sampling,independent_TEST=None,invalid_arm_excluded='followup critic_terminal1 actually combined HIL8; use terminal1-corrected only'),indent=2))
    print(json.dumps(dict(figures=4,summary_rows=len(summary),sampling_rows=len(sampling),bootstrap_unit='whole Episode',independent_test=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();run(a.output)
