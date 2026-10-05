#!/usr/bin/env python3
"""Render evidence-supported selection, never substitute proxies for robot success."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def read(name):return json.loads((out/name).read_text())
    def save(name,fig):
        fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=170);fig.savefig(figdir/(name+'.svg'));plt.close(fig)
    stage=read('stage1_full/report.json');rows=stage['episodes'];complete=stage.get('complete',False)
    n=len(rows);picks=np.random.default_rng(42).integers(n,size=(10000,n))
    metrics=['sample30_first10','legacy_first10_at20','retimed30to20_first10','hold_current_state_at20']
    labels=['Model at demonstration 30Hz','Legacy samples executed at 20Hz','Retimed 30->20Hz proposal','Hold current state at 20Hz']
    colors=['#24547A','#B03A2E','#17815A','#888888']
    stage_summary={}
    for metric in metrics:
        values=np.asarray([r[metric]['mae_per_dim'] for r in rows]);boots=values[picks].mean(1)
        stage_summary[metric]={'joint_mae':float(values[:,:6].mean()),'gripper_mae_m':float(values[:,-1].mean()),
            'mae_per_dim':values.mean(0).tolist(),'joint_ci95':np.quantile(boots[:,:6].mean(1),[.025,.975]).tolist() if complete else None}
    legacy=np.asarray([r[metrics[1]]['mae_per_dim'] for r in rows]);retimed=np.asarray([r[metrics[2]]['mae_per_dim'] for r in rows]);delta=retimed-legacy
    stage_summary['paired_retiming']={'episodes':n,'joint_delta':float(delta[:,:6].mean()),
        'joint_delta_ci95':np.quantile(delta[picks][:,:,:6].mean((1,2)),[.025,.975]).tolist() if complete else None,
        'gripper_delta':float(delta[:,-1].mean()),'gripper_delta_ci95':np.quantile(delta[picks][:,:,-1].mean(1),[.025,.975]).tolist() if complete else None}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for metric,label,color in zip(metrics,labels,colors):
        values=np.asarray([r[metric]['mae_per_dim'] for r in rows]);axes[0].plot([r['episode'] for r in rows],values[:,:6].mean(1)*1000,'.-',color=color,label=label,lw=.7,ms=2)
        axes[1].plot([r['episode'] for r in rows],values[:,-1]*1000,'.-',color=color,label=label,lw=.7,ms=2)
    axes[0].set_ylabel('Six-joint mean absolute error (mrad)');axes[1].set_ylabel('Gripper mean absolute error (mm)')
    for ax in axes:ax.set_xlabel('Complete Episode index');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle(f'Stage1: {n}/134 training-seen Episodes; no independent test; valid future slots only')
    save('stage1_timebase',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for metric,label,color in zip(metrics[:3],labels[:3],colors[:3]):
        v=np.asarray([r[metric]['horizon_mae_per_dim'] for r in rows])
        axes[0].plot(np.arange(10),v[:,:,:6].mean((0,2))*1000,color=color,label=label)
        axes[1].plot(np.arange(10),v[:,:,-1].mean(0)*1000,color=color)
    for ax in axes:ax.set_xlabel('Action index within C10');ax.grid(alpha=.2)
    axes[0].set_ylabel('Joint MAE (mrad)');axes[1].set_ylabel('Gripper MAE (mm)');axes[0].legend(fontsize=7)
    fig.suptitle('Equal complete-Episode means; interpolation is a timing proposal, not robot execution')
    save('stage1_horizon_error',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,(metric,label,color) in enumerate(zip(metrics[:3],labels[:3],colors[:3])):
        values=np.asarray([r[metric]['mae_per_dim'] for r in rows]);mean=values.mean(0)
        ci=np.quantile(values[picks].mean(1),[.025,.975],axis=0) if complete else np.stack([mean,mean])
        axes[0].bar(np.arange(6)+(i-1)*.25,mean[:6]*1000,.25,color=color,label=label,
                    yerr=np.maximum(0,np.stack([mean[:6]-ci[0,:6],ci[1,:6]-mean[:6]]))*1000,capsize=2)
        axes[1].bar(i,mean[-1]*1000,color=color,
                    yerr=np.maximum(0,np.asarray([[mean[-1]-ci[0,-1]],[ci[1,-1]-mean[-1]]]))*1000,capsize=3)
    axes[0].set_xticks(range(6));axes[0].set_xticklabels(['J1','J2','J3','J4','J5','J6']);axes[0].set_ylabel('Joint MAE (mrad)');axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(3));axes[1].set_xticklabels(['native30','legacy20','retimed20']);axes[1].set_ylabel('Gripper MAE (mm)')
    fig.suptitle('134 complete training-seen Episodes; equal Episode means and 95% Episode bootstrap CI')
    save('stage1_joint_gripper_error',fig)
    comparison=read('candidate_comparison/comparison.json');full=[r for r in comparison['models'] if r['label'].startswith('full_pool/')]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for i,r in enumerate(full):
        value=r['paired_vs_warmup5k'];d=value['joint_mean_delta']*1e6;lo,hi=np.asarray(value['joint_ci95'])*1e6
        axes[0].errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color='#24547A',capsize=3)
    axes[0].set_yticks(range(len(full)));axes[0].set_yticklabels([r['label'].split('/')[-1] for r in full],fontsize=8)
    axes[0].axvline(0,color='black',lw=.8);axes[0].set_xlabel('Joint MAE delta vs Warmup5k (microrad); lower is better')
    values=np.asarray([r['paired_vs_warmup5k']['per_dim_mean_delta'] for r in full]);values[:,:6]*=1e6;values[:,-1]*=1e6
    heat=axes[1].imshow(values,aspect='auto',cmap='RdBu_r',vmin=-200,vmax=200);axes[1].set_yticks([])
    axes[1].set_xticks(range(7));axes[1].set_xticklabels(['J1','J2','J3','J4','J5','J6','Grip'])
    axes[1].set_title('Paired per-dimension MAE delta\nJoints: microrad; gripper: micrometre',fontsize=9);fig.colorbar(heat,ax=axes[1],shrink=.7)
    fig.suptitle('20 reused rollout-development Episodes / 6 HIL Episodes; full-Episode CI; stored targets')
    save('candidate_development',fig)
    dropout=read('reference_dropout/report.json');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for probability,color in [(.5,'#24547A'),(0.,'#B03A2E')]:
        runs=[r for r in dropout['runs'] if r['dropout']==probability]
        for dimension,ax in enumerate(axes):
            vals=[]
            for run in runs:
                vals.append([np.mean([np.mean(r['human_mae_per_dim'][:6]) if dimension==0 else r['human_mae_per_dim'][-1]
                    for r in row['evaluation']['development'] if r['hil_steps']]) for row in run['curve']])
            vals=np.asarray(vals)*1000;x=np.asarray([r['updates'] for r in runs[0]['curve']])+5000
            ax.plot(x,vals.mean(0),'o-',color=color,label='reference dropout '+str(probability));ax.fill_between(x,vals.min(0),vals.max(0),alpha=.15,color=color)
    for dimension,ax in enumerate(axes):
        baseline=np.mean([np.mean(r['human_mae_per_dim'][:6]) if dimension==0 else r['human_mae_per_dim'][-1]
            for r in dropout['baseline']['development'] if r['hil_steps']])*1000
        ax.axhline(baseline,color='black',linestyle='--',label='Initial Warmup5k');ax.set_xlabel('Actual Critic update count');ax.grid(alpha=.2)
    axes[0].set_ylabel('HIL joint MAE (mrad)');axes[1].set_ylabel('HIL gripper MAE (mm)');axes[0].legend(fontsize=8)
    fig.suptitle('6 HIL Episodes in reused 20-Episode development; shading = 3 continuation-seed range')
    save('reference_dropout',fig)
    token=read('token_readout/report.json');fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=[r['name'].replace('_','\n') for r in token['models']]
    means=np.asarray([r['episode_mae_mean'] for r in token['models']]);ci=np.asarray([r['episode_mae_ci95'] for r in token['models']])
    axes[0].bar(range(3),means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),color=['#888888','#24547A','#17815A'],capsize=4)
    axes[1].bar(range(3),[r['first_anchor_success_failure_auc'] for r in token['models']],color=['#888888','#24547A','#17815A']);axes[1].axhline(.5,color='black',linestyle='--')
    for ax in axes:ax.set_xticks(range(3));ax.set_xticklabels(names,fontsize=8)
    axes[0].set_ylabel('Observed-behavior return MAE; Episode CI');axes[1].set_ylabel('First-anchor success/failure AUC')
    fig.suptitle('Train-only PCA/ridge; 20 reused development Episodes; assisted return is not autonomous value')
    save('token_information',fig)
    adaptation=read('actor_time_adaptation/report.json');fig,axes=plt.subplots(1,2,figsize=(11,4))
    adaptation_summary=[]
    for hz,color in [(30,'#24547A'),(20,'#17815A')]:
        runs=[r for r in adaptation['runs'] if r['reference_hz']==hz]
        for dim,ax in enumerate(axes):
            values=np.asarray([[np.mean([np.mean(e['mae_per_dim'][:6]) if dim==0 else e['mae_per_dim'][-1]
                for e in point['evaluation']['development']]) for point in run['curve']] for run in runs])*1000
            x=np.asarray([r['actor_updates'] for r in runs[0]['curve']])+2500
            ax.plot(x,values.mean(0),'o-',color=color,label='reference '+str(hz)+'Hz');ax.fill_between(x,values.min(0),values.max(0),color=color,alpha=.15)
        for run in runs:
            v=np.asarray([e['mae_per_dim'] for e in run['evaluation']['development']]);adaptation_summary.append({'seed':run['seed'],'reference_hz':hz,'joint_mae':float(v[:,:6].mean()),'gripper_mae_m':float(v[:,-1].mean()),'critic_unchanged':run['critic_state_unchanged']})
    for dim,ax in enumerate(axes):
        base=np.asarray([e['mae_per_dim'] for e in adaptation['baseline']['30']['development']]);ref=np.asarray([e['reference_mae_per_dim'] for e in adaptation['baseline']['20']['development']])
        ax.axhline((base[:,:6].mean() if dim==0 else base[:,-1].mean())*1000,color='black',ls='--',label='Initial legacy5k Actor')
        ax.axhline((ref[:,:6].mean() if dim==0 else ref[:,-1].mean())*1000,color='#17815A',ls=':',label='Retimed reference only')
        ax.set_xlabel('Actual Actor update count (Critic remains at 5000)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Joint MAE (mrad)');axes[1].set_ylabel('Gripper MAE (mm)');axes[0].legend(fontsize=7)
    fig.suptitle('Expert-only Actor/Q=0 adaptation; 14 reused development Episodes, all Stage1-seen; 3 seed range')
    save('actor_time_adaptation',fig)
    summary={'stage1_complete':complete,'stage1_episodes':n,'stage1_frames':sum(r['frames'] for r in rows),
        'stage1':stage_summary,'new_dropout_runs':len(dropout['runs']),'retained_candidates_compared':len(comparison['models']),
        'staged_runtime':read('staged_publication/report.json'),'new_actor_promoted':False,
        'best_supported_initial_actor':'Existing immutable Warmup5k; no tested continuation established joint/gripper development dominance.',
        'online_autonomous_improvement':'证据不足','robot_release':False}
    summary['actor_only_time_adaptation']=adaptation_summary
    summary['actual_service_independent_restore']=read('service_restore_comparison.json')
    wrong=read('wrong_prompt_training_join.json')
    summary['wrong_task_training_membership']={'episodes':[3,7],'records':sum(len(r['matched_windows']) for r in wrong),
        'evidence':'wrong_prompt_training_join.json','boundary':'Exact action/source/start+next-state join to recorded wrong prompt; historical VLA request receipt missing. Not the proven sole cause.'}
    if (out/'warmup_prompt_filter/report.json').exists():
        prompt_filter=read('warmup_prompt_filter/report.json');summary['fresh_native_warmup_prompt_filter']={'complete':prompt_filter.get('sources_unchanged',False),'runs':len(prompt_filter['runs']),'source_report':'warmup_prompt_filter/report.json'}
        if len(prompt_filter['runs'])==6:
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for condition,color in [('all','#24547A'),('exclude_wrong_task','#17815A')]:
                runs=[r for r in prompt_filter['runs'] if r['condition']==condition]
                for dim,ax in enumerate(axes):
                    values=np.asarray([[np.mean([np.mean(e['human_mae_per_dim'][:6]) if dim==0 else e['human_mae_per_dim'][-1] for e in point['evaluation']['development'] if e['hil_steps']]) for point in run['curve']] for run in runs])*1000
                    x=np.asarray([r['updates'] for r in runs[0]['curve']]);ax.plot(x,values.mean(0),'o-',color=color,label=condition);ax.fill_between(x,values.min(0),values.max(0),color=color,alpha=.15)
            for dim,ax in enumerate(axes):
                values=[np.mean(e['human_mae_per_dim'][:6]) if dim==0 else e['human_mae_per_dim'][-1] for e in prompt_filter['baseline']['development'] if e['hil_steps']]
                ax.axhline(np.mean(values)*1000,color='black',ls='--',label='Archived legacy5k');ax.set_xlabel('Actual Critic update count; Actor updates half');ax.grid(alpha=.2)
            axes[0].set_ylabel('HIL joint MAE (mrad)');axes[1].set_ylabel('HIL gripper MAE (mm)');axes[0].legend(fontsize=8)
            fig.suptitle('Cold native Warmup5k: exclude 29 wrong-task records only; 6 HIL/reused 20 dev Episodes; 3 seed range')
            save('warmup_prompt_filter',fig)
    if (out/'reference_timebase/report.json').exists():
        factor=read('reference_timebase/report.json');summary['prompt_time_comparison']=factor['summaries']
        fig,axes=plt.subplots(1,2,figsize=(11,4));names=[]
        for model,color in [('reference','#24547A'),('actor','#B03A2E')]:
            rows=[r for r in factor['summaries'] if r['model']==model];x=np.arange(len(rows))+(-.15 if model=='reference' else .15)
            axes[0].bar(x,[r['joint_mae']*1000 for r in rows],width=.3,color=color,label=model)
            axes[1].bar(x,[r['mae_per_dim'][-1]*1000 for r in rows],width=.3,color=color,label=model)
            names=[r['prompt'].replace('_prompt','')+'\n'+r['timing'] for r in rows]
        for ax in axes:ax.set_xticks(range(4));ax.set_xticklabels(names,fontsize=8)
        axes[0].set_ylabel('Joint MAE (mrad)');axes[1].set_ylabel('Gripper MAE (mm)');axes[0].legend()
        fig.suptitle('Language/time-base factorial: same frozen Actor, 28 complete training-seen Episodes')
        save('prompt_timebase_actor',fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    if (out/'expert_time_contract/report.json').exists():
        expert=read('expert_time_contract/report.json')
        summary['expert_time_contract']={'status':expert['status'],'episodes':expert['expert_episodes'],
            'transitions':expert['expert_transitions'],'reconstruction_max_error':max(r['reconstruction_max_error'] for r in expert['episodes']),
            'actual_target_first_to_last_seconds_mean':float(np.mean([r['actual_c10_first_to_last_seconds_mean'] for r in expert['episodes']])),
            'legacy_reference_first_to_last_seconds':.3,'required_reference_first_to_last_seconds':.45,
            'source_report':'expert_time_contract/report.json'}
    summary['input_contract_release']={'existing_actor_expected_reference_hz':30,'execute_logical_hz':20,
        'retimed_reference_with_existing_actor':'失败: matched 28-Episode action fit worsened; do not silently mix',
        'new_actor_promoted':False,'consistent_20hz_rematerialization_and_retraining':'not performed; old rollout C10 cannot supply horizon13.5'}
    for label in ['reference_contract','reference_contract_batch1','reference_contract_shared','reference_contract_service_numeric']:
        if (out/(label+'/report.json')).exists():
            s=read(label+'/report.json');summary[label]={'status':s['status'],'rows':s['rows'],
                'source_report':label+'/report.json','boundary':s['boundary']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    figures=['stage1_timebase','stage1_horizon_error','stage1_joint_gripper_error','candidate_development','reference_dropout','token_information','actor_time_adaptation']
    if (figdir/'warmup_prompt_filter.png').exists():figures.append('warmup_prompt_filter')
    if (figdir/'prompt_timebase_actor.png').exists():figures.append('prompt_timebase_actor')
    html='''<!doctype html><meta charset="utf-8"><title>Offline model selection</title><style>body{max-width:1100px;margin:32px auto;font:16px/1.6 sans-serif;color:#203040}img{width:100%}pre{white-space:pre-wrap;background:#f2f5f8;padding:16px}h2{margin-top:40px}</style><h1>真机 Online 前：模型验证与候选选择</h1><p>完整 Episode、真实 checkpoint、只读数据；没有机器人动作或独立真机成功率。此页持续保存进度，最终状态以 progress.json 为准。</p><p><b>结论边界：</b>已验证离线运行与数据合同；新训练模型尚未建立稳定优势，不以低 TD loss、Actor Q 上升或拟合最小值放行。当前保留 Warmup5k 作初始 Actor，分离训练候选与正在执行的 Actor。</p>'''
    html+='<p><b>新的关键问题：</b>Stage1 的 30 Hz 输出与 Warmup 专家 20 Hz 动作目标不在同一时基。参考轨迹重采样改善训练内拟合，但现有 Actor 已适应旧参考输入；单独换成 20 Hz 参考使同一 Actor 的关节误差增加约 16%，该组合被拒绝。旧 Actor 仍配旧合同。训练、反复选型开发集和独立测试不可混用；本次独立测试为空。</p>'
    html+='<p><b>任务数据：</b>两个早期插入 Episode 记录的是锅盖任务提示词，29 条 transition 已精确追到 Warmup 训练清单。已修复插入环境入口的静默错误任务默认，排除训练对照使用独立副本。大多数历史插入轨迹提示词正确，不能把这项问题概括为整个 Online 的根因。</p>'
    html+='<p><b>数值验证：</b>3 个固定真实输入、两个独立恢复进程的实际服务动作和 Token 完全一致；训练加载器批处理缓存与服务的严格数值阈值仍有失败，全部保留。前者不替后者补证，也不证明机器人动作与成功率。</p>'
    html+='<p><b>三阶段与历史修复：</b> <a href="../full-chain-audit-20261001/index.html">Stage1 / Warmup / Online 原始审计与论文</a> · <a href="../preonline-verification-20261005/index.html">Critic target、真实多进程、Replay 精度与预算</a> · <a href="operator_acceptance_prompt.txt">现场负责人验收 Prompt</a> · <a href="delivery.json">最终交付与优先级</a> · <a href="progress.json">恢复进度</a></p>'
    for name in figures:html+='<h2>'+name+'</h2><img src="figures/'+name+'.png">'
    html+='<h2>完整可追溯结果</h2><pre>'+json.dumps(summary,ensure_ascii=False,indent=2)+'</pre>'
    (out/'index.html').write_text(html,encoding='utf-8')


if __name__=='__main__':main()
