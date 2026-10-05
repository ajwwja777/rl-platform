#!/usr/bin/env python3
"""Render retained evidence into a small pre-online handoff report."""
import argparse
import json
from pathlib import Path
import subprocess
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    report=json.loads((out/'critic_targets/target_attribution.json').read_text())
    replay=json.loads((out/'replay_integrity.json').read_text())
    runtime=json.loads((out/'spawned_runtime/report.json').read_text())
    seed=json.loads((out/'seed_budget_validation.json').read_text())
    code=json.loads((out/'code_identity.json').read_text())
    def figsave(fig,name):
        fig.savefig(str(out/(name+'.png')),dpi=180,bbox_inches='tight')
        fig.savefig(str(out/(name+'.svg')),bbox_inches='tight');plt.close(fig)
    cohorts=['train_hil','dev_hil','old_dev_hil']
    labels=['Continued-training pool\n24 HIL Episodes','Warmup development\n2 HIL Episodes; seen at 5k','Reused Online development\n6 HIL Episodes']
    fields=['q_recorded_minus_target','target_minus_observed','full_recorded_minus_actor']
    legends=['Q1(recorded) - TD target','TD target - observed return','Q1(recorded) - Q1(Actor)']
    colors=['#355c7d','#e67e22','#8e44ad']
    fig,ax=plt.subplots(figsize=(11,5.2));x=np.arange(3)
    for j,field in enumerate(fields):
        rows=[report['frozen_summary'][c][field] for c in cohorts]
        y=np.array([r['mean'] for r in rows]);ci=np.array([r['ci95'] for r in rows]);pos=x+(j-1)*.24
        ax.bar(pos,y,width=.22,label=legends[j],color=colors[j])
        ax.errorbar(pos,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='none',color='#222222',capsize=3)
    ax.axhline(0,color='#555555',lw=1);ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylabel('Value difference; equal Episode weight')
    ax.legend(loc='upper left',fontsize=8);ax.set_title('Warmup5k: recorded-action fit and proposal preference are different questions')
    fig.text(.06,.01,'Bands: 95% Episode bootstrap, conditional on inspected cohorts. All Warmup was trained at 5k; no independent test.\nObserved returns include assistance and logical-step discount; they are not optimal or autonomous alternative-action values.',fontsize=9)
    fig.subplots_adjust(bottom=.24);figsave(fig,'target_decomposition')

    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,cohort,label in zip(axes,cohorts,labels):
        for variant,color in [('fixed_bootstrap','#355c7d'),('observed_behavior_return','#e67e22')]:
            runs=[r for r in report['runs'] if r['variant']==variant]
            steps=[0]+[r['updates'] for r in runs[0]['curve']]
            curves=np.array([[report['frozen_summary'][cohort]['full_recorded_minus_actor']['mean']]+[r['summary'][cohort]['recorded_minus_actor']['mean'] for r in run['curve']] for run in runs])
            ax.plot(steps,curves.mean(0),color=color,label=variant)
            ax.fill_between(steps,curves.min(0),curves.max(0),color=color,alpha=.15)
        ax.axhline(0,color='#555555',lw=1);ax.set_title(label,fontsize=10);ax.set_xlabel('Critic-only updates')
    axes[0].set_ylabel('Q1(recorded) - Q1(frozen Actor)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,fontsize=9)
    fig.text(.05,.01,'Actor, target networks, inputs, optimizer initialization and indices fixed per seed; only regression target changes.\nShading: range of 3 continuation seeds, not confidence intervals. No Actor improvement or robot success was measured.',fontsize=9)
    fig.subplots_adjust(top=.75,bottom=.22,wspace=.27);figsave(fig,'target_refit')

    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    vals=[seed['results'][k]['startup_pending_updates'] for k in ['inherit','new_arrivals']]
    axes[0].bar(['Inherited\nold anchor','New-arrival\nbranch anchor'],vals,color=['#c0392b','#27ae60'])
    for i,v in enumerate(vals):axes[0].text(i,v+150,str(v),ha='center')
    axes[0].set_ylim(0,8500);axes[0].set_ylabel('Earned updates at startup');axes[0].set_title('Actual 5k state + synthetic adds_total=4013')
    parts=[runtime['spawned_updates'],runtime['resumed_updates']]
    axes[1].bar(['Before stop','After exact restore'],parts,color=['#355c7d','#27ae60'])
    for i,v in enumerate(parts):axes[1].text(i,v+.3,str(v),ha='center')
    axes[1].set_ylim(0,17);axes[1].set_ylabel('Learner updates');axes[1].set_title('3 synthetic transitions earn 15 updates')
    fig.text(.04,.01,'Real project launcher, CPU network/checkpoint, private loopback RPC and journal. No ROS/Stage1 or robot.\nExisting data stay sampleable; the new branch changes only its update-budget anchor. These are plumbing tests.',fontsize=9)
    fig.subplots_adjust(bottom=.25,wspace=.3);figsave(fig,'update_budget')

    matched=sum(r['exact_adjacent_state_links'] for r in replay['episode_rows'])
    mismatched=sum(len(r['mismatched_adjacent_state_links']) for r in replay['episode_rows'])
    integrity=json.loads((out/'critic_targets/episode_integrity.json').read_text())
    frozen_values=dict(np.load(out/'critic_targets/frozen_values.npz'))
    from methods.openpi_rlt.experiments.target_attribution import episode_interval
    warmup_mask=(frozen_values['phase_id']==1)&frozen_values['hil']
    warmup_pair=episode_interval(frozen_values['qrecord'][:,0]-frozen_values['qactor'][:,0],frozen_values['episode_id'],frozen_values['phase_id'],warmup_mask)
    summary=dict(status='offline plumbing verified; robot/online learning release not granted',code=code,
        stage1='All 134 training Episodes seen; semantic task completion and independent generalization remain insufficient. Reuse prior full-chain report.',
        warmup_q_preference=warmup_pair,new_critic_refit_runs=6,new_actor_candidates=0,
        current_replay=dict(records=replay['records'],episodes=replay['episodes'],sha256=replay['journal_sha256'],
            duplicate_identities=replay['duplicate_identities'],nonfinite_records=replay['nonfinite_records'],
            adjacent_state_links_compared=matched+mismatched,adjacent_state_links_mismatched=mismatched,storage=replay['precision']),
        cached_pool_integrity=dict(windows=integrity['windows'],return_available=integrity['available_observed_returns'],
            duplicate_identities=integrity['phase_episode_step_duplicate_count'],conflicting_overlap_episodes=sum(bool(r['conflicting_overlap_steps']) for r in integrity['episodes']),
            inconsistent_terminal_episodes=sum(not r['terminal_reward_consistent'] for r in integrity['episodes'])),
        tests=dict(project_cpu=383,online_environment_related=34,additional_acceptance_gate=8),runtime=runtime,seed=seed,
        release=dict(offline_candidate_preparation='已验证',fresh_field_no_motion_consistency='证据不足',frozen_robot_acceptance='证据不足',online_autonomous_improvement='证据不足'),
        boundary=['No independent test or robot movement','HIL measured feedback is not verified human command','Critic target-change effect is not proof of optimal action preference','FP32 rollout storage / HIL sampling / branch fixes not deployed','No new Actor promoted; existing Warmup5k is acceptance baseline only'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    head=html.escape(code['head'])
    rows=[
        ('Stage1 实际加载','已验证','134 条全部参加训练；120/14 清单未实现独立留出。原始语义和泛化缺口见前轮审计。'),
        ('Stage1 语义完成与独立泛化','证据不足','结构检查和重建 loss 不证明真实插入完成或 RL Token 适合 Critic。'),
        ('当前 Replay 结构','已验证','4013 条 / 279 Episode；0 重复身份、0 非有限记录；%d 个可对接状态链接均一致。'%matched),
        ('Replay 动作精度','失败','专家 1186 条原本 FP32；rollout 2827 条 FP16。修复覆盖进程启动和序列化，旧误差不能靠 cast 恢复。'),
        ('同步 HIL 时基','失败','原路径逐到达反馈帧记录，未主动限于逻辑20；已加可选 logical20，默认 legacy 保留，真机时序尚待实测。'),
        ('Warmup 动作偏好','已验证','低 Q 现象已出现在 Warmup5k；记录动作贴近 TD 不意味着同状态替代动作排序可信。'),
        ('目标影响 Q 排序','已验证','3 seeds × 2 Critic-only 重拟合；仅改目标后平均排序反转。Actor 未变；行为回报含辅助，不是正确 Q 标签。'),
        ('新在线分支预算','失败','旧起点在合成4013计数下产生7230追赶更新；修复 new_arrivals 起点0，新经验5次更新及恢复已验证。'),
        ('新分支修复结果','已验证','全部参数/优化器/RNG与5k种子逐值一致，仅独立分支预算元数据改变；旧分支不改。'),
        ('真实多进程离线闭环','已验证','实际入口/CPU Actor/Learner/Replay RPC：3模拟经验→15更新，1次退出前+14次恢复后；发布、状态保存与恢复一致。'),
        ('独立自主真机收益','证据不足','没有新独立测试。不能保证 Online 越学越好，不把离线重拟合或测试通过当作放行。'),
    ]
    table=''.join('<tr><td>%s</td><td>%s</td><td>%s</td></tr>'%tuple(html.escape(x) for x in r) for r in rows)
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Online 真机前验证交付</title><style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.7 system-ui;color:#223}h1{font-size:28px}h2{font-size:22px;margin-top:35px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd;text-align:left}code{overflow-wrap:anywhere}a{color:#246}article{padding:14px;background:#eef4f8}small{color:#567}</style><h1>真机 Online RL 前：修复与验证交付</h1><article><b>已修复可确认的数据与启动问题，离线更新闭环已跑通；尚未放行自主 Online RL。</b><p>本轮新增6次 Critic诊断训练，复用此前18次 Actor/Critic 对照。未发布新 Actor；受控冻结验收以现有 Warmup5k 为基线。生产Replay、权重、配置、固定上游及现场均未修改。</p></article><h2>已定位的问题与证据边界</h2><table><tr><th>检查</th><th>结论</th><th>证据与边界</th></tr>'''+table+'''</table><h2>Critic 的目标和动作偏好</h2><p>训练 HIL 的记录动作Q1平均比当前TD目标高约0.011，不是普遍拟合不上目标。TD目标与观测回报的差异在训练和旧开发数据上方向不同。不能把“人类动作Q低”简单归结为标签都低，也不能给未执行 Actor 动作补造真实回报。</p><img src="target_decomposition.png"><img src="target_refit.png"><p>上述训练仅更新Critic；Actor、target网络、初始优化器和每seed采样身份固定。改变回归目标能改变排序，是机制证据；不支持自主能力变好、HIL最优、或直接把MC100部署的结论。</p><h2>启动→更新→保存→恢复</h2><img src="update_budget.png"><p>完整离线闭环使用真实5k checkpoint与网络、独立loopback角色，DummyFeatureProvider/DummyChunkEnv提供合成经验。它验证状态和预算，不验证真实视觉、触觉、现场时延或插入成功。</p><h2>可回退的实现</h2><ul><li>每个spawn子进程重新安装项目补丁，验证FP32动作和真实batch审计有效。</li><li>新建5k分支默认new_arrivals预算起点；显式inherit可复现历史追赶，现有分支不迁移。</li><li>COBOT_RLT_REPLAY_ACTION_PRECISION=float32 为可选新经验精度；撤销需下次启动回到legacy，旧FP16数据不能恢复。</li><li>COBOT_RLT_HIL_SAMPLING=logical20 为同步HIL可选采样；撤销回legacy。它仍记录反馈，不冒充人类命令，也不保证严格物理20Hz。</li><li>Actor版本相同但参数不同拒绝分支；旧快照缺step时只能依靠逐值匹配完整checkpoint确认。</li></ul><h2>验收包与尚未通过的条件</h2><p>独立测试、自主收益、真实HIL命令/反馈对应以及现场deadline/跟踪仍缺证。交付包分无动作一致性、冻结验收、分批学习三个门槛，禁止从前两项测试结果自动放行学习。</p><p><a href="operator_acceptance_prompt.txt">单独转交现场负责人的验收prompt</a> · <a href="acceptance_bundle.json">验收配置与门槛</a> · <a href="summary.json">本轮机器可读总结</a></p><h2>完整过程与复现</h2><p>入口和Git核对→复用历史产物→当前Replay只读结构检查→逐Episode目标归因→单因素Critic重拟合→定位spawn继承和seed预算→隔离修复→真实多进程CPU闭环→完整项目383项与在线环境34项回归→输出小报告/代码差异。未改正式流程MD。</p><p>代码提交 <code>'''+head+'''</code>，分支 audit/q-guidance-20261005；未合并/部署。实验启动的dirty source SHA在launch收据中，不冒称使用后来提交的代码。</p><p><a href="critic_targets/target_attribution.json">6次训练、实际配置、输入SHA和每Episode指标</a> · <a href="critic_targets/episode_integrity.json">窗口重叠和terminal审计</a> · <a href="replay_integrity.json">当前生产Replay只读收据</a> · <a href="seed_budget_validation.json">真实种子预算验证</a> · <a href="spawned_runtime/report.json">真实多进程验证</a> · <a href="full_regression_final.log">完整CPU回归</a> · <a href="online_environment_tests.log">在线Python版本回归</a></p><p><a href="../full-chain-audit-20261001/index.html">Stage1/Warmup/Online完整追溯及论文借鉴</a> · <a href="../model-repair-20261005/index.html">前轮18次训练和6张动作/质量图</a></p><small>缺失证据留空。训练、反复选型开发和独立测试分开；没有独立测试。区间按Episode计算；重叠窗口不是独立样本。</small></html>'''
    (out/'index.html').write_text(text,encoding='utf-8')
    mc_path=out/'spawned_runtime_mc30/report.json'
    if mc_path.exists():
        summary['mc30_spawned_fixture']=json.loads(mc_path.read_text())
        summary['mc30_fixture_boundary']='Same Warmup5k fixture used to test MC30 role initialization and restore; not the actual historical MC30 initialization or field model.'
        (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        text=text.replace('完整CPU回归</a>','完整CPU回归</a>')
        text=text.replace('<h2>可回退的实现</h2>','<p>MC30入口也通过真实多进程闭环：spawn后的实际MC权重0.3已核对，精度、batch审计、退出与恢复通过。这里使用同一Warmup5k状态作为入口测试fixture，不冒称核实了历史MC30的实际初始化。</p><p><a href="spawned_runtime_mc30/report.json">MC30多进程入口验证</a> · <a href="acceptance_gate_tests.log">额外8项完整Episode门槛测试</a></p><h2>可回退的实现</h2>')
        (out/'index.html').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
