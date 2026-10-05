"""Scientific figures and evidence report; uncertainty unit is complete Episode."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/model-repair-20261005')
figdir=out/'figures';figdir.mkdir(exist_ok=True)
frozen=json.loads((out/'frozen_diagnosis.json').read_text())
fixed=json.loads((out/'full_pool/fixed_comparison.json').read_text())
training=json.loads((out/'full_pool/training_comparison.json').read_text())
partial=json.loads((out/'fixed_comparison.json').read_text())
grad=json.loads((out/'actor_gradient_diagnosis.json').read_text())
data=dict(np.load(out/'precision_inputs.npz'));values=dict(np.load(out/'frozen_values.npz'))
names=['legacy_fp16','raw_fp32','raw_fp32_qoff'];colors=['#586f7c','#e59a35','#207a70']
labels=['Stored FP16','Recovered FP32','FP32 + Actor Q off']

def savefig(name):
    plt.tight_layout(rect=(0,0,1,.89) if getattr(plt.gcf(),'_suptitle',None) else (0,0,1,1))
    plt.savefig(figdir/(name+'.png'),dpi=160);plt.savefig(figdir/(name+'.svg'));plt.close()
def ep_errors(rows):
    return np.asarray([e['human_per_dimension_mae'] for e in rows if e['human_per_dimension_mae'] is not None])
def mean_joint(rows):return float(ep_errors(rows)[:,:6].mean())
def paired_ci(a,b):
    delta=ep_errors(b)[:,:6].mean(1)-ep_errors(a)[:,:6].mean(1)
    if len(delta)<2:return None
    boot=np.random.default_rng(42).choice(delta,(10000,len(delta)),replace=True).mean(1)
    return {'episodes':len(delta),'mean_rad':float(delta.mean()),'episode_bootstrap_ci95_rad':np.quantile(boot,[.025,.975]).tolist()}

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
rows=[e for e in frozen['episodes'] if e['assisted'] and e['raw_verified']]
axes[0].bar(range(len(rows)),[e['human_endpoint_q1_change'] for e in rows],color=['#207a70' if e['phase']=='warmup' else '#8c54a2' for e in rows])
axes[0].set_xticks(range(len(rows)));axes[0].set_xticklabels([str(e['episode_id']) for e in rows],rotation=60)
axes[0].axhline(0,color='#444444',lw=.7);axes[0].set_ylabel('Q1(HIL-slot replacement) - Q1(Actor)')
axes[0].set_title('Verified FP32: 12 Warmup train + 1 Online dev\nAll assisted success; no independent test')
old=[e for e in frozen['episodes'] if e['assisted'] and not e['raw_verified']]
axes[1].bar(range(len(old)),[e['q_gradient_toward_human'] for e in old],color='#586f7c')
axes[1].set_xticks(range(len(old)));axes[1].set_xticklabels([str(e['episode_id']) for e in old])
axes[1].axhline(0,color='#444444',lw=.7);axes[1].set_ylabel('grad Q1(Actor) dot direction to HIL')
axes[1].set_title('Six repeatedly selected development Episodes\nStored FP16 targets; not independent test')
savefig('q_direction')

idx=np.flatnonzero(data['phase_online'] & (data['episode_id']==184));x=data['step_id'][idx]
plt.figure(figsize=(10,4.5))
for a,label,color in [(values['td_target'][idx],'TD target for recorded action','#222222'),(values['q_legacy'][idx,0],'Q1 recorded full chunk','#586f7c'),(values['q_actor'][idx,0],'Q1 offline Actor proposal','#e59a35'),(values['q_endpoint'][idx,0],'Q1 HIL-slot replacement','#207a70')]:plt.plot(x,a,label=label,color=color)
plt.xlabel('Recorded logical step index (no semantic stage label)');plt.ylabel('Value')
plt.title('Episode 184, assisted success / reused development\nWarmup5k checkpoint; retained target networks + specified RNG draw')
plt.legend(fontsize=8);plt.text(.02,.02,'The TD target belongs to the recorded action only; counterfactual successors are unobserved.',transform=plt.gca().transAxes,fontsize=8)
savefig('td_and_proposals')

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,split,title in zip(axes,['development','historical_online_development'],['Two raw-verified HIL Episodes\nPreviously trained Warmup development','Six HIL Episodes, stored FP16 targets\nRepeatedly used Online development']):
    for name,color,label in zip(names,colors,labels):
        runs=[r for r in training['runs'] if r['variant']==name]
        curves=np.asarray([[mean_joint(training['baseline'][split])]+[mean_joint(point['evaluation'][split]) for point in r['curve']] for r in runs])*1000
        steps=[0]+[p['updates'] for p in runs[0]['curve']]
        ax.plot(steps,curves.mean(0),label=label,color=color)
        ax.fill_between(steps,curves.min(0),curves.max(0),color=color,alpha=.15)
    ax.set_title(title);ax.set_xlabel('Additional learner updates from Warmup5k');ax.set_ylabel('Episode-equal joint MAE (mrad)')
axes[1].legend(fontsize=8);fig.suptitle('Full Warmup pool retained; bands are ranges across three training seeds, not confidence intervals',fontsize=10)
savefig('learning_curves')

fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(7);width=.24
for ax,split,title in zip(axes,['development','historical_online_development'],['Raw-verified HIL development (2 Episodes)','Repeated Online development (6 Episodes; FP16 targets)']):
    for j,(name,color,label) in enumerate(zip(names,colors,labels)):
        errors=np.asarray([ep_errors(r['evaluation'][split]).mean(0) for r in fixed['variants'] if r['variant']==name])*1000
        ax.bar(x+(j-1)*width,errors.mean(0),width,color=color,label=label)
    ax.set_xticks(x);ax.set_xticklabels(['J1','J2','J3','J4','J5','J6','Grip'])
    ax.set_ylabel('Joints: mrad; gripper: mm');ax.set_title(title,fontsize=10)
axes[1].legend(fontsize=8);savefig('per_dimension_errors')

plt.figure(figsize=(10,4.5))
base=np.mean([np.mean(e['human_per_dimension_mae'][:6]) for e in fixed['initial']['retention'] if e['expert']])*1000
for j,(name,color,label) in enumerate(zip(names,colors,labels)):
    y=[]
    for d in [partial,fixed]:
        y.append([np.mean([np.mean(e['human_per_dimension_mae'][:6]) for e in r['retention'] if e['expert']])*1000 for r in d['variants'] if r['variant']==name])
    y=np.asarray(y)
    plt.bar(np.arange(2)+(j-1)*width,y.mean(1),width,color=color,label=label)
plt.axhline(base,color='#222222',ls='--',label='Initial Warmup5k')
plt.xticks([0,1],['Selective traceable pool (80 Episodes)','Full Warmup retained + selective Online'])
plt.ylabel('Episode-equal expert joint MAE (mrad)');plt.title('120 previously trained expert Episodes: fitting retention, not independent generalization')
plt.legend(fontsize=8);savefig('retention')

comparisons=[]
for seed in [41,42,43]:
    runs={r['variant']:r for r in fixed['variants'] if r['seed']==seed}
    h={r['variant']:r['indices_sha256'] for r in training['runs'] if r['seed']==seed}
    assert len(set(h.values()))==1
    for a,bname in [('legacy_fp16','raw_fp32'),('raw_fp32','raw_fp32_qoff')]:
        comparisons.append({'seed':seed,'reference':a,'candidate':bname,
          'development':paired_ci(runs[a]['evaluation']['development'],runs[bname]['evaluation']['development']),
          'historical_online_development':paired_ci(runs[a]['evaluation']['historical_online_development'],runs[bname]['evaluation']['historical_online_development'])})

# Regenerate the exact sampled indices from the frozen identity and RNG recipe.
import pickle
pool=json.loads((out/'full_pool/pool_identity.json').read_text())
rs=[]
with Path(pool['warmup_path']).open('rb') as f:
    while True:
        try:rs.append(pickle.load(f))
        except EOFError:break
rows=[dict(phase_online=False,episode_id=int(r['episode_id']),step_id=int(r['step_id']),expert=int(r['episode_id'])>=100000 or int(r['episode_id'])<0,
           source_chunk=np.asarray(r['source_chunk']),success=bool(r['success'])) for r in rs]
for i in np.flatnonzero(data['phase_online']):rows.append(dict(phase_online=True,episode_id=int(data['episode_id'][i]),step_id=int(data['step_id'][i]),expert=False,
           source_chunk=data['source_chunk'][i],success=bool(data['success'][i])))
trainkeys=set(map(tuple,training['train_episodes']));trainids=np.asarray([i for k in sorted(trainkeys) for i,r in enumerate(rows) if (r['phase_online'],r['episode_id'])==k])
counts=np.zeros(len(rows),np.int64)
for seed in [41,42,43]:
    sampled=np.random.default_rng(seed).choice(trainids,(training['updates'],128),replace=True)
    expected=next(r['indices_sha256'] for r in training['runs'] if r['seed']==seed)
    assert hashlib.sha256(sampled.tobytes()).hexdigest()==expected
    np.add.at(counts,sampled.ravel(),1)
epmeta={}
for r in rows:
    key=(r['phase_online'],r['episode_id'])
    m=epmeta.setdefault(key,{'expert':r['expert'],'hil':False,'success':False})
    if not r['expert']:m['hil']|=bool(np.isin(r['source_chunk'],[2,3]).any())
    m['success']|=r['success']
sampling=[]
for label,predicate in [('Expert',lambda r:r['expert']),('Rollout with HIL',lambda r:not r['expert'] and epmeta[(r['phase_online'],r['episode_id'])]['hil']),('Rollout without HIL',lambda r:not r['expert'] and not epmeta[(r['phase_online'],r['episode_id'])]['hil'])]:
    ids=[i for i in trainids if predicate(rows[i])]
    sampling.append({'label':label,'episodes':len({(rows[i]['phase_online'],rows[i]['episode_id']) for i in ids}),
       'transitions':len(ids),'sampled_slots':int(counts[ids].sum()),'actual_batch_ratio':float(counts[ids].sum()/counts.sum())})
plt.figure(figsize=(9,4));x=np.arange(3)
for j,(key,label) in enumerate([('episodes','Episode fraction'),('transitions','Transition fraction'),('sampled_slots','Actual sampled fraction')]):
    v=np.asarray([r[key] for r in sampling],float);plt.bar(x+(j-1)*.24,v/v.sum(),.24,label=label)
plt.xticks(x,[r['label'] for r in sampling]);plt.ylabel('Fraction');plt.title('Full-pool comparison: actual sampling recovered and hash verified\nHIL excludes experts here; success/assistance overlap is reported separately')
plt.legend(fontsize=8);savefig('sampling')

summary={'status':'completed isolated repairs and 18 matched offline runs; no candidate released',
 'frozen':{k:frozen[k] for k in ['checkpoint_sha256','raw_precision_q1_effect','raw_hil_endpoint_q1_change','old_six_hil_endpoint_q1_change']},
 'gradients':grad,'paired_comparisons':comparisons,'actual_sampling':sampling,
 'initial_metrics':{s:mean_joint(fixed['initial']['evaluation'][s]) for s in ['development','historical_online_development']},
 'final_metrics':[{'variant':r['variant'],'seed':r['seed'],**{s:mean_joint(r['evaluation'][s]) for s in ['development','historical_online_development']},
  'expert_mae_rad':float(np.mean([np.mean(e['human_per_dimension_mae'][:6]) for e in r['retention'] if e['expert']]))} for r in fixed['variants']],
 'limits':['No independent test, robot motion or success measurements.','All 120 expert retention Episodes were previously trained.','Only two raw-verified HIL development Episodes; evidence is weak.','Six Online development Episodes were repeatedly used for selection; original FP32 HIL actions and contemporaneous Actor proposals unavailable.',
 'Full-pool comparison preserves all original Warmup experts; only 10 selective Online Episodes are available, so this is not the production Online distribution.',
 'All three seeds resume one shared Warmup checkpoint, measuring continuation RNG variability only.','Measured HIL feedback is not established to be the human command or optimal counterfactual.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
table=''.join('<tr><td>{variant}</td><td>{seed}</td><td>{development:.6f}</td><td>{historical_online_development:.6f}</td><td>{expert_mae_rad:.6f}</td></tr>'.format(**r) for r in summary['final_metrics'])
html='''<!doctype html><meta charset="utf-8"><title>模型问题定位与隔离修复</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:36px auto;color:#243444}h1,h2{line-height:1.35}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5da;padding:8px}code{overflow-wrap:anywhere}aside{background:#edf5f3;padding:18px}small{color:#5e6b74}</style>
<h1>问题已定位到 Warmup 的 Q 引导与数据契约；真机主因仍未证实</h1>
<aside>已验证：Warmup5k 已存在 Q 与所记录人类动作拟合方向的冲突，Online 之前就有。精度恢复无法解释大部分 Q 偏好差距。<br>已修复：Replay 动作精度可选补丁、Episode trace 串写、暂停后终局与 UUID 补记、终局身份先于通知、动作语义/计划提议/单调时钟记录。<br>训练结果：18 次隔离对照。关闭 Actor Q 项对旧 Online 开发集动作拟合有约 1% 增量收益，但另一组 HIL 开发数据仍比初始模型差。未选定可放行 Actor。</aside>
<h2>一路怎么排查</h2><ol><li>先追溯原始 trace → Replay → delta → 归一化。证实 float16 绝对动作减 float32 状态造成虚假 delta；不把数值问题直接归因为真机失败。</li><li>仅纳入原始动作、source、起始状态和 next-state 都精确匹配、无歧义且完整覆盖的 Episode。共 80 个 Episode / 1,276 个窗口；另六个历史开发 Episode 缺原始 trace，仅保留存储精度分析。</li><li>找到与现场哈希相同的 Warmup5k 完整 checkpoint，比较固定同状态下存储动作、原始动作、离线 Actor 提议与 HIL 槽替换。13 个原始可核实 HIL Episode 的 ΔQ1 均值 −0.02913，Episode bootstrap 95% 区间 [−0.04138, −0.01764]。这些均为辅助成功，不能称为自主成功证据。</li><li>核对 retained target 网络与 RNG 下的诊断 TD 目标。已执行动作的 Q 接近约 0.3–0.5 的 bootstrap 目标，成功终局目标约 0.9135；未执行 Actor 提议没有真实后继和直接值标签。TD 拟合不能验证反事实动作排序。</li><li>按实际 Actor loss 分解梯度：Q 与 BC 参数梯度余弦 −0.282；Q 梯度范数约为 BC 的 7.04%，原生单位 delta 项仅 0.0145%。这是一次固定数据/RNG 检查，不能推断所有训练步。</li><li>分别验证精度和 Q 项。每个因素使用相同初始化、优化器、采样序列、RNG、2,000 learner 更新 / 1,000 Actor 更新，三种子 41/42/43。保留参数、完整状态、实际配置、日志、指标和采样 hash。</li><li>局部数据训练使专家拟合退化，于是增加完整 Warmup 池对照：2,567 个 Warmup 窗口 + 10 个可追溯 Online Episode；六个旧开发 Episode 完全排除本轮训练。202 个训练 Episode、12 个开发 Episode。开发数据仍不是独立测试。</li></ol>
<h2>证据表</h2><table><tr><th>检查层</th><th>结论</th><th>直接证据与边界</th></tr>
<tr><td>原始动作进入 Replay 的数值保真</td><td>失败；隔离补丁通过验证</td><td>Episode163 step50：原始 J3 delta=0，序列化后归一化误差 −0.11207。补丁保留 FP32 action，参考/特征仍沿用原格式；旧半精度数据不可凭空恢复。</td></tr>
<tr><td>HIL 动作、时基、Episode 身份</td><td>证据不足；记录工具缺陷已修复</td><td>HIL action 是反馈而非独立人类命令；来源 ROS 时间不等于发布单调时钟。旧 trace 缺失不能补造，补丁只改善未来记录。</td></tr>
<tr><td>训练目标与实际更新</td><td>已验证（指定 checkpoint/本轮对照）</td><td>原生 TD、Q1 Actor 目标与优化器恢复已核对，样本序列三因素 hash 完全一致。历史真实 batch 身份仍不可恢复。</td></tr>
<tr><td>Critic 能指导纠正动作</td><td>证据不足，存在相反方向证据</td><td>Q 端点与梯度冲突在 Warmup 已存在；人工动作不是任意状态下的已证最优动作，不把“Q 更低”等同 Critic 真实价值标签错误。</td></tr>
<tr><td>Actor 改善自主插入</td><td>证据不足</td><td>仅动作拟合改善；无新独立真机评测。原版 Online / MC30 既有对照见前次完整审计。</td></tr>
<tr><td>进入受控冻结真机验收</td><td>证据不足，当前不放行新模型</td><td>新候选在两条原始 HIL 开发 Episode 上均值仍比初始模型差；需先验证输出一致性、数据契约与执行日志。没有自动部署或运动。</td></tr></table>
<h2>关键图</h2><img src="figures/q_direction.png"><img src="figures/td_and_proposals.png"><img src="figures/learning_curves.png"><img src="figures/per_dimension_errors.png"><img src="figures/retention.png"><img src="figures/sampling.png">
<h2>最终固定对象比较</h2><small>单位 rad；每 Episode 等权后取六关节平均。列 1 只有两条 HIL 开发 Episode（初始化训练见过）；列 2 六条反复选型 Online 开发 Episode；专家列 120 条初始化训练见过的 Episode。夹爪单独见图。不能汇总为自主成功率。</small><table><tr><th>因素</th><th>种子</th><th>原始 HIL 开发 MAE</th><th>旧 Online 开发 MAE</th><th>专家保留 MAE</th></tr>'''+table+'''</table>
<h2>下一步优先级与回退</h2><ol><li>先合并/审核可选 Replay 精度补丁和 trace 完整性修复；保持生产数据和默认不变。精度模式必须在各角色启动前一致指定，回退需下一次启动采用 legacy，不能在线混换 buffer dtype。</li><li>补齐同状态原始人类命令、计划提议的 anchor、实际发布命令、反馈、Episode UUID 与单调时钟。HIL 历史反馈不能直接替代真实命令；需可靠时间对应后再建重采样/credit 实验。</li><li>Actor Q off 暂作为研究候选。它相对相同 FP32 输入的原 Q 项只带来小幅模仿收益；不以该指标自动上线。固定模型做更广的完整 Episode 诊断；不把已有开发集改名为测试。</li><li>按原生单位校准 delta 约束是待验证假设，先报告梯度比例，再单因素修改；时基未一致时不根据旧 HIL 帧差盲目放大该项。</li><li>执行实验固定 Actor，先 faithful20 对 async_rtc20；再分别改变 RTC、EMA 和频率。新独立验收按完整 Episode 区分自主/辅助/失败，记录 intervention 及 deadline/跟踪误差。任何不一致、异常动作或退化即停止，回退固定 Warmup5k 与既有执行 profile。</li></ol>
<h2>代码、验证与身份</h2><p>代码独立分支 <code>audit/q-guidance-20261005</code>。生产 main、固定上游、现场服务、机器人、生产 Replay/权重/默认未修改。本轮只用 A6000 GPU0；训练结束资源已释放。修复影响范围的 CPU 检查记录见 <a href="repair_tests.log">repair_tests.log</a>。初次全套检查受缺少 worktree 依赖/augmax/fastapi 等环境影响，未宣称全部测试通过。</p>
<p>实际数据身份 <a href="input_identity.json">input_identity.json</a>，完整池身份 <a href="full_pool/pool_identity.json">pool_identity.json</a>，代码身份 <a href="code_identity.json">code_identity.json</a>，命令 <a href="launch.json">初次训练</a> / <a href="qoff_launch.json">Q 对照</a> / <a href="full_pool_launch.json">完整 Warmup 池对照</a>，配置与状态 <a href="full_pool/training_comparison.json">training_comparison.json</a>，最终对象 <a href="full_pool/fixed_comparison.json">fixed_comparison.json</a>，配对 Episode 不确定性 <a href="summary.json">summary.json</a>。Warmup5k SHA <code>be50b6cf0586864d89173a6456128bffa8b37621a6fda500f5be287223050f33</code>；生产 Replay SHA <code>0fb87e9ecc3b4e0ede50208caa3ceea93db3330bf976edf682cd6c21f684525a</code>。</p>
<p>前次三阶段与论文诊断：<a href="../full-chain-audit-20261001/index.html">完整审计</a>；原始精度问题：<a href="../q-guidance-localization-20261005/index.html">CPU 追溯</a>。新交付保存为 HTML/JSON/图与代码，未新增正式流程 MD。</p>'''
(out/'index.html').write_text(html,encoding='utf-8')
print(json.dumps({'figures':6,'comparisons':comparisons,'sampling':sampling}))
