"""CPU analysis and numerical reproduction without importing model packages."""
import ast
import collections
import hashlib
import html
import json
from pathlib import Path
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform')
OUT = ROOT / 'outputs/q-guidance-localization-20261005'
WORK = Path(__file__).resolve().parents[1]


def extract_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def main():
    raw = json.loads((OUT/'data_contract_raw.json').read_text())
    guidance_path = ROOT/'outputs/rlt-diagnosis-20260930/critic_guidance.json'
    guidance = json.loads(guidance_path.read_text())
    replay_source = ROOT/'third_party/openpi-rlt/rlt_online_rl/src/rlt_online_rl/replay.py'
    adapter_source = ROOT/'third_party/openpi-rlt/rlt_online_rl/src/rlt_online_rl/action_representation.py'
    namespace = dict(np=np, _ensure_array=lambda x,dtype:np.asarray(x,dtype=dtype),
                     collection_phase_to_id=lambda x:dict(warmup=1,online=2)[x])
    serialize = extract_function(replay_source, 'to_numpy', namespace, 'RLTTransition')
    for n in ['_broadcast_state0','_zero_row_mask','_quantile_normalize']:
        extract_function(adapter_source,n,namespace)
    delta = extract_function(adapter_source,'_abs_to_delta_chunk',namespace,'ActionRepresentationAdapter')
    norm = namespace['_quantile_normalize']
    example = next(e for e in raw['matched_hold_examples'] if e['episode_id']==163 and e['step_id']==50)
    state = np.asarray(example['proprio'],np.float32)
    action = np.repeat(state[None],10,axis=0)
    transition = types.SimpleNamespace(z_rl=np.zeros(2048,np.float32),proprio=state,ref_chunk=action,
        action_chunk=action,rewards=np.zeros(10,np.float32),done=False,next_z_rl=np.zeros(2048,np.float32),
        next_proprio=state,next_ref_chunk=action,source=2,source_chunk=np.full(10,2),
        collection_phase='online',success=0,intervention_flag=True,episode_id=163,step_id=50)
    stored=serialize(transition)
    original_delta=delta(None,action,state)
    stored_delta=delta(None,stored['action_chunk'],stored['proprio'])
    st=types.SimpleNamespace(q01=np.asarray(raw['normalization']['stats']['q01'],np.float32),
                             q99=np.asarray(raw['normalization']['stats']['q99'],np.float32))
    norm_error=norm(stored_delta,st)-norm(original_delta,st)
    checks={
        'historical_report_identity_matches_journal_prefix':raw['historical_3917_prefix_sha256']==guidance['source_sha256'],
        'journal_stable_during_read':raw['journal_stable'],
        'actual_serializer_action_fp16_state_fp32':stored['action_chunk'].dtype==np.float16 and stored['proprio'].dtype==np.float32,
        'raw_hold_has_zero_joint_delta':bool(np.all(original_delta[:,:6]==0)),
        'serialized_hold_has_false_joint_delta':bool(np.any(stored_delta[:,:6]!=0)),
        'serializer_reproduces_observed_replay_action':bool(np.array_equal(stored['action_chunk'][0],example['stored_action'])),
        'actual_normalizer_reproduces_observed_error':bool(np.allclose(norm_error[0,:6],example['normalized_joint_delta_error'],rtol=1e-5,atol=1e-6)),
        'six_terminal_targets_equal_discounted_success':len(raw['episodes'])==6 and all(np.allclose(e['observed_terminal_targets'],[.99**9]) for e in raw['episodes']),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    matched=collections.defaultdict(list)
    for t in raw['traces']:
        for m in t['replay_exact_window_matches']:
            if m['start_state_exact'] and m['next_state_exact']:
                matched[m['replay_index']].append(m)
    unique=[ms[0] for ms in matched.values() if len(ms)==1]
    online=next(t for t in raw['traces'] if '1790660568' in t['path'])
    summary=dict(status='CPU localization complete; solution awaiting user review',
        code_head='bf111eb70cb25d3e65de58a0dd8a63c8f8b23cbe',
        pinned_upstream='8cef77eb7c5211b45382bf9199c9cdf0aaf60a19',
        data_identity=dict(journal_sha256=raw['journal_sha256'],historical_prefix_sha256=raw['historical_3917_prefix_sha256'],
                           normalization_sha256=raw['normalization']['sha256']),
        checks=checks, example=example,
        raw_to_replay=dict(state_and_action_matches=len(matched),unique_origins=len(unique),ambiguous_origins=len(matched)-len(unique),
            episodes=len(set((m['phase'],m['episode_id']) for m in unique)),
            target_six_episode_trace_matches=sum(m['episode_id'] in [184,193,212,215,217,233] for m in unique)),
        guidance_endpoint='Historical recorded actions after FP16 absolute-action storage; Actor actions are offline deterministic proposals.',
        numerical_fidelity='failed: nonzero delta is created for a verified raw hold action',
        robot_failure_causality='insufficient evidence: no model forward or robot experiment in this audit',
        next_state_nonterminal_td='insufficient evidence: needs target network and RNG reconstruction',
        source_files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [replay_source,adapter_source]})
    (OUT/'findings.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps(dict(checks=checks,passed=sum(checks.values()),model_imports=False,
        note='AST extracts actual serializer/delta/normalizer functions; phase-ID conversion is a stub unrelated to numerical checks.'),indent=2)+'\n')
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(np.arange(6),stored_delta[0,:6]*1000,color='#cf5c36',label='Stored FP16 action - FP32 state')
    axes[0].axhline(0,color='#283845');axes[0].set_xticks(np.arange(6));axes[0].set_xticklabels(['J%d'%i for i in range(1,7)])
    axes[0].set_ylabel('Joint delta (mrad)');axes[0].set_title('Raw hold delta = 0 on every joint')
    axes[1].bar(np.arange(6),norm_error[0,:6],color='#386cb0');axes[1].axhline(0,color='#283845')
    axes[1].set_xticks(np.arange(6));axes[1].set_xticklabels(['J%d'%i for i in range(1,7)])
    axes[1].set_ylabel('Normalized action error');axes[1].set_title('Error reaches the training input')
    fig.suptitle('Training Replay: Online Episode 163, step 50; numerical reproduction, not model error')
    fig.tight_layout(rect=[0,0,1,.9])
    for ext in ['png','svg']:fig.savefig(OUT/'figures'/('hold_quantization.'+ext))
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    runs=online['human_runs'];x=np.arange(len(runs));w=.35
    axes[0].bar(x-w/2,[r['source_timestamp_span'] for r in runs],w,label='ROS-source timestamp span')
    axes[0].bar(x+w/2,[r['assumed_20hz_span'] for r in runs],w,label='(rows - 1) / 20')
    axes[0].set_xticks(x);axes[0].set_xticklabels(['HIL run 1','HIL run 2']);axes[0].set_ylabel('Span (s)');axes[0].legend(fontsize=8)
    axes[0].set_title('Sampling time is not a fixed 50 ms grid')
    variants=['warmup5000','baseline','mc_30'];means=[];low=[];high=[]
    for name in variants:
        v=guidance['variants'][name]['q_paths']['own_actor']['q1_actor_objective']['episode_equal_mean_q_change']
        means.append(v['mean']);low.append(v['mean']-v['bootstrap_95_interval'][0]);high.append(v['bootstrap_95_interval'][1]-v['mean'])
    axes[1].errorbar(range(3),means,yerr=[low,high],fmt='o',capsize=4,color='#386cb0');axes[1].axhline(0,color='#555555',ls='--')
    axes[1].set_xticks(range(3));axes[1].set_xticklabels(['Warmup5k','Native7k*','MC30 7k*']);axes[1].set_ylabel('Q1(replacement) - Q1(Actor)')
    axes[1].set_title('Existing development result: only 6 Episodes')
    fig.suptitle('Left: trace fragments, no confidence interval. Right: Episode bootstrap; * matched offline candidates')
    fig.tight_layout(rect=[0,0,1,.9])
    for ext in ['png','svg']:fig.savefig(OUT/'figures'/('timing_and_q_boundary.'+ext))
    plt.close(fig)
    plan='''待审查方案：先做动作精度单因素诊断，再决定是否修改运行入口。

1. 对可追溯的完整 Episode 建立独立诊断清单。只恢复有原始 float32 数值、动作/source/当前状态/下一状态一致且身份可判定的动作；歧义窗口和缺失片段留空。不将旧 float16 升成 float32 后称为恢复。
2. 固定 Warmup5k learner5000/Actor2500、Critic、状态、Reference、归一化、reward、discount、样本身份。对照原始 float32 执行动作与它的 float16 存储版本，量化 BC 目标、Q1/Q2 及 Q 对纠正方向的梯度差异。有匹配 target 网络状态时才计算非终止 TD 残差；没有就留空，不能拿当前 Actor/Critic 冒充 target 网络。未执行 Actor 动作不套实际执行动作的 next-state。资源为 A6000 CPU 与既有小 Actor/Critic；不加载 Stage1，不占现场 GPU，不训练。拟读取 outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials/experts120_5000/{final_actor_snapshot.pkl,final_critic_snapshot.pkl}，候选资产约9.2MB，文件哈希在 cpu_model_asset_candidates.json；执行前核对模型步数/参数/配置与固定5k一致，不一致即停止。不复制整份权重或Replay，只读取必要且可追溯的Episode输入。
3. 若支持精度影响，项目内增加可选 Replay 动作精度适配：action_chunk 保持 float32，proprio/reward 等保持原样；首轮精度对照保留 Reference 的现有精度，以免同时改变两个因素。EnvDriver 发送、Replay 入库、journal 恢复和 batch 存储全链均须验证，不能只改写盘一处。固定 third_party 不改；旧入口默认不变；新记录与数据版本可识别。
4. 先做序列化/RPC/buffer/恢复的 CPU 契约验证，再在独立只读数据与输出上做匹配对照。没有证据前不开展新训练。若以后批准训练，两个分支同起点、同抽样序列、同更新预算，唯一区别为动作存储精度；结果按 Episode 统计，不将开发集当独立测试。
5. Reference/next-Reference 也被量化，可能产生训练与推理输入差异，另作单因素检查。HIL 反馈代替命令与采样时基也分别验证/修复，不与动作精度实验一起改。需要控制项目提供只读实际 command 与时间戳；不由本对话改控制/RTC。

支持的结论：CPU数值验证可确认精度契约；固定模型诊断可确认 Q/BC 是否受此影响；离线拟合改善不能证明自主成功率提升。
停止/回退：身份、时钟、归一化不一致立即停止；旧数据不可恢复则不纳入成对比较；实验输出独立，原Replay/权重/默认/上游保持原样；恢复原入口即可撤回可选适配。真机与部署另行验收。
本方案尚未执行；等待用户审查。'''
    (OUT/'proposed_solution.txt').write_text(plan,encoding='utf-8')
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>纠正动作低Q：数据契约定位</title>
<style>body{max-width:1000px;margin:36px auto;padding:0 20px;font:17px/1.7 system-ui;color:#23303b}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px;font-size:14px}small{color:#596570}a{color:#235fa8}</style>
<h1>先定位到数据精度损失，尚未证实低Q的唯一原因</h1>
<p><b>已验证：</b>Warmup/Online 的绝对动作在进入 Replay 时转为 float16，proprio 保留 float32；训练时才相减得到 delta。原始“保持不动”的动作因此变成非零训练目标。最早定位到的环节是进入 Warmup Replay 的序列化，Online沿用；Stage1不走这条Replay链。</p>
<p><b>失败（数值保真）：</b>真实 Online Episode163 / step50 的 J3 原始状态与动作都是 −1.4029337168rad，存储后动作变为 −1.4033203125rad；实际零delta变成 −0.0003865957rad，按实际归一化放大为约 −0.11207 的差异。该差异会进入 BC 监督及 Critic 动作输入。</p>
<img src="figures/hold_quantization.png" alt="实际保持动作经过存储后出现非零delta">
<p>这不是对模型误差的估计。用固定上游的实际序列化、delta、归一化函数在CPU复现，未导入模型。float16上游设计本身可能是节省内存的取舍；本次确定的是它损失了此任务的细微动作信息，尚未证明它单独导致真机失败。</p>
<h2>沿途核对与排除</h2><table><tr><th>检查</th><th>结论</th><th>边界</th></tr>
<tr><td>数据身份</td><td>已验证：当前4013条Replay的前3917条字节哈希与旧Q报告完全一致。</td><td>当前追加不改变旧比较对象。</td></tr>
<tr><td>成功奖励</td><td>已验证：6条比较Episode均有终止成功奖励，末端动作块TD目标均为0.99⁹≈0.913517。</td><td>不能据此判断较早非终止动作的TD目标；目标网络未运行。</td></tr>
<tr><td>trace与Replay对应</td><td>已验证：按实际float16序列化比对动作/source，再核对float32起始与后继状态。具体覆盖见findings.json。</td><td>排除歧义匹配；原Q比较的6条没有直接trace匹配。3份HDF仅是时间兼容候选，未冒充UUID确认。</td></tr>
<tr><td>人工动作语义</td><td>已验证：保留Online trace的354个人工行，action全部等于next-state反馈；自主行记录下发目标。</td><td>反馈轨迹可用于模仿，但与实际command不是同一量。对应HDF人工command与feedback也不同；不能把末端辅助成功当每步最优。</td></tr>
<tr><td>采样时基</td><td>已验证：两个HIL片段ROS来源时间跨度分别约3.436/4.288秒，按固定20Hz行数解释却为7.1/10.5秒。</td><td>来源时间戳不是发布时钟。没有据此计算真机deadline或发布Hz；确认的是不能把这些行无条件当固定50ms物理步。</td></tr>
<tr><td>Critic是否低估有效纠正</td><td>证据不足：已有负ΔQ比较使用量化后的记录动作；缺原始6条片段和逐窗口Q/实际TD目标。</td><td>不能宣称Q偏差或数据精度损失已证明自主失败因果。</td></tr></table>
<img src="figures/timing_and_q_boundary.png" alt="HIL来源时间跨度与现有Episode开发集Q比较">
<h2>排查方法</h2><p>从同状态动作比较出发 → 先确认Actor是离线重新计算的提议 → 核对奖励和终止 → 尝试原始trace对齐 → 精确匹配失败后追到序列化 → 发现float16转换 → 按真实转换恢复匹配 → 用保持动作在实际函数中复现伪delta → 将确定的数据损失与待验证的Q/真机因果分开。图只用于展示这条证据链。</p>
<h2>供审查的解决方案</h2><pre>PLAN</pre><p>下一步建议批准第1–2项的只读精度对照；得到Q/BC变化证据后，再审查第3项运行适配。尚未修改算法、生产Replay、权重、默认入口或固定上游；没有模型加载、GPU、训练、机器人动作或服务操作。</p>
<h2>身份与复现</h2><pre>IDENTITY</pre><p><a href="findings.json">结论与身份</a> · <a href="validation.json">数值验证</a> · <a href="collector_identity.json">采集命令</a> · <a href="proposed_solution.txt">待审查方案</a> · <a href="data_contract_raw.json">小规模数值证据</a></p></html>'''
    identity=dict(main_head=summary['code_head'],upstream=summary['pinned_upstream'],data=summary['data_identity'],
        read_command='ssh agilex@10.7.165.64 /home/agilex/jiaan/project/rl-platform/envs/stage1/bin/python - < scripts/collect_q_data_contract.py',
        analysis_command='python3 scripts/analyze_q_data_contract.py',output=str(OUT),
        boundaries='CPU NumPy/AST only, no model packages loaded; charts are training-data numerical diagnostics / reused development evidence; no independent test',
        trace_coverage=summary['raw_to_replay'])
    content=content.replace('PLAN',html.escape(plan)).replace('IDENTITY',html.escape(json.dumps(identity,ensure_ascii=False,indent=2)))
    (OUT/'index.html').write_text(content,encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps(dict(stage='CPU diagnostic complete; solution pending user review',
        completed=['data identity','sparse reward verification','actual serializer and raw trace match','hold action corruption reproduction','report and proposed solution'],
        pending=['user review','read-only Actor/Critic precision comparison','potential opt-in project adapter'],
        production_changes=False,model_loads=False,gpu=False),indent=2)+'\n')
    print(json.dumps(dict(validation=checks,coverage=summary['raw_to_replay'],output=str(OUT))))


if __name__=='__main__':
    main()
