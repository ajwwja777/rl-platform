# RLT 原仓库对齐审计与隔离基线

## 状态

尚不可交付真机。停止沿用 supported-IQL 候选；它已现场失败、选择器已拒绝。恢复论文/复现仓库的网络与目标函数，先建立同步 C=10 基线，再检验RTC适配。

## 依据

- 论文：[RL Token](https://arxiv.org/html/2604.23073v2)，原论文基座为pi0.6。
- 用户选定复现：[Yyshadow/openpi-RLT](https://github.com/Yyshadow/openpi-RLT)，pi0.5实现；固定commit `c1e40ac360185778c98cf20da2820e22d2d415e7`，只读工作树干净。它是复现仓库，不称为论文作者官方代码。
- 具体审计文件：`rlt_online_rl/src/rlt_online_rl/{networks,trainer,action_representation,inference,replay}.py` 与 `configs/tasks/agilex_ethernet/online_rl.yaml`。

## 已确认的差异

| 项目 | 复现仓库 | 旧plug_v2实现 |
|---|---|---|
| actor | 直接预测归一化动作块 | reference + 0.05*tanh残差；corrective另有0.2放大与brake |
| dropout | reference输入置零，独立生成动作 | 编码输入置零，但输出仍直接加回reference，存在旁路 |
| 归一化 | 单臂delta_chunk，任务q01/q99 | critic动作delta除0.05；训练/部署不同conditioner |
| LayerNorm | 无仿射，epsilon=1e-6 | Torch默认可学习仿射，epsilon=1e-5 |
| actor目标 | stochastic action，BC - Q + delta | 多轮自定义masked/conditioned BC、smooth，后来改IQL，非同一目标 |
| target更新 | actor每2步更新时同步Polyak | 旧代码每critic步更新target |
| 动作窗口 | 实际执行完整C10，补终局对齐窗口 | RTC决策点后移6步动作，奖励仍按原决策区间；部分终局遗漏 |
| 时序 | 同步单臂 | 30Hz异步RTC、99维state+prefix扩展，需要独立验证 |

这些不是仅仅调不同超参数。不能继续把旧候选的结果归为忠实原版RLT warmup效果。

## 为什么“相同参数”效果变化很大

- 历史成功使用 `rtc-corrective-r1@2000`，SHA256 `a197ab49db46bf7089740eb868400df57861051c34eeeaee625394d713a44918`；近期失败使用 `supported-iql-r1@500`，SHA256 `9ecccc0152055477a480e2f2aa82920322325994200f91e5d48641485ee1914d`。不是同一actor。
- 早期同一corrective权重跨r4/r5/r6部署时，运行时、重规划处理与采样设置改变；仅文件名/step一致不足以证明同一策略。
- 近期三轮supported测试内部权重相同、无在线更新；三轮漂移方向和暂停位置基本重复。因此这三轮不是“在线训练把模型改坏”。
- 离线同一观测重放不能冒充独立reference真机成功率；两次历史成功不能证明稳定成功率。

## 现场失败回归证据

- trace session `260615a5-995b-4278-8f7f-fff7daeb0fa7` 三轮各214条command，约7.1s，工作区x方向预测超出+0.125m而暂停。没有放宽边界。
- 右关节1实际反向72/74/76次（连续两步都大于1e-4rad）；actor原始相邻动作最大约0.071rad。限速后单步<=0.003333rad并不代表平滑。
- 两轮首段连续command时间间隔0.89ms/0.54ms；第三轮没有此突发仍抖，故时钟bug不是唯一原因。
- `control_clock.py`修复迟到后追赶发令；18项clock/conditioning/RPC测试通过。未重启服务，未执行机器人动作。
- 证据在Cobot `runs/plug_v2/diagnostics/upstream-audit-20260919/{raw-conditioned-motion,success-failure-motion}.json`。

## 同步原版基线

- 新代码仅隔离runner：`methods/openpi_rlt/plug_v2/{upstream_replay,upstream_baseline}.py`，不修改上游网络或loss。
- 调用上游完整窗口/终局补窗规则；同步观测及执行动作，不混入d6未来动作；暂停、无效帧、时钟断点不拼接。终局丢失时不提前贴成功标签；source/HIL按每步记录。固定原UUID拆分。
- 已生成169个episode文件、2239个窗口。train expert/success/failure=857/729/265，val=91/182/115。val failure8条均包含终局。比对旧warmup审计中的4条val failure没有done，说明旧缓存终局覆盖不足。
- 直接调用未修改的上游LearnerService，任务配置20k critic updates、actor每2步、lr1e-4、gamma.99、tau.005、std.002、dropout.5、warmup BC10/Q.1/delta10、batch128。动作分位数只从train计算。
- stratified使用原仓库0.4recent/0.3warmup/0.2human/0.1uniform；各pool重叠，human pool也含专家，不能将其说成“实际HIL恰好20%”。跟踪实际human_mask_ratio、bc_ref_penalty、bc_human_penalty及source比例。
- 本次同步模型明确 `live_RTC_compatible=false`，不发布、不接入现有99维RTC接口。当前已有val集仅作回归，不宣称独立未见测试集。
- 验证保存逐episode Q_data/Q_ref/Q_actor/Q_hold、位置、source、reward、done；早期policy Q按episode等权比较成功与失败，不能以专家/人工后段的高Q代替自主纠偏能力。

## 交付前仍需完成

1. 验证原版20k结果，并与reference和旧warmup用同输入、同单位评估动作拟合/高频反向/跳变；不以单一loss宣布改善。
2. RTC延迟适配：观测、动作、reward、next-state及prefix必须逐项说明；不把同步actor直接塞入旧99维接口。
3. 完整权重+归一化+运行时+参数不可变manifest；确定性重放、真实时钟回归、HIL/暂停/重启无旧chunk恢复。
4. 在线更新沿用原目标函数、版本冻结与拒绝发布机制；只有离线通过后给现场验收命令。真机效果及在线改进趋势仍须用户验收。

## 本轮核验结果（2026-09-19，离线诊断完成，尚无新真机交付）

- 原版配置20000步已完成：`runs/plug_v2/learning/upstream-baseline-20260919-r3`，126.11秒；单因素delta_weight=300对照5000步已完成：`runs/plug_v2/learning/upstream-delta300-20260919`，47.19秒。所有模型、optimizer与中间checkpoint保留，均未发布，live_RTC_compatible=false。训练进程已完成，不是后台持续训练。
- I/O优化只改外层状态/metrics落盘频率。两个500步重训不是逐位一致（最大actor参数差约1.99e-4）；raw heldout human MSE相差2.83e-9，rng相同。不能将此写成bitwise parity，证据 `io-optimization-parity.json`。
- 与reference相比，原版2000步的raw人工动作MSE低约17.4%，5000步低约15.3%；20000步回到略差于reference。训练loss下降不代表留出表现改善。
- 同步共同输入比较：旧warmup20000 raw human MSE=0.0002850524；reference=0.0002286958；delta300@5000=0.0001910144（比旧warmup低约33.0%、比reference低约16.5%）。但经同一个0.1rad/s、0.9rad/s²执行filter后，旧warmup=0.0002976332、新对照=0.0002959192，仅约0.58%改善。不能以raw动作优势宣称执行已改善。
- delta300由训练集输出梯度量纲校准：原delta项梯度仅BC的0.344%，目标10%对应weight约290，取300；单因素不改网络/奖励/TD。5000步raw动作delta-RMS中位数仍是reference的约1.14倍，未通过更平滑的要求。
- Q必须限定含义：验证集11条成功全部后来有右HIL介入，8条失败；**没有独立自主成功留出回合**。两个自主成功都在train。前30真实控制步严格对齐后，原版2000/5000/20000 AUC=0.989/0.966/0.443，delta300@5000=0.989；这区分的是“后续HIL成功/失败”，不能当作自主成功概率或在线自我纠偏证据。
- 原仓库stratified池有重叠，human含专家；不声称干预刚好20%。终局补窗已纠正旧缓存漏done问题，但同步dataset不是旧RTC缓存的原地替换。
- 示教/执行速率失配：按完整human窗口统计（重叠加权），48.6%的tick至少一个关节超过现部署0.1rad/s，13.36%的关节步超过该限速。此结果不授权放宽安全限制；应先统一动作时序、速度契约与训练目标，而非让filter掩盖输出。
- 证据/图均在Cobot `runs/plug_v2/diagnostics/upstream-audit-20260919/`：`model-comparison.json`、`outcome-strata.json`、`time-aligned-q.json`、`human-vs-deployment-rate.json`、`delta-units.json`、`upstream-diagnostics.png/pdf`。
- 下一阶段：原版actor/critic与RTC延迟/已承诺prefix的动作契约；执行滤波对策略学习和标签的影响；用历史成功/失败轨迹做部署回归。当前任何新checkpoint都不可直接装进旧99维服务。旧失败候选仍拒绝，未进行机器人动作、服务重启或根仓库commit/push。
