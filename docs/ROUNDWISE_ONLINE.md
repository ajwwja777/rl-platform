# 按轮在线更新候选（可选，未放行）

2026-10-06。A6000为代码、Git、报告与候选权重主位置；现场默认Actor、Replay、Learner与执行配置不变。本方案是新实验目标，不宣称忠实复现评审四头Critic或自动提升自主成功率。

## 流程

固定Actor采集并记录版本／执行配置 → 闭合完整Episode及只读快照SHA → Episode分组和固定开发集 → Critic使用末段逻辑时间窗口，Actor保留全轨迹 → CPU私有训练候选 → 导出一致性、旧场景保持与独立冻结对照 → 通过后由现场负责人发布；失败保留原Actor。采集期间不更新或自动发布，不混用不同起点／执行方式作算法因果比较。

tail_steps=60、接管前20／后10只是逻辑步时间代理，不是语义阶段；专家示教与rollout有历史采样时基差异，不能统一叫“最后3秒”。

保留自主成功、辅助成功、失败和HIL原标签。每次policy→human分别截断自举，接管前派生回报置零仅定义“自主完成”实验目标，不证明此前动作都错误。辅助成功中的策略不归入自主成功。混合来源窗口拒绝训练，完整性、终止奖励、source冲突失败记录原因。

Critic每轮重新初始化，固定本轮起始Actor作TD target，保留原生TwinCritic两头、EMA和归一化。仅验证“唯一末端单位奖励，其余0”后把TD target限制[0,1]；报告原始Q不裁剪。gamma=.99，C10 discount=.99^10，末端奖励槽位仍乘对应gamma，成功末窗不是直接等于1。

Actor从指定权重继承，Adam每轮重新初始化。人类／专家和自主成功使用历史实际动作；失败／辅助策略使用弱Reference锚。无Q梯度、无Q优势权重。人类动作不是已证明最优控制命令。暂不替换人类next Reference，不增Critic头数，不修改Stage1。

四个互斥类别配额 → 类别内先抽Episode → 再抽窗口。20/20/20/30权重和90，要归一化；batch128实际29/28/28/43。failure池包含辅助策略，不是任务失败率。最近轮次加权2必须提供显式Episode→round清单，没有则recent_ratio留空，不按ID猜测。

## 执行入口

从A6000仓库根目录运行；网页／机器人无需在线：

~~~bash
.venv/bin/python scripts/run_roundwise_update.py --profile conservative \
  --journal /absolute/closed_snapshot/replay_journal.pkl \
  --init-checkpoint /absolute/checkpoints/latest.pkl \
  --norm-stats /absolute/action_norm_stats.json \
  --development-journal /absolute/fixed_development/replay_journal.pkl \
  --rounds /absolute/episode_rounds.json --round-id round-001 \
  --output /absolute/independent/candidates/round-001 --dry-run
~~~

去掉dry-run才训练，强制CPU／最多4核。输出必须独立绝对candidates目录，不覆盖输入或已有输出。生产journal增长时，先由采集负责人闭合／提供快照；读取变化或pickle截断拒绝。

configs/experiments/roundwise_online.json默认未启用：
- conservative：全轨迹Actor，LR1e-5，Critic1000／Actor1500。
- learning_rate_control：全轨迹，LR1e-4；与conservative仅改变LR。
- coverage_control：末段，LR1e-4；与learning_rate_control仅改变Actor覆盖。

固定batch128、delta_weight1、failure_anchor.1、std.002、Reference dropout.5、有效BC系数5。分离Critic／Actor阶段，不宣称原生Actor每两次更新周期。内部hash20%开发、历史外部20Episode开发均不是独立测试，初始Actor可能见过内部开发。归档专家ID100000+只在核对materialize_warmup_experts.py后显式加--legacy-expert-id-base 100000；现场默认负ID规则不变。错任务排除用具体journal SHA绑定的--excluded-episodes，不按相同数字ID跨资产排除。

同轮恢复用相同输入／配置及--resume；Actor更新后禁止延长Critic。下一轮init-checkpoint可用上一轮round_state.pkl，继承Actor权重／版本，Critic和优化器重新初始化。round_state.pkl不是上游Learner的latest.pkl；部署格式为actor_snapshot/actor_snapshot.pkl及相邻norm。publication.json固定staged／release_authorized=false，不覆盖原生自动Learner资产。

## 验证与停止

outputs/roundwise-online-review-20261006/REPORT.md、coverage_commands.json、各metadata.json记录身份／实际命令。六候选均完成1000／1500，源SHA不变，精确断点恢复和CPU原生导出一致性通过。全量417通过／18基线同名失败，不称全绿。

失败：末段Actor在两个种子下损害旧专家和自主成功拟合。全轨迹／低LR缓解，但自主成功开发关节MAE仍略增。开发状态AUC=1不是放行门槛。无独立OOD任务评测，任何候选均未批准发布。

无动作加载只验接口、norm、版本和动作一致性 → 受控冻结真机验收 → 分批学习。固定Actor和执行方式，独立未调参目标位置档位，基线与候选随机交错，禁止Learner；完整Episode区分自主／辅助／失败及区间。预先定义自主收益、旧场景非劣界限、安全／抖动阈值，统计发布间隔、推理预算、deadline、跟踪与HIL时间；缺trace不补造。

输入身份改变、奖励/source不符、非有限值、保持下降、越界／反馈异常或版本混乱，停止候选更新／发布，回原5k／原配置；不删Replay、不改标签。CPU拟合不能保证真机在线越来越好。
