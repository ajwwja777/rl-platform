# RL 平台：当前进度与关键经验

2026-10-14 RLT：[Teacher5首批真实Online审计](online/tasks/2026-10-11-online-learning-evidence/TEACHER5_FIRST_ONLINE_BATCH.md)：恢复后3集2成功/1失败、89条入库/89更新/45Actor，最新8234/4117未发布；执行仍8127/4063。新增3集实际Critic/Actor采样4.55%/4.10%，新TRAIN回报残差下降，重复DEV平均拟合保持；无新Actor自主增益证明。用户说明抓持相对位置有变化；现场随后已交EXPO，数据/结论分开。

2026-10-14 RLT恢复：[Teacher5 CPU Learner恢复结果](execution/tasks/2026-10-14-teacher5-learner-crash/RESULT.md)。已捕获正式GPU训练编译SIGSEGV；仅Teacher5 Learner改CPU，19回归通过/已部署，正式18更新完成8127→8145并保存重载，预算0。served4063不变/pending4072，ready暂停0命令、三相机约30Hz；用户可继续受控Online。GPU底层原因及长期rollout稳定性仍待验证。

2026-10-14 EXPO：[Cobot本地Actor / LFT1 Learner经LFT2中转](online/tasks/2026-10-14-expo-native-sync/TASK.md)正式Web暂停加载交付完成：两端ready/version0一致、25seed/3301帧、online0、Session启停/目录/健康检查通过，发布0；用户已结束Teacher5。原生20Critic/1VLA与三机传输隔离验证通过；真实rollout/HIL与成功率仍由操作员验收。

2026-10-14 RLT：[Stage1证据复核](stage1/tasks/2026-10-14-evidence-reconciliation/RESULT.md)确认已有134集/19022帧分析图前向，并完成三权重CPU比较。Teacher5训练内六关节MAE1.612mrad、HIL8 1.640、原5k 1.651；单改Reference为20Hz均变差。保留Teacher5原输入合同，不作权重/现场切换；自主Online增益仍证据不足。以下加载/进程描述为各日期历史回执，不代表当前现场。

最新交付（2026-10-13）：[Teacher5 8127](online/tasks/2026-10-11-online-learning-evidence/TEACHER5_DELIVERY.md)已注册/同步并真实暂停加载，Actor4063、Replay3623、新预算0，50HzRTC/EMA、发布0。用户已批准从EXPO切换；首轮真实采集和学习闭环待用户，不是自主收益认证。此前“尚未登记”描述是历史状态。

最新（2026-10-13）：[online-review核验与候选](online/tasks/2026-10-11-online-learning-evidence/REVIEW_VALIDATION.md)完成旧review回报口径修复、9组CPU对照和原生包验证。clip10从未触发、min-Q无新纠正收益，旧AWBC有保持代价；保留teacher5，8127/Actor4063包已准备但未现场登记/同步。独立自主收益仍证据不足；旧评估口径不能再作为Q高估根因证据。

最新（2026-10-13）：[teacher续训对照](online/tasks/2026-10-11-online-learning-evidence/TEACHER_CONTINUATION.md)从HIL8 7281完整state、同teacher/数据/batch做50 vs5，6组CPU完成；teacher5新HIL DEV较50改善1.61%，旧DEV均值/尾部在工程筛选内，尚无独立自主收益。现场保持Actor3640；当前constant新运行处于evaluation路由，探索/Replay关闭，不能将口述12/15当完整训练数据或低频探索生效证明。

最新结论（2026-10-12）：[匹配学习结果](online/tasks/2026-10-11-online-learning-evidence/RESULT.md)已完成独立journal命名空间修复、2配方×3seed×846次CPU更新及86项回归/精确恢复/导出。真实Actor新数据占比原13.3%/HIL8 29.6%；HIL8新HIL TRAIN/DEV拟合−2.29%/−1.44%、旧DEV保持较好，但新HIL各仅一集，不支持自主Online增益或自动发布。新权重未上机。[低频探索候选](execution/tasks/2026-10-11-chunk-continuity/LOW_FREQUENCY.md)已安装/网页可见，固定Actor3640、Learner关闭；开环命令加速度−9.9%，真机抖动仍待验收。以下旧日期内容是历史状态，以此段和对应TASK最新节接续。

2026-10-12当前主线：[原配方/HIL8匹配学习对照](online/tasks/2026-10-11-online-learning-evidence/TASK.md)。用户已确认开始；原5k完整state与归一化核验、29Ep/1068条探索快照和23TRAIN/6DEV拆分完成。两组同起点/预算，先验证真实batch与旧DEV隔离再训练。新旧Episode编号需仅在实验副本映射，避免近期采样失效；没有新训练或现场更改。

上一项（已完成）：[忠实复现差异核对](architecture/tasks/2026-10-12-rlt-reproduction-audit/TASK.md)。当前HIL8/Supported是RLT衍生候选；网络保持但MC30/teacher/LR/HIL采样/预算/执行均有差异。std.002、BC5/Q.1/delta10来自原插孔任务配置，非全部后加。现场只读explore/Actor3640/learner关闭。暂停新防抖/训练改动，先向用户明确基线与候选边界。

上一项（2026-10-12，当前暂停实施）：用户要求从最终输出与训练动作一致性重新设计平顺执行，不继续叠加EMA。[动作契约与训练RTC核对](execution/tasks/2026-10-11-chunk-continuity/ACTION_CONTRACT.md)确认当前20Hz逻辑action只取每段最后50Hz命令；读取原RLT/训练RTC/XR-1/Real-Time EXPO原文和源码，提出前缀条件/延迟一致性及低频探索方向，尚未实施新训练或现场切换。

历史优先项（2026-10-11）：用户要求先修执行平顺性，离线BC/Q训练暂缓。[RLT平滑候选](execution/tasks/2026-10-11-chunk-continuity/TASK.md)在相同历史目标复放中命令加速度RMS中位下降47%，目标跟随误差增加，未证明消除大幅往返或提高成功率；新可选入口使用同Actor3640/RTC50Hz/探索，关闭Learner、独立数据，旧入口保持。现场交付状态见RESULT。

最新实际核对（2026-10-11 夜）：[第三批探索数据](online/tasks/2026-10-11-exploration-collection/BATCH3.md)累计29完整Episode（11自主成功/2辅助成功/16失败），1068条Replay与strict回执逐数组一致；固定Actor3640、训练batch0。用户中途移动插孔，前后结果不可直接归因。含HIL窗口仅19条/1.78%，下一步为匹配离线纠正BC与加Q诊断；尚未执行新训练。缓存新运行56hits/62requests，构建9.017秒。

2026-10-11 晚：[探索采集收尾/目录/末端往返](online/tasks/2026-10-11-exploration-collection/FINISH_AND_MOTION.md)：已定位test历史竞态与269/564重复特征引用，修复CPU验证通过；新计划Reference在末端每.25s交替跳变，纯Stage1/RTC/探索归因仍需单因素证据。

更新：2026-10-11。本次为文档框架整理，技术状态引用已有证据，不代表重新查询现场、复跑实验或得到新的成功率。

## 当前重点

Online 新经验能否改善自主能力仍是主要问题。最近正式交付为 HIL8 可选分批 Online 候选；已有离线软件一致性和局部模仿改善证据，失败终点 Q 残差、位置 OOD、独立 TEST 与自主持续收益仍不足。下一步见 [Online 当前问题](online/tasks/2026-10-11-online-learning-evidence/TASK.md)，不因文档整理自动训练或部署。

| 模块 | 当前进度与未完成项 | 入口 |
|---|---|---|
| 架构/环境/迁移 | MD 模块/任务整理完成；代码与资产原位置保留 | [architecture](architecture/README.md) |
| Stage1 | 数据/输入/原生数值链已有历史审计；独立泛化与 RL 表征能力仍不足 | [stage1](stage1/README.md) |
| Warmup | 5k/20k 选择与后续候选有记录；重复 DEV 不是独立 TEST | [warmup](warmup/README.md) |
| Online | HIL8 3seed 局部拟合改善；接管截断/末端配额未通过保持；自主持续收益未证实 | [online](online/README.md) |
| 执行 | 发布准备/trace/反馈修复有交付证据；当前进程与闭环表现须另查 | [execution](execution/README.md) |

文档整理本身的进度/验收：[任务](architecture/tasks/2026-10-11-documentation-framework/TASK.md)。另一对话研究草稿保持未合并，入口在 Online 模块；不将草稿主张升级为事实。

当前执行交付：[固定 HIL8 探索采集](online/tasks/2026-10-11-exploration-collection/RESULT.md)。A6000 101项/现场52项 CPU 合约与回归通过，新入口已同步且网页可见；固定权重、Learner关闭、独立数据目录。历史第二批：用户已采11个完整Episode（6失败5自主成功、无HIL；新增4轮同位置均成功），探索/Actor身份有实际trace；模型目录自动匹配已同步现场，77项前端回归通过，实际history可读11条。通用HDF5 action存在双臂联合有效性限制，右臂coordinator字段与独立Replay须分开核对。[保存位置纠错](online/tasks/2026-10-11-exploration-collection/STORAGE.md)。Online收益仍未证实。

## 关键经验

- 先闭合数据身份、真实训练输入、更新和 served 权重，再讨论能力；配置值、文件存在、测试通过不是当前运行证据。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- TRAIN、重复 DEV、独立 TEST 分开；自主/辅助成功分开；完整 Episode 是不确定性单位，重叠窗口不是独立试验。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- 同时看损失/排序、同状态动作偏好、物理动作幅度与闭环结果。低 TD loss、Actor Q 上升、末段 AUC 或模仿改善，均不能单独证明 Online 越来越好。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- 历史失败、反例、错误实验标签和回退也保留；资源/生产边界与算法假设分开。恢复要验证完整状态与采样轨迹，不能只看步数。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- 文档按模块归属，每个问题一个主任务；详细证据归任务和原产物，guide 只保留摘要。多对话接管先查进度、Git 和 coordination。[约定](../AGENTS.md)

## 如何继续工作

从模块 README 找任务，再读 TASK.md 的范围、验收、进度和下一步；复杂任务再读 ANALYSIS.md/RESULT.md。已验证/失败/证据不足必须有证据入口。具体任务进度只在 TASK 维护，模块记录稳定认识，这里只汇总关键变化。

代码导航见 [项目 README](../README.md)，现场操作见 [执行手册](execution/RUNBOOK.md)，历史批次见 [迁移记录](architecture/MIGRATION.md)。旧路径页仅供兼容；新增正文写对应模块。outputs 中原始报告、图、日志和 manifest 不移动，各模块列出所属入口。

新增经验：合并独立Replay不能把源内Episode小号当全局新旧顺序；先建立带源身份的时间有序映射，再检查实际Actor/Critic批次和DEV零抽样。最终buffer比例不能代替逐集到达过程的训练比例。[证据](online/tasks/2026-10-11-online-learning-evidence/RESULT.md)

2026-10-13新增交付：[EXPO左臂双腕RTC-SFT2000网页接入](execution/tasks/2026-10-13-expo-web/TASK.md)，独立原生checkpoint/norm与环境，14执行行为及46Web相关CPU回归、真实GPU/HTTP暂停加载通过；最终ready/paused、发布0命令，无训练/动作。HIL等操作接口已接通，真机短轮及成功率待用户，不改变RLT主线。

## 2026-10-13 EXPO 单卡同步验证

Cobot4090 原生完整 EXPO 推理通过，完整更新 batch2/UTD1 两种分配器均 OOM；原 SFT2000 已恢复暂停，尚未启用在线学习。[配置、峰值与证据](online/tasks/2026-10-13-expo-online/TASK.md)。

新增经验：重叠窗口的折扣回报按原始step_id计算；不能把接管前观察回报置零后仍称原目标校准。候选按训练源码SHA锁定时，新实验应隔离模块，避免无意破坏旧候选加载；工具合并必须核对实际各臂LR/delta元数据。[证据](online/tasks/2026-10-11-online-learning-evidence/REVIEW_VALIDATION.md)

## 2026-10-13 EXPO 单卡 Online 接入

新增独立 EXPO Online 网页选项，Cobot 单张4090上串行 rollout/HIL 与原生学习，完整更新/保存/恢复及合成回合续训通过。未改原固定SFT模型或RLT默认，尚不代表真机效果提升。详细状态、路径和验证边界：[EXPO Online 交付](online/tasks/2026-10-13-expo-online/RESULT.md)。
