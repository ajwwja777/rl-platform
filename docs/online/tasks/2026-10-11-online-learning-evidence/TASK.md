# Online 学习效果与候选证据接续

状态：2026-10-13 从当前HIL8 7281的teacher50/5单因素续训已完成，见TEACHER_CONTINUATION。teacher5通过离线工程筛选，未发布；新现场评测未形成采集Replay，自主收益与低频探索实效仍需区分。后文按日期保留历史。

## 问题与验收

要回答：新增纠正与失败经验是否以正确目标、实际采样和足够动作幅度影响 Actor；怎样确认训练改善而非只改变代理指标。

离线验收须固定代码、数据、初始化/优化器/RNG与执行合同，核对目标和真实 Actor/Critic 批次，检查六活动关节的物理动作变化、旧 DEV 保持和恢复/导出一致性。自主成功、位置 OOD 和持续 Online 收益需要独立完整 Episode，现有重复 DEV 不能替代。

## 范围与边界

后续算法研究采用独立 worktree、只读资产快照与独立输出；GPU/实验预算在具体实验前登记，当前没有新资源任务。代码错误可在授权范围修复；假设须有单因素对照。禁止把文档归位当新验证、删除 HIL 标签、覆盖生产 Replay/权重或自动发布候选。

停止条件：数据/身份不一致、指标非有限、验证工具错误、旧 DEV 保持越界或资源边界不满足时，停止相应实验/交付，保留失败并继续独立核查。

## 已有证据

- [10 月 10 日匹配研究与交付](../2026-10-10-online-credit-repair/TASK.md)：候选 HIL8 7281/3640，单条新 HIL TRAIN 拟合改善；失败终点 Q、独立 TEST 与自主收益仍未解决。
- [实际采样报告](../../../../outputs/online-sampling-20261010/REPORT.txt)：历史 281 更新的身份与比例，不是 10 月 10 日新训练。
- 另一对话 `audit/online-review-20261011` 的研究经验未合并，具体入口见模块 README。它包含待检验判断；例如 TD 多项保护与本项目 matched takeover-censor 的保持副作用不能混为同一已验证结论。

## 进度与唯一下一步

已完成：将原记录与产物接入本模块，保留另一对话成果。
未完成：没有新增独立 TEST、真机自主增益或已解决全部 Critic 问题的证据。
下一步：继续研究前先核对实际新入库/训练/served 身份，并把待合并研究的每条主张对应到代码、数据和单因素证据；优先处理能改变交付决定的问题，不重复旧实验。现场若已有新轮次，重新登记数据身份后再分析。

本任务跨执行模块；执行延迟/录制故障只引用 execution 的验证，不由学习指标替代。

## 2026-10-11 探索采集就绪复核

已完成只读核查及隔离 6 项单元测试，见 [ANALYSIS](ANALYSIS.md)。另一对话探索/AWBC 尚未提交，现场不存在探索模块/入口；冻结入口强制确定性会跳过探索。W2 同种子同步数对照存在旧自主 DEV 尾部拟合代价，未取得 Online 交付认证。

本次核查无代码修改、部署或训练。当前下一步以此补记为准：补齐固定 Actor/探索开启/学习关闭的真实采集入口及日志/时序合同验证，再交付首批受控采集；W2 继续作为离线候选，不直接边采边学。具体证据与边界见分析，不覆盖原研究对话。

## 2026-10-11 执行接续

探索原型缺少部署/冻结模式/动作来源合约的问题已由[固定HIL8探索采集任务](../2026-10-11-exploration-collection/TASK.md)实现并同步。新入口 `plug-v3-hil8-explore` 固定权重、Learner关闭；当前唯一现场下一步为用户采3–5个完整Episode，先核对探索与实际动作证据。W2保持问题及持续自主收益仍待后续离线对照，不能把采集入口交付当学习放行。

## 2026-10-12：原配方与HIL8匹配对照，第1步

用户确认主线：固定基线 → 同起点/同新数据比较原配方与HIL8 → 依据结果做必要单因素实验 → 有证据的候选再交用户受控Online。该比较不是完整复现原实验：固定预算后原UTD5不在本轮复现范围，不能把多因素配方差异归因到某一项。暂停新增平滑/Stage1改造，不覆盖其他对话工作。

执行工作区：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/matched-learning-20261012/rl-platform`，分支`experiment/matched-learning-20261012`，代码基线542f560（已包含其他对话的目录接口修改，未改动其工作树）。本阶段只读Cobot、A6000 CPU解包/hash，不加载网络、不训练/GPU、不动现场服务/机器人/生产Replay。

### 已验证

- 共同起点选原5k完整checkpoint：`models/rlt/plug_v3_yyshadow/warmup-5000/checkpoints/step_5000.pkl`，SHA `be50b6cf0586864d89173a6456128bffa8b37621a6fda500f5be287223050f33`；实际state含Actor/Critic、target、两组优化器、RNG，步数5000/Actor2500。恢复逐叶/首步一致性尚待训练入口校验。
- 原5k与HIL8 action_norm_stats数值完全一致；文件SHA不同仅格式及provenance元数据，不应误报归一化改变。
- 现场探索Replay未新增：29完整Episode、1068条，SHA `919ff1adb31b6fcae0f3b3380ca80dc2e881b7f2501566d7042237674d0dcad8`；1068唯一Episode/step和strict回执数组一致。10,373,484字节数值快照已取到A6000独立输出并复核SHA；未复制视频/Stage1或修改原件。
- 在训练前按完整Episode分层并用固定hash顺序划分：23 TRAIN/846条（9自主成功、1辅助成功、13失败），6 DEV/222条（2自主成功、1辅助成功、3失败）。两条HIL分开；但该29条历史上已被分析，DEV不是独立TEST；一条训练HIL仅支持局部诊断。native DEV ID为2/5/15/17/23/25，录制ID不能直接当native ID。
- 拟定两个配方各846次Critic/423次Actor更新、batch128、配对seed41/42/43；从同一5k完整state开始。原配方使用原生TD/原采样与lr；HIL8使用当前MC/clip/teacher/分层/HIL8/lr。保留不更新的step0作为参照。实际入口逐项配置序列化及首步一致性尚待验证。

### 发现的接入风险及边界

新buffer的native ID为2–39，旧HIL8池中Online ID到10010；现有采样器按max ID近20范围及floor10000识别近期HIL。直接拼接时新数据不会进入近期/HIL8额外槽，且原始ID36/38/39与旧数据重复（这里只说明裸ID重复，不冒称完整phase/ID键碰撞）。实验manifest已记录独立来源namespace及10011起的新ID映射，后续仅在派生实验输入应用并验证采样回执。不是已证明的历史Online生产故障，不改生产Replay。

当前拟共用旧池为HIL8归档2777条/208 Episode；不能拿其原来已通过筛选的历史结论替代本轮DEV身份核对。旧DEV清单连接/排除、初始checkpoint已见数据边界仍待逐项确认。新数据的TRAIN/DEV划分已固定，不因训练结果重分。

现场最新只读GET session返回HTTP409；collection/model报告runtime_process_exited、ROS shut down while waiting for object reset，recorder stopped/complete、handover fault。记录事实，不在本离线任务诊断或恢复硬件；此前paused状态不是当前状态。接口错误已隔离，不妨碍独立Replay只读核验。

### 当前唯一下一步

完成旧DEV身份/泄漏清单与派生数据映射，搭建并验证两个训练配方的原生入口及step0一致性，先检查真实batch曝光，再按固定预算开始A6000 CPU匹配训练。CPU亲和性/可用资源与具体命令在启动前登记；不占现场GPU。此次没有新权重，也没有Online放行结论。

证据根：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/matched-learning-20261012/`。comparison-manifest.json、asset-identities.json、normalization-check.json、state-and-episode-check.json、check_replay.json、snapshot.json及freeze_manifest.py分别记录划分/计划、SHA、数值、实际state、Replay逐数组检查、现场只读事实及固定划分脚本。原始29条的结果/位置/HIL语义边界沿用BATCH3，不重新声称独立成功检测。


## 2026-10-12 本轮完成与接续

[完整结果](RESULT.md)：新近期/HIL专用池从错误裸拼的0/0恢复到565/10；真实Actor新数据占比原13.265%/HIL8 29.619%。同起点846次更新×3seed，HIL8新HIL TRAIN/DEV MAE平均−2.29%/−1.44%，旧DEV人工MAE基本不变；原配方保持变差。两条HIL的Q1相对Actor偏好均在三个种子转正，但不能证明自主价值。86CPU回归及精确恢复/导出通过。代码fab2cc0/0b9dd4a已push；全部产物outputs/matched-learning-20261012，未修改生产Replay/权重或使用GPU。

下一步：算法只改teacher50→5的匹配诊断，固定采样/MC/LR/预算/数据；尚未启动。现场仅验收[固定Actor低频探索](../../../execution/tasks/2026-10-11-chunk-continuity/LOW_FREQUENCY.md)，新入口已安装/网页可见、Learner关闭，用户负责加载与运动。本轮研究权重不自动晋升，不称可持续Online学习已通过。


## 2026-10-13 从当前7281的teacher单因素续训完成

用户批准当前7281完整state起点、固定teacher文件，仅50→5。6组×846 CPU更新完成，Actor4063；两组真实batch逐数组一致、DEV0抽样，16采样/导入测试和149叶精确恢复/导出通过。teacher5新HIL DEV相对teacher50再改善1.61%，旧DEV人工均值相对7281+0.23%、p95+0.28%；通过预登记工程筛选，但新HIL各一集，不能认证自主Online收益。

[完整结果与新记录缺口](TEACHER_CONTINUATION.md)。现场新constant路由实际为evaluation，private Replay0；找到2个完整成功结果，口述约12/15未逐轮闭合，不能声称低频探索已在那些评测中启用。当前Actor3640/现场资产未变。下一步先闭合1集固定权重collection的真实trace/Replay，teacher5候选另作打包/冻结验收，不自动Online。所有产物outputs/teacher-continuation-20261013。


## 2026-10-13 online-review核验完成

[REVIEW_VALIDATION](REVIEW_VALIDATION.md)：修复旧review回报时基/接管前置零与历史实验配置记录，完成teacher5/clip10/min-Q三臂三seed单因素对照及旧AWBC保持复核。选择teacher5；clip10无触发、min-Q无新增纠正收益，AWBC保持不合格。8127/Actor4063独立包通过原生CPU加载/预算/恢复/导出；未登记或同步现场，未动生产Replay/默认。唯一下一步是按已有发布流程准备本包受控分批Online，先确认采集路由与首集真实入库；不重复旧实验。


## 2026-10-13 Teacher5现场交付完成
源码f4bae5c已push/同步；Teacher58127/Actor4063已实际暂停加载，50HzRTC/EMA、Replay3623、新预算0、发布0/录制idle。用户明确授权切换EXPO后完成，未自动运动。原生现场CPU恢复/输出及初始资产身份通过；pending启动重导出的字节差已验证参数完全相同。唯一下一步由用户首个完整采集Episode，然后核对真实入库/预算/批次/pending，不自动晋升、不声明自主收益。详见[TEACHER5_DELIVERY](TEACHER5_DELIVERY.md)，产物outputs/teacher5-delivery-20261013。
