# Online：Replay、采样、学习与 Actor 发布

2026-10-20 EXPO?SFT2000 532 HPC4?[????](tasks/2026-10-20-expo-532-hpc/STATUS.md)?4H100?????partial/checkpoint?????Actor???Cobot???????Web??????Actor0/Replay3301/online0??????????????realtime-expo-ft-532-hpc4?10???5???????????????????????????63?64?/20Critic+1VLA?????????????????740494??2026-10-20 22:34:20????????????????????

2026-10-18 EXPO：[HPC四卡验收与释放](tasks/2026-10-18-expo-hpc4/STATUS.md)。生产Online98完整保留，4×H100隔离更新/恢复、Cobot返回Actor与正式HTTP全量下载通过；批次3/5重启不丢，5/5才学习。稳定20Critic+1VLA为60.5秒，四卡不等于线性提速；5回合全周期估计30–40分钟。已按用户要求释放734617，恢复入口支持新作业号，物理rollout/HIL与收益仍待现场验收。

2026-10-14 RLT：[Teacher5 8234离线接续检查](tasks/2026-10-11-online-learning-evidence/TEACHER5_PREONLINE_8234.md)：两套重复DEV保持门槛通过、149状态叶/双RNG/8vs4+4恢复精确、64输入导出差1.19e-7、14回归通过。Actor4117独立接续包已备，现场未晋升，EXPO保持；新DEV min-Q残差和单集HIL偏好仍不充分，不能声称自主收益。下次受控Online前仅沿既有切换做现场身份核验。

2026-10-14 EXPO：[独立edit0.01 Online交付](tasks/2026-10-14-expo-native-sync/EDIT001_ONLINE.md)完成Actor/Learner一致尺度、独立Replay/快照身份及LFT2 GPU1正式暂停加载。完整20Critic/1VLA与恢复、本地更新后Actor和60CPU通过；修改动作编辑尺度必须同时修改训练目标/损失配置，并隔离身份。并发状态落盘采用唯一临时文件，避免收尾与更新线程竞争。真实rollout/HIL收益仍待验收。

2026-10-14 RLT：[Teacher5首批真实Online审计](tasks/2026-10-11-online-learning-evidence/TEACHER5_FIRST_ONLINE_BATCH.md)：恢复后3集2成功/1失败、89条入库/89更新/45Actor，最新8234/4117未发布；执行仍8127/4063。新增3集实际Critic/Actor采样4.55%/4.10%，新TRAIN回报残差下降，重复DEV平均拟合保持；无新Actor自主增益证明。用户说明抓持相对位置有变化；现场随后已交EXPO，数据/结论分开。

2026-10-14 EXPO：[Cobot本地Actor / LFT1 Learner经LFT2中转](tasks/2026-10-14-expo-native-sync/TASK.md)正式Web暂停加载交付完成：两端ready/version0一致、25seed/3301帧、online0、Session启停/目录/健康检查通过，发布0；用户已结束Teacher5。原生20Critic/1VLA与三机传输隔离验证通过；真实rollout/HIL与成功率仍由操作员验收。

[全项目进度与经验](../README.md) · 更新：2026-10-11（文档整理，无新增模型能力验证）。

追溯录制→准入→Replay→实际批次→更新→pending/served 身份；维护训练目标、HIL、采样与候选离线验证。

## 现状与边界

- 最近正式证据为 [2026-10-10 HIL8 研究/交付](tasks/2026-10-10-online-credit-repair/TASK.md)，不是今日运行查询。候选 `supported_hil8_clip_20261010` 初始 Learner7281/Actor3640；可选冻结/分批 Online，pending 不自动晋升。
- 已验证（既有实验）：3 个采样种子，新单条 HIL TRAIN 六关节拟合改善 5.32–6.48%，20 条重复 DEV 在预设保持容差内；接管截断、Critic 失败末端配额有副作用，未采用。不能把改善百分比当物理插入精度收益。
- 证据不足：失败终点 Q 残差、独立 TEST/位置 OOD、自主持续收益。接续见 [当前问题](tasks/2026-10-11-online-learning-evidence/TASK.md)。本次没有新模型训练/发布。
- 另一对话研究经验尚未合并：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/online-review-20261011/rl-platform/docs/RESEARCH_LESSONS.md`，coordination 同名。保留其工作树；其中假设尚需与正式 matched 实验证据核对，不视为正式结论。未来作者按本模块归位，避免重建全局经验文件。

## 经验

HIL 标签必须保留；删标签不会消除人工救回的 credit 路径。Actor Q1 与 TD min-Q 分开，同状态重算提议不冒充历史真实提议。Q 范围由实际 reward/discount 推导。真实 batch 身份、采样 RNG、完整优化器/target 和预算一起恢复；步数相同不能证明权重相同。来源：[HIL8 任务](tasks/2026-10-10-online-credit-repair/TASK.md)。

## 专题入口

- [DIAGNOSTICS.md](DIAGNOSTICS.md)
- [ROUNDWISE_ONLINE.md](ROUNDWISE_ONLINE.md)
- [SUPPORTED_ONLINE_DELIVERY.md](SUPPORTED_ONLINE_DELIVERY.md)
- [ONLINE_RL_INTERFACE.md](rlt/ONLINE_RL_INTERFACE.md)
- [PLUG_V2_TODO.md](rlt/PLUG_V2_TODO.md)

## 任务

- [2026-10-11 固定 HIL8 探索采集](tasks/2026-10-11-exploration-collection/TASK.md)：A6000 101项/现场52项 CPU 检查通过；新入口已同步，网页可见；独立采集、禁用 Learner；不等于自主能力通过。

日期任务保留当时证据；归档不是重新验收。新任务按问题建立，跨模块只存一份正文。

- [2026-10-11-online-learning-evidence](tasks/2026-10-11-online-learning-evidence/TASK.md)
- [2026-10-10-online-credit-repair](tasks/2026-10-10-online-credit-repair/TASK.md)
- [2026-10-07-online-rollout-strategy](tasks/2026-10-07-online-rollout-strategy/TASK.md)
- [2026-10-07-online-current-data-terminal-exposure](tasks/2026-10-07-online-current-data-terminal-exposure/TASK.md)
- [2026-10-07-online-admission-closure](tasks/2026-10-07-online-admission-closure/TASK.md)
- [2026-09-30-credit-and-execution-experiments](tasks/2026-09-30-credit-and-execution-experiments/TASK.md)
- [2026-09-29-learning-diagnosis](tasks/2026-09-29-learning-diagnosis/TASK.md)
- [2026-09-20-execution-contract-retraining](tasks/2026-09-20-execution-contract-retraining/TASK.md)
- [2026-09-20-clean-training-review](tasks/2026-09-20-clean-training-review/TASK.md)
- [2026-09-19-conditioning-credit-and-q-direction](tasks/2026-09-19-conditioning-credit-and-q-direction/TASK.md)
- [2026-09-18-plug-v2-training-and-rtc](tasks/2026-09-18-plug-v2-training-and-rtc/TASK.md)
- [2026-09-18-plug-v2-rtc-supervision-and-credit-contract](tasks/2026-09-18-plug-v2-rtc-supervision-and-credit-contract/TASK.md)
- [2026-09-17-bounded-online-rl-evaluation](tasks/2026-09-17-bounded-online-rl-evaluation/TASK.md)
- [2026-09-14-training-action-contract](tasks/2026-09-14-training-action-contract/TASK.md)
- [2026-09-14-continue-from5416](tasks/2026-09-14-continue-from5416/TASK.md)
- [2026-09-10-rollout-quality-and-training-proposal](tasks/2026-09-10-rollout-quality-and-training-proposal/TASK.md)

## 既有产物入口

以下仅核对路径并建立索引，正文/图/manifest 保留原位置；目录存在不代表本次复核了结论。跨模块产物以主要问题归属，按需互相引用。

- [critic-policy-bootstrap-20261006](../../outputs/critic-policy-bootstrap-20261006/REPORT.md)
- [erlt-reference-review-20261006](../../outputs/erlt-reference-review-20261006/REPORT.md)
- [gripper-update-diagnosis-20261005](../../outputs/gripper-update-diagnosis-20261005/REPORT.md)
- [hil-command-audit-20261005](../../outputs/hil-command-audit-20261005/conclusion.txt)
- [hil-target-contract-20261005](../../outputs/hil-target-contract-20261005/REPORT.md)
- [model-repair-20261005](../../outputs/model-repair-20261005/progress.json)
- [offline-model-selection-20261005](../../outputs/offline-model-selection-20261005/conclusion.txt)
- [online-admission-20261007](../../outputs/online-admission-20261007/REPORT.md)
- [online-admission-closure-20261007](../../outputs/online-admission-closure-20261007/REPORT.md)
- [online-credit-repair-20261010](../../outputs/online-credit-repair-20261010/REPORT.md)
- [online-first-five-20261007](../../outputs/online-first-five-20261007/REPORT.md)
- [online-learning-signals-20261007](../../outputs/online-learning-signals-20261007/REPORT.md)
- [online-provenance-audit-20261005](../../outputs/online-provenance-audit-20261005/REPORT.md)
- [online-review-20261011](../../outputs/online-review-20261011)（目录）
- [online-rollout-strategy-20261007](../../outputs/online-rollout-strategy-20261007/REPORT.md)
- [online-sampling-20261010](../../outputs/online-sampling-20261010/REPORT.txt)
- [preonline-delivery-package-20261005](../../outputs/preonline-delivery-package-20261005/REPORT.md)
- [preonline-verification-20261005](../../outputs/preonline-verification-20261005/progress.json)
- [q-guidance-localization-20261005](../../outputs/q-guidance-localization-20261005/progress.json)
- [recent-sampling-model-study-20261005](../../outputs/recent-sampling-model-study-20261005/REPORT.md)
- [rlt-diagnosis-20260929](../../outputs/rlt-diagnosis-20260929)（目录）
- [rlt-diagnosis-20260930](../../outputs/rlt-diagnosis-20260930)（目录）
- [roundwise-online-review-20261006](../../outputs/roundwise-online-review-20261006/REPORT.md)
- [supported-online-20261006](../../outputs/supported-online-20261006/REPORT.md)
- [supported-online-field-delivery-20261006](../../outputs/supported-online-field-delivery-20261006/REPORT.md)
- [terminal-exposure-20261007](../../outputs/terminal-exposure-20261007/REPORT.md)


2026-10-11 采集经验：[固定探索7轮核对](tasks/2026-10-11-exploration-collection/ANALYSIS.md)。模型专用目录必须同时登记网页选择、实际录制与历史显示；已同步自动匹配，77项前端/15项RLT CPU回归通过。单臂任务的通用双臂HDF5 action可能全NaN，应读取右臂coordinator字段及单臂valid_mask，不跨缺口拼接，也不能据此推断独立Replay无效。Learner禁用时Replay增长不代表发生学习。


## 2026-10-12 实践更新

独立采集journal并入旧池必须用带源身份的顺序映射；新数据小号会漏出按ID近期池。实际batch而非静态buffer比例决定训练曝光。2配方3seed匹配结果见[当前任务结果](tasks/2026-10-11-online-learning-evidence/RESULT.md)，不据局部拟合放行自主Online。

## 2026-10-13 EXPO 单卡验证

xiqing 原生 RealTimeEXPOFTLearner 在 Cobot4090 完整推理通过；batch2/UTD1 的完整更新在默认及 platform 分配器均 OOM。原 SFT2000 已恢复暂停，在线接入未完成。详见 [实测记录](tasks/2026-10-13-expo-online/TASK.md)。


2026-10-13：[handoff验证与teacher5候选](tasks/2026-10-11-online-learning-evidence/REVIEW_VALIDATION.md)。旧回报分析错误已修复；clip10/min-Q未通过增益筛选、AWBC有保持代价；8127/4063原生候选软件验证通过，现场未注册。实际动作价值和自主收益不能由状态AUC/拟合误差替代。

## 2026-10-13 EXPO Web 在线学习

原生完整更新因显存不足失败后，单卡适配通过 buffer donation、未参与损失的目标 Actor CPU EMA 与分块保存，完成 batch4 连续更新、检查点恢复、模拟 rollout/HIL 回合和恢复续训。原生损失/可训练范围保留；回合内固定策略，回合结束标注后同步更新，未知/放弃不入 Replay。现场切换状态及边界见 [交付记录](tasks/2026-10-13-expo-online/RESULT.md)。真实成功率未验收。

2026-10-13交付：[Teacher5现场交付](tasks/2026-10-11-online-learning-evidence/TEACHER5_DELIVERY.md)已实际暂停加载8127/4063、3623Replay、新预算0；独立seed/runtime、staged与每50步存档。原生启动重导出pending可能改变pickle字节，核对参数树/版本/config后再判断权重改变；served和训练state仍逐SHA保持。首轮真实Online采集入库/学习待验证。

2026-10-14 EXPO抖动：[配对诊断](tasks/2026-10-14-expo-native-sync/JITTER.md)显示version0编辑扰动0.1是录制窗口内动作粗糙度主要增量；0.01相同RTC5/滤波限幅下二阶差分降约79%。独立本地低扰动预览已登记/HTTP验证，无Learner；真机待验收，原Online未改，LFT1 GPU占用阻止其加载。

2026-10-15 EXPO batch5 Web readiness regression fixed: local Actor and remote Learner validation separated, role/identity checks retained. 188 CPU tests and paused Web load passed; Actor/Learner11, dual GPU, batch0/5. See [batch5 delivery](tasks/2026-10-14-expo-native-sync/BATCH5_DUAL_GPU.md).

2026-10-15：[RLT四图监控](tasks/2026-10-15-rlt-monitoring/TASK.md)新增独立日志读取、Online/Warmup/评测隔离、实际回合自主成功率和CPU后台固定Critic动作Q1；源代码已同步，Web激活等待现场暂停。不得用训练批次actor_q代替单轮动作Q，也不以低loss证明成功率改善。

2026-10-19：用户指定排除 EXPO 失败 episode16（2b56b902），当时 Critic/VLA 均0；edit005 独立 run 已备份并撤回118条 Replay 与1轮热身计数，保留成功17的262条。两端 journal 标记 training_eligible=false，保留 accepted 回执防重复恢复；正式暂停重载验证 Replay3563、有效轮数1、热身剩余9、batch0/5、Actor0。原始样本和 Actor 共521文件SHA未变，Session未启动。证据：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-episode16-exclusion-20261019/{repair,acceptance}.json`。这是指定回合的维护，不改变网页删除行为。

2026-10-19：按用户要求删除最新 EXPO rollout episode23（6aafaa86，267条，失败），原始网页11文件经SHA核验备份后使用正式DELETE接口移除；Replay与计数同步排除。重载验收：Replay8522、在线5221、有效轮数32、batch0/5、credit16，Actor121/Critic2420/checkpoint6未变，暂停且Session未启动。证据 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-latest-exclusion-20261019/acceptance.json`；仅处理指定回合，未改变网页删除默认行为。
2026-10-19：[EXPO demo50](tasks/2026-10-19-expo-demo50/TASK.md)新增逐Critic小批次32示教/32在线的可选采样，141 CPU及原生Replay gather通过；现场启用另记。

2026-10-19：新增独立 EXPO SFT2000 50/30/20 Online 模式；149 CPU及原生Replay采样验证通过，未启动现场新训练。详情 tasks/2026-10-19-expo-532/TASK.md。
