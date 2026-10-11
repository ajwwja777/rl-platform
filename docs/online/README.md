# Online：Replay、采样、学习与 Actor 发布

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
