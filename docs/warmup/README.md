# Warmup：初始化、训练与候选选择

[全项目进度与经验](../README.md) · 更新：2026-10-11（文档整理，无新增模型能力验证）。

维护离线初始化数据、实际采样/目标、Actor/Critic 训练、checkpoint 选择与保持边界；Warmup 种子与 Online 延续身份分开。

## 现状与边界

- 2026-10-15 [realtime_expo先采Reference再Warmup](tasks/2026-10-15-reference-rollout-first/TASK.md)：纯专家7k只保留对照，不能代替含真实rollout的初始化。用户要求先收集新Reference数据，本次不再训练。

- 5k、rollout-only20k、experts+rollout20k 的选择依据见下方 9 月 24 日任务；初始固定 5k 不等于后续 Supported7000 或 HIL8 7281。
- 外部 Episode holdout 已反复参与选型，按 DEV 使用；没有因此获得独立 TEST。后续候选当前状态见 [Online](../online/README.md)。
- 本次未修改训练方式/比例、未重训；历史训练结果与每组配置以任务原文及 manifest 为准。

## 经验

统计 Episode、Replay 窗和实际 batch 三种比例，HIL/专家/成功标签可能重叠。随机 transition 验证不能替代 Episode 隔离；最小 loss、最大 Q 或末段 AUC 不足以选择可自主成功的模型。依据：9 月 24 日数据/配置复核、heldout 诊断及选择记录。

## 专题入口

- [WARMUP_HANDOFF.md](WARMUP_HANDOFF.md)
- [PLUG_INSERTION_R2_TRAINING.md](rlt/PLUG_INSERTION_R2_TRAINING.md)

## 任务

日期任务保留当时证据；归档不是重新验收。新任务按问题建立，跨模块只存一份正文。

- [2026-09-26-plug-v3-faithful-warmup-5k-online-release](tasks/2026-09-26-plug-v3-faithful-warmup-5k-online-release/TASK.md)
- [2026-09-24-plug-v3-warmup-selection](tasks/2026-09-24-plug-v3-warmup-selection/TASK.md)
- [2026-09-24-plug-v3-warmup-data-and-config-review](tasks/2026-09-24-plug-v3-warmup-data-and-config-review/TASK.md)
- [2026-09-24-plug-v3-episode-heldout-warmup-20k-diagnostics](tasks/2026-09-24-plug-v3-episode-heldout-warmup-20k-diagnostics/TASK.md)
- [2026-09-18-plug-v2-warmup-replay-repair](tasks/2026-09-18-plug-v2-warmup-replay-repair/TASK.md)
- [2026-09-18-plug-v2-warmup-data-readiness](tasks/2026-09-18-plug-v2-warmup-data-readiness/TASK.md)
- [2026-09-18-plug-v2-human-sampling-bc-q-diagnostics](tasks/2026-09-18-plug-v2-human-sampling-bc-q-diagnostics/TASK.md)
- [2026-09-18-plug-v2-formal-warmup](tasks/2026-09-18-plug-v2-formal-warmup/TASK.md)
- [2026-09-18-plug-v2-checkpoint-comparison](tasks/2026-09-18-plug-v2-checkpoint-comparison/TASK.md)
- [2026-09-18-plug-v2-actor-critic-mechanism-validation](tasks/2026-09-18-plug-v2-actor-critic-mechanism-validation/TASK.md)
- [2026-09-11-plug-offline-warmup-delivery](tasks/2026-09-11-plug-offline-warmup-delivery/TASK.md)
