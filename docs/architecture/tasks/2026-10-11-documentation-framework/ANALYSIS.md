# 文档框架整理：依据与映射

## 决策与边界

用户先要求专题归整，后确认采用可复用的“实际模块 → tasks”框架，顶层一份进度与经验总览。本次按最终方案实施，不创建全局 analysis/history 任务区，不移动代码/outputs/模型/Replay。既有历史正文归模块，旧路径保留兼容跳转与原标题锚点；原有历史失效链接转为明确的原路径文字，避免假装可用。

原始 98 份 tracked MD 已保存在本任务 scratch/original-markdown.json；95 份专题正文迁移，AGENTS/项目 README 更新，integrations/README 不改。旧项目 README、包含另一对话最新追加的 guide 完整摘要在本目录保留追溯快照。报告原件不改。

所有正文只加日期/边界提示并调整相对链接，历史命令与参数不重写成当前事实。六个已失效旧项目链接保留原始地址与缺失说明；正文中的裸路径/命令仍按原文机器和工作目录理解。复杂任务新建 TASK/ANALYSIS/RESULT，历史单份报告用 TASK 保留，不虚构验收和执行进度。

## 逐文件归位清单

路径相对项目根；左侧仅兼容，右侧为正文。完整修改文件绝对路径见 [MODIFIED_FILES](MODIFIED_FILES.md)。

| 原路径 | 正文路径 |
|---|---|
| `docs/ANALYSIS.md` | `docs/online/DIAGNOSTICS.md` |
| `docs/ARCHITECTURE.md` | `docs/architecture/ARCHITECTURE.md` |
| `docs/DEPLOYMENT.md` | `docs/execution/DEPLOYMENT.md` |
| `docs/DIAGNOSIS_20260929.md` | `docs/online/tasks/2026-09-29-learning-diagnosis/TASK.md` |
| `docs/EXPERIMENTS_20260930.md` | `docs/online/tasks/2026-09-30-credit-and-execution-experiments/TASK.md` |
| `docs/FROZEN_LOCALIZATION.md` | `docs/execution/FROZEN_LOCALIZATION.md` |
| `docs/MIGRATION.md` | `docs/architecture/MIGRATION.md` |
| `docs/ROUNDWISE_ONLINE.md` | `docs/online/ROUNDWISE_ONLINE.md` |
| `docs/RUNBOOK.md` | `docs/execution/RUNBOOK.md` |
| `docs/SUPPORTED_ONLINE_DELIVERY.md` | `docs/online/SUPPORTED_ONLINE_DELIVERY.md` |
| `docs/audits/2026-10-07-online-admission-closure.md` | `docs/online/tasks/2026-10-07-online-admission-closure/TASK.md` |
| `docs/audits/2026-10-07-online-current-data-terminal-exposure.md` | `docs/online/tasks/2026-10-07-online-current-data-terminal-exposure/TASK.md` |
| `docs/audits/2026-10-07-online-rollout-strategy.md` | `docs/online/tasks/2026-10-07-online-rollout-strategy/TASK.md` |
| `docs/audits/2026-10-07-publication-preparation.md` | `docs/execution/tasks/2026-10-07-publication-preparation/TASK.md` |
| `docs/audits/2026-10-08-publication-feedback.md` | `docs/execution/tasks/2026-10-08-publication-feedback/TASK.md` |
| `docs/audits/2026-10-08-recurrent-publication-timeout.md` | `docs/execution/tasks/2026-10-08-recurrent-publication-timeout/TASK.md` |
| `docs/audits/2026-10-10-online-credit-repair.md` | `docs/online/tasks/2026-10-10-online-credit-repair/TASK.md` |
| `docs/history/README.md` | `docs/architecture/MIGRATION_CONTEXT.md` |
| `methods/openpi_rlt/COBOT_DEPLOYMENT_SOP.md` | `docs/execution/rlt/COBOT_DEPLOYMENT_SOP.md` |
| `methods/openpi_rlt/COBOT_GAP_ANALYSIS.md` | `docs/architecture/rlt/COBOT_GAP_ANALYSIS.md` |
| `methods/openpi_rlt/HPC_MIGRATION_MANIFEST.md` | `docs/architecture/rlt/HPC_MIGRATION_MANIFEST.md` |
| `methods/openpi_rlt/HPC_R1_TRAINING_REPORT.md` | `docs/stage1/rlt/HPC_R1_TRAINING_REPORT.md` |
| `methods/openpi_rlt/LOCAL_REPRODUCTION.md` | `docs/architecture/rlt/LOCAL_REPRODUCTION.md` |
| `methods/openpi_rlt/ONLINE_RL_INTERFACE.md` | `docs/online/rlt/ONLINE_RL_INTERFACE.md` |
| `methods/openpi_rlt/PLUG_INSERTION_R2_TRAINING.md` | `docs/warmup/rlt/PLUG_INSERTION_R2_TRAINING.md` |
| `methods/openpi_rlt/PLUG_V2_ROLLOUT_SOP.md` | `docs/execution/rlt/PLUG_V2_ROLLOUT_SOP.md` |
| `methods/openpi_rlt/PLUG_V2_TODO.md` | `docs/online/rlt/PLUG_V2_TODO.md` |
| `methods/openpi_rlt/UNIFIED_CONSOLE_SOP.md` | `docs/execution/rlt/UNIFIED_CONSOLE_SOP.md` |
| `methods/openpi_rlt/UNIFIED_CONSOLE_TODO.md` | `docs/execution/rlt/UNIFIED_CONSOLE_TODO.md` |
| `methods/openpi_rlt/UPSTREAM_MANIFEST.md` | `docs/architecture/rlt/UPSTREAM_MANIFEST.md` |
| `methods/openpi_rlt/WORKFLOW.md` | `docs/architecture/rlt/WORKFLOW.md` |
| `methods/openpi_rlt/audits/2026-09-10-collection-gate-hil-home.md` | `docs/execution/tasks/2026-09-10-collection-gate-hil-home/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-10-recorder-concurrency-and-status-io.md` | `docs/execution/tasks/2026-09-10-recorder-concurrency-and-status-io/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-10-rollout-quality-and-training-proposal.md` | `docs/online/tasks/2026-09-10-rollout-quality-and-training-proposal/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-11-online-rollout-cycle-delivery.md` | `docs/execution/tasks/2026-09-11-online-rollout-cycle-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-11-plug-offline-warmup-delivery.md` | `docs/warmup/tasks/2026-09-11-plug-offline-warmup-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-command-smoothing-demo.md` | `docs/execution/tasks/2026-09-14-command-smoothing-demo/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-continue-from5416.md` | `docs/online/tasks/2026-09-14-continue-from5416/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-daily-summary.md` | `docs/architecture/tasks/2026-09-14-daily-summary/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-online-precompile-segfault.md` | `docs/execution/tasks/2026-09-14-online-precompile-segfault/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-request-poll-race.md` | `docs/execution/tasks/2026-09-14-request-poll-race/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-training-action-contract.md` | `docs/online/tasks/2026-09-14-training-action-contract/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-14-xr1-smoothness-reference.md` | `docs/execution/tasks/2026-09-14-xr1-smoothness-reference/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-16-daily-summary.md` | `docs/architecture/tasks/2026-09-16-daily-summary/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-16-unified-console-lifecycle-delivery.md` | `docs/execution/tasks/2026-09-16-unified-console-lifecycle-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-17-18-daily-summary.md` | `docs/architecture/tasks/2026-09-17-18-daily-summary/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-17-evaluation-without-recording.md` | `docs/execution/tasks/2026-09-17-evaluation-without-recording/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-17-tracking-conflict-recovery.md` | `docs/architecture/tasks/2026-09-17-tracking-conflict-recovery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-actor-critic-mechanism-validation.md` | `docs/warmup/tasks/2026-09-18-plug-v2-actor-critic-mechanism-validation/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-checkpoint-comparison.md` | `docs/warmup/tasks/2026-09-18-plug-v2-checkpoint-comparison/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-demonstrations-conversion.md` | `docs/stage1/tasks/2026-09-18-plug-v2-demonstrations-conversion/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-fixed-rtc-corrective-deployment.md` | `docs/execution/tasks/2026-09-18-plug-v2-fixed-rtc-corrective-deployment/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-formal-warmup.md` | `docs/warmup/tasks/2026-09-18-plug-v2-formal-warmup/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-hpc-training-preparation.md` | `docs/architecture/tasks/2026-09-18-plug-v2-hpc-training-preparation/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-human-sampling-bc-q-diagnostics.md` | `docs/warmup/tasks/2026-09-18-plug-v2-human-sampling-bc-q-diagnostics/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery.md` | `docs/execution/tasks/2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-rtc-supervision-and-credit-contract.md` | `docs/online/tasks/2026-09-18-plug-v2-rtc-supervision-and-credit-contract/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-warmup-data-readiness.md` | `docs/warmup/tasks/2026-09-18-plug-v2-warmup-data-readiness/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-18-plug-v2-warmup-replay-repair.md` | `docs/warmup/tasks/2026-09-18-plug-v2-warmup-replay-repair/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-conditioning-credit-and-q-direction.md` | `docs/online/tasks/2026-09-19-conditioning-credit-and-q-direction/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-plug-v2-first-autonomous-success-and-runtime-r4.md` | `docs/execution/tasks/2026-09-19-plug-v2-first-autonomous-success-and-runtime-r4/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-plug-v2-post-success-instability-and-runtime-r5.md` | `docs/execution/tasks/2026-09-19-plug-v2-post-success-instability-and-runtime-r5/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-plug-v2-r5-canary-divergence-and-runtime-r6.md` | `docs/execution/tasks/2026-09-19-plug-v2-r5-canary-divergence-and-runtime-r6/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-rtc-decision-trace-and-startup-scheduling.md` | `docs/execution/tasks/2026-09-19-rtc-decision-trace-and-startup-scheduling/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-supported-online-delivery.md` | `docs/execution/tasks/2026-09-19-supported-online-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-19-upstream-reproduction-reset.md` | `docs/architecture/tasks/2026-09-19-upstream-reproduction-reset/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-20-causal-passive-online-release.md` | `docs/execution/tasks/2026-09-20-causal-passive-online-release/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-20-clean-training-review.md` | `docs/online/tasks/2026-09-20-clean-training-review/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-20-execution-contract-retraining.md` | `docs/online/tasks/2026-09-20-execution-contract-retraining/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-20-onsite-online-rl-r13-and-actor8445.md` | `docs/execution/tasks/2026-09-20-onsite-online-rl-r13-and-actor8445/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-20-restart-deadline-followup.md` | `docs/execution/tasks/2026-09-20-restart-deadline-followup/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-20-upstream-rtc-online-delivery.md` | `docs/execution/tasks/2026-09-20-upstream-rtc-online-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-21-online-learning-diagnosis-and-framework-review.md` | `docs/architecture/tasks/2026-09-21-online-learning-diagnosis-and-framework-review/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-22-plug-v2-value-recovery-console-and-online-delivery.md` | `docs/execution/tasks/2026-09-22-plug-v2-value-recovery-console-and-online-delivery/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-23-plug-v3-yyshadow-stage0-stage1.md` | `docs/stage1/tasks/2026-09-23-plug-v3-yyshadow-stage0-stage1/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-24-plug-v3-episode-heldout-warmup-20k-diagnostics.md` | `docs/warmup/tasks/2026-09-24-plug-v3-episode-heldout-warmup-20k-diagnostics/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-24-plug-v3-first-reference-and-console-controls.md` | `docs/execution/tasks/2026-09-24-plug-v3-first-reference-and-console-controls/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-24-plug-v3-warmup-data-and-config-review.md` | `docs/warmup/tasks/2026-09-24-plug-v3-warmup-data-and-config-review/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-24-plug-v3-warmup-selection.md` | `docs/warmup/tasks/2026-09-24-plug-v3-warmup-selection/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-complete-english-locale.md` | `docs/execution/tasks/2026-09-26-complete-english-locale/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-discard-without-records.md` | `docs/execution/tasks/2026-09-26-discard-without-records/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-fixed-collection-workspace.md` | `docs/execution/tasks/2026-09-26-fixed-collection-workspace/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-model-loading-text.md` | `docs/execution/tasks/2026-09-26-model-loading-text/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-panel-tool-buttons.md` | `docs/execution/tasks/2026-09-26-panel-tool-buttons/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-plug-v3-faithful-warmup-5k-online-release.md` | `docs/warmup/tasks/2026-09-26-plug-v3-faithful-warmup-5k-online-release/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-single-collection-page-and-keys.md` | `docs/execution/tasks/2026-09-26-single-collection-page-and-keys/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-26-unified-collection-controls.md` | `docs/execution/tasks/2026-09-26-unified-collection-controls/TASK.md` |
| `methods/openpi_rlt/audits/2026-09-27-getea-usb-offline-ui-recovery.md` | `docs/execution/tasks/2026-09-27-getea-usb-offline-ui-recovery/TASK.md` |
| `methods/openpi_rlt/plans/2026-09-17-bounded-online-rl-evaluation.md` | `docs/online/tasks/2026-09-17-bounded-online-rl-evaluation/TASK.md` |
| `methods/openpi_rlt/plans/2026-09-18-plug-v2-training-and-rtc.md` | `docs/online/tasks/2026-09-18-plug-v2-training-and-rtc/TASK.md` |
| `methods/openpi_rlt/plug_v2/README.md` | `docs/architecture/rlt/PLUG_V2.md` |
| `methods/openpi_rlt/plug_v3_yyshadow/DATA_TRAINING_SOP.md` | `docs/stage1/DATA_TRAINING_SOP.md` |
| `methods/openpi_rlt/plug_v3_yyshadow/README.md` | `docs/stage1/PLUG_V3_BASELINE.md` |
| `methods/openpi_rlt/plug_v3_yyshadow/SCENE_CHECKLIST.md` | `docs/stage1/SCENE_CHECKLIST.md` |
| `methods/openpi_rlt/plug_v3_yyshadow/WARMUP_HANDOFF.md` | `docs/warmup/WARMUP_HANDOFF.md` |

## 原有失效链接

本次未找到以下旧位置，没有猜测替代文件，也没有删除该引用的历史地址。

- `methods/openpi_rlt/audits/2026-09-14-daily-summary.md` → `../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-14-segmented-history-video.md`
- `methods/openpi_rlt/audits/2026-09-16-daily-summary.md` → `../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-16-latched-mode-readiness-fix.md`
- `methods/openpi_rlt/audits/2026-09-16-daily-summary.md` → `../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-16-session-proxy-reentry-fix.md`
- `methods/openpi_rlt/audits/2026-09-17-18-daily-summary.md` → `../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-17-platform-single-entry-and-cleanup.md`
- `methods/openpi_rlt/audits/2026-09-17-18-daily-summary.md` → `../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-17-task2-recovery-and-synchronized-homing.md`
- `methods/openpi_rlt/WORKFLOW.md` → `../../../proj-20260829-cobot-realworld-vla/platform/WORKSPACE_LAYOUT.md`

## 验证方法

本次只运行文档校验，不运行训练或现场测试：检查全部项目 Markdown 的相对文件链接/锚点、95 份正文按允许变换后的 SHA、465 份非 MD Git 文件 SHA、793 份 outputs 小报告/记录 SHA、另一对话 3 文件 SHA；git diff --check 与变更后缀检查。

原始基线、完整映射与机器校验回执：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/docs-organization-20261011/`；脚本为 `validate_docs.py`，命令 `python3 /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/docs-organization-20261011/validate_docs.py`。guide 链接在合并前按主项目目标映射到隔离 worktree 校验；合并后再次直接检查主目录。

验证边界：原始产物路径只做导航/小文件 SHA，不重跑算法、不重新认可历史结论；未遍历巨量权重数据，不以本次文件整理认证模型。没有连接 Cobot，没有 GPU、服务或机器人操作。另一对话研究草稿继续由其作者负责。

暂存后补检发现两处原有 Markdown 空白问题：LOCAL_REPRODUCTION 的双空格换行等价改为 `<br>`，SCENE_CHECKLIST 去掉末尾多余空行；内容校验显式逆变换复核原文 SHA。校验使用 `git diff HEAD --check` 包含已暂存的新文件，不只检查未暂存变更。
