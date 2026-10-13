# 执行：Session、时序、RTC 与现场交付

[全项目进度与经验](../README.md) · 更新：2026-10-11（文档整理，无新增模型能力验证）。

负责 RLT Session、推理/发布/反馈时序、trace 与训练准入接口及候选交付；共用 RTC/滤波归 VLA，录制归 dagger，网页恢复归 web，硬件归 control。

## 现状与边界

- 10 月 7–8 日已有发布准备前移、有界异步 trace、发送前纯反馈检查修复；各自的复现、回归和现场同步边界见下方任务。
- 用户报告 50Hz+RTC+EMA 较不抖，不据此认定自主收益或长期稳定已验收。代码已同步与当前进程已加载须分开核验；本次没有现场连接/启停/动作。
- 当前候选及分批学习边界引用 [Online](../online/README.md)，实际操作从 [RUNBOOK](RUNBOOK.md) 和对应候选交付记录进入，先确认机器/模型身份。

## 经验

控制超时要拆推理、准备、发布、反馈、录制和调度延迟，不能只看 Hz。记录工具本身可能阻塞控制；录制丢帧/中止不是任务自主失败。测试成功不替代真机 deadline 与跟踪验证。固定 Actor 对照执行方式，算法试验固定执行方式。来源：publication-preparation、recurrent-publication-timeout、publication-feedback 三个任务。

## 专题入口

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [FROZEN_LOCALIZATION.md](FROZEN_LOCALIZATION.md)
- [RUNBOOK.md](RUNBOOK.md)
- [COBOT_DEPLOYMENT_SOP.md](rlt/COBOT_DEPLOYMENT_SOP.md)
- [PLUG_V2_ROLLOUT_SOP.md](rlt/PLUG_V2_ROLLOUT_SOP.md)
- [UNIFIED_CONSOLE_SOP.md](rlt/UNIFIED_CONSOLE_SOP.md)
- [UNIFIED_CONSOLE_TODO.md](rlt/UNIFIED_CONSOLE_TODO.md)

## 任务

- [2026-10-11-chunk-continuity](tasks/2026-10-11-chunk-continuity/TASK.md)：实际RTC/50Hz/EMA已启用；RLT专用两级滤波候选离线降低命令加速度，但增加滞后，真机待验收。

日期任务保留当时证据；归档不是重新验收。新任务按问题建立，跨模块只存一份正文。

- [2026-10-08-recurrent-publication-timeout](tasks/2026-10-08-recurrent-publication-timeout/TASK.md)
- [2026-10-08-publication-feedback](tasks/2026-10-08-publication-feedback/TASK.md)
- [2026-10-07-publication-preparation](tasks/2026-10-07-publication-preparation/TASK.md)
- [2026-09-27-getea-usb-offline-ui-recovery](tasks/2026-09-27-getea-usb-offline-ui-recovery/TASK.md)
- [2026-09-26-unified-collection-controls](tasks/2026-09-26-unified-collection-controls/TASK.md)
- [2026-09-26-single-collection-page-and-keys](tasks/2026-09-26-single-collection-page-and-keys/TASK.md)
- [2026-09-26-panel-tool-buttons](tasks/2026-09-26-panel-tool-buttons/TASK.md)
- [2026-09-26-model-loading-text](tasks/2026-09-26-model-loading-text/TASK.md)
- [2026-09-26-fixed-collection-workspace](tasks/2026-09-26-fixed-collection-workspace/TASK.md)
- [2026-09-26-discard-without-records](tasks/2026-09-26-discard-without-records/TASK.md)
- [2026-09-26-complete-english-locale](tasks/2026-09-26-complete-english-locale/TASK.md)
- [2026-09-24-plug-v3-first-reference-and-console-controls](tasks/2026-09-24-plug-v3-first-reference-and-console-controls/TASK.md)
- [2026-09-22-plug-v2-value-recovery-console-and-online-delivery](tasks/2026-09-22-plug-v2-value-recovery-console-and-online-delivery/TASK.md)
- [2026-09-20-upstream-rtc-online-delivery](tasks/2026-09-20-upstream-rtc-online-delivery/TASK.md)
- [2026-09-20-restart-deadline-followup](tasks/2026-09-20-restart-deadline-followup/TASK.md)
- [2026-09-20-onsite-online-rl-r13-and-actor8445](tasks/2026-09-20-onsite-online-rl-r13-and-actor8445/TASK.md)
- [2026-09-20-causal-passive-online-release](tasks/2026-09-20-causal-passive-online-release/TASK.md)
- [2026-09-19-supported-online-delivery](tasks/2026-09-19-supported-online-delivery/TASK.md)
- [2026-09-19-rtc-decision-trace-and-startup-scheduling](tasks/2026-09-19-rtc-decision-trace-and-startup-scheduling/TASK.md)
- [2026-09-19-plug-v2-r5-canary-divergence-and-runtime-r6](tasks/2026-09-19-plug-v2-r5-canary-divergence-and-runtime-r6/TASK.md)
- [2026-09-19-plug-v2-post-success-instability-and-runtime-r5](tasks/2026-09-19-plug-v2-post-success-instability-and-runtime-r5/TASK.md)
- [2026-09-19-plug-v2-first-autonomous-success-and-runtime-r4](tasks/2026-09-19-plug-v2-first-autonomous-success-and-runtime-r4/TASK.md)
- [2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery](tasks/2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery/TASK.md)
- [2026-09-18-plug-v2-fixed-rtc-corrective-deployment](tasks/2026-09-18-plug-v2-fixed-rtc-corrective-deployment/TASK.md)
- [2026-09-17-evaluation-without-recording](tasks/2026-09-17-evaluation-without-recording/TASK.md)
- [2026-09-16-unified-console-lifecycle-delivery](tasks/2026-09-16-unified-console-lifecycle-delivery/TASK.md)
- [2026-09-14-xr1-smoothness-reference](tasks/2026-09-14-xr1-smoothness-reference/TASK.md)
- [2026-09-14-request-poll-race](tasks/2026-09-14-request-poll-race/TASK.md)
- [2026-09-14-online-precompile-segfault](tasks/2026-09-14-online-precompile-segfault/TASK.md)
- [2026-09-14-command-smoothing-demo](tasks/2026-09-14-command-smoothing-demo/TASK.md)
- [2026-09-11-online-rollout-cycle-delivery](tasks/2026-09-11-online-rollout-cycle-delivery/TASK.md)
- [2026-09-10-recorder-concurrency-and-status-io](tasks/2026-09-10-recorder-concurrency-and-status-io/TASK.md)
- [2026-09-10-collection-gate-hil-home](tasks/2026-09-10-collection-gate-hil-home/TASK.md)

## 既有产物入口

以下仅核对路径并建立索引，正文/图/manifest 保留原位置；目录存在不代表本次复核了结论。跨模块产物以主要问题归属，按需互相引用。

- [acceleration-hil-ablation-20261006](../../outputs/acceleration-hil-ablation-20261006/REPORT.md)
- [control-publish-timeout-20261007](../../outputs/control-publish-timeout-20261007/REPORT.md)
- [online-collection-readiness-20261008](../../outputs/online-collection-readiness-20261008)（目录）
- [position-jitter-diagnosis-20261006](../../outputs/position-jitter-diagnosis-20261006/REPORT.md)
- [preonline-field-verification-20261005](../../outputs/preonline-field-verification-20261005/RESULT.md)
- [preonline-gpu-20261006](../../outputs/preonline-gpu-20261006/REPORT.md)
- [proposal-execution-diagnosis-20261005](../../outputs/proposal-execution-diagnosis-20261005/REPORT.md)
- [publication-feedback-20261008](../../outputs/publication-feedback-20261008/REPORT.md)
- [publication-preparation-20261007](../../outputs/publication-preparation-20261007/REPORT.md)
- [recurrent-publication-timeout-20261008](../../outputs/recurrent-publication-timeout-20261008/REPORT.md)
- [startup-optimization-20260929](../../outputs/startup-optimization-20260929)（目录）
- [web-deployment-handoff-20261006](../../outputs/web-deployment-handoff-20261006)（目录）


## 2026-10-12 自选录制目录的校验范围

网页已允许 Getea1 盘下自选，但 RLT 原校验只认识主 DATA／legacy 根；profile_storage 现消费同一 COBOT_ADDITIONAL_DATA_ROOTS，保留绝对路径、别名迁移和符号链接边界。仅存储契约变更，模型固定探索目录与默认根不变。已提交／同步，正式网页重载后 xiqing HTTP browse/prepare、新子目录创建和现场 RLT CPU 校验通过，无模型／硬件启停、录制或资产修改。详情：[任务与证据](tasks/2026-10-12-custom-recording-roots/TASK.md)。


## 2026-10-12 实践更新

探索的时间变化与滤波分开验收：新episode_constant候选固定每集偏移并平滑进入，不叠加EMA；开环改善不代表接触成功。已安装且网页可见，使用/停止/回退见[低频探索](tasks/2026-10-11-chunk-continuity/LOW_FREQUENCY.md)。

## 2026-10-13 EXPO 左臂网页接入

独立EXPO2000/左臂7D/两腕/10Hz RTC-SFT已通过原生GPU推理与Web暂停加载，默认不训练。复用现有受管理模型采集/评测、HIL及标签入口，接管使旧动作失效；现场Start/HIL/归位尚待用户短轮验收。记录：[任务与证据](tasks/2026-10-13-expo-web/TASK.md)。
