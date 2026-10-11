> 归档/专题正文，2026-10-11 从 `methods/openpi_rlt/plug_v2/README.md` 归位。下文保留原记录的时间与验证边界，不代表当前运行状态。最新结论见 [模块入口](../README.md)。文内命令仍按原文指定的机器与工作目录执行。

# plug_v2执行来源

本目录准备代码是RTC训练前历史快照，不能作为生产入口。完整代码在执行服务器自己的 methods/openpi_rlt/plug_v2，固定上游c1e40ac，不改upstream、不复制完整代码/模型到框架。

HPC /data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl
Cobot /media/agilex/Getea1/jiaan/projects/rlt

唯一入口cobot-platform/scripts，模型manifest projects/rlt/deployments/plug_v2/manifest.json。SHA来源在execution_sources.json（交付快照），[操作](../../execution/rlt/PLUG_V2_ROLLOUT_SOP.md)、[证据](../../execution/tasks/2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery/TASK.md)。

2026-09-19离线诊断源码/证据快照：`execution_sources_20260919.json`。旧execution_sources.json保留9月18日交付事实，不能视为当前发布资格。
