# Stage1：示范、训练输入与 Reference

2026-10-15：[realtime_expo左臂RLT训练](tasks/2026-10-15-realtime-expo-rlt/TASK.md)。独立数据/输入准备完成，保持旧右臂资产不变；GPU训练等待双卡，不把CPU预检当训练完成。

[全项目进度与经验](../README.md) · 更新：2026-10-11（文档整理，无新增模型能力验证）。

追溯原始示范、转换/质检、真实加载器、归一化、Stage1 checkpoint 和表征；与 Warmup/Online 的输入合同衔接。

2026-10-14：[全量历史证据复核与候选时间合同](tasks/2026-10-14-evidence-reconciliation/RESULT.md)完成140缓存校验和三权重CPU对照。134条是分析图训练内前向；服务路径仅3集，不能混称全量服务认证。Teacher5保持原合同拟合略优，单独切20Hz Reference变差约20.75%，未作切换。

## 现状与边界

- 历史结构/训练入口见 [数据训练 SOP](DATA_TRAINING_SOP.md) 与 [Stage0/Stage1 审计](tasks/2026-09-23-plug-v3-yyshadow-stage0-stage1/TASK.md)。这是历史证据，本次没有重训或复跑模型。
- 已有完整链审计指出转换清单 120/14 不等于实际训练隔离，134 条均参加训练；这 14 条不能当独立 TEST。证据入口为下方 full-chain-audit 及后续 HIL8 审计引用。
- 证据不足：示范语义正确性、目标位置 OOD、自主能力和 RL Token 对 Critic 的可用性，不能由结构检查或重建 loss 代替。

## 经验

结构通过与语义完成分开；采样时基、H50/C10、真实动作与 Reference、原始图像变换都要追到实际加载路径。用户确认夹爪/左臂固定，活动能力诊断聚焦六关节。完整输入身份和原生数值闭环记录见下方 input/stage1 产物。

## 专题入口

- [DATA_TRAINING_SOP.md](DATA_TRAINING_SOP.md)
- [PLUG_V3_BASELINE.md](PLUG_V3_BASELINE.md)
- [SCENE_CHECKLIST.md](SCENE_CHECKLIST.md)
- [HPC_R1_TRAINING_REPORT.md](rlt/HPC_R1_TRAINING_REPORT.md)

## 任务

日期任务保留当时证据；归档不是重新验收。新任务按问题建立，跨模块只存一份正文。

- [2026-09-23-plug-v3-yyshadow-stage0-stage1](tasks/2026-09-23-plug-v3-yyshadow-stage0-stage1/TASK.md)
- [2026-09-18-plug-v2-demonstrations-conversion](tasks/2026-09-18-plug-v2-demonstrations-conversion/TASK.md)

## 既有产物入口

以下仅核对路径并建立索引，正文/图/manifest 保留原位置；目录存在不代表本次复核了结论。跨模块产物以主要问题归属，按需互相引用。

- [full-chain-audit-20261001](../../outputs/full-chain-audit-20261001/progress.json)
- [input-image-identity-20261005](../../outputs/input-image-identity-20261005/REPORT.md)
- [input-learning-chain-20261005](../../outputs/input-learning-chain-20261005/REPORT.md)
- [input-snapshot-verification-20261005](../../outputs/input-snapshot-verification-20261005/REPORT.md)
- [stage1-actor-chain-20261005](../../outputs/stage1-actor-chain-20261005/REPORT.md)
- [stage1-input-snapshot-model-20261005](../../outputs/stage1-input-snapshot-model-20261005/REPORT.md)

- [offline-model-selection-20261005 全量历史模型诊断](../../outputs/offline-model-selection-20261005/index.html)
