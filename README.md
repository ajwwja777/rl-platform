# 统一 RL 实验平台

参考 FluxVLA 的模块化方式组织配置、数据／Replay、采样、学习、模型发布与评测；先接入 RLT，再按现有进度接入 EXPO-FT。

## 入口与位置

- 先读统一框架：`/data/LFT-W02_data/jiaan/jiaan/agent-guide/AGENTS.md`。
- A6000 主工作区：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`。
- 笔记本对话入口：`D:\Code\jiaan_workspace\rl-platform`。
- 自有独立仓库：`https://github.com/ajwwja777/rl-platform`（目标分支 `main`）。
- Cobot 目标部署位置：`/home/agilex/jiaan/project/rl-platform`，本轮尚未部署。
- 当前阶段：入口与仓库初始化；旧业务代码、环境、模型和数据尚未迁移，现有服务入口未切换。

## 负责什么

算法适配、奖励和标签语义、replay、actor／learner 生命周期、checkpoint、离线与在线实验协议。算法特有的目标和更新规则保持可追溯。

复用 vla-platform 的模型与评测能力、cobot-dagger 的数据语义以及 cobot-control 的设备接口。现有 EXPO-FT 和配套 OpenPI 仓库保留独立历史和环境；统一项目入口不要求合并算法源码或改写忠实复现基线。

## 机器与资产

A6000 负责主代码、Git、维护文档、主要开发验证环境、数据处理和离线评测；训练按资源需要在 A6000／已授权训练机进行。Cobot 只部署本项目现场实际需要的硬件、采集、推理、网页或维护组件，不复制仿真资产和完整训练环境。

Cobot 采集及评测数据统一规划在 `/home/agilex/jiaan/data/`。模型放所属项目的 `models/`（上游已有 `checkpoints/` 等目录时保留其源码布局，由配置明确实际权重位置）；同一资产跨项目引用，避免重复复制。现场服务日志、PID 和状态归实际负责项目；网页编排任务使用 `cobot-web/runtime/`；训练 checkpoint、配置和指标保留在所属项目 `outputs/<实验>/`。环境、模型、大数据与 runtime 不入 Git。

## 项目协作

Replay、奖励、learner 与在线更新由本项目负责；推理预处理交 vla-platform；硬件控制交 cobot-control；网页编排交 cobot-web，算法服务和存储由本项目排查。

先读本次任务涉及的依赖项目入口和接口说明，再修改相关边界；接口变更要记录受影响调用方与验证方式。常用项目：`cobot-control`、`cobot-dagger`、`vla-platform`、`rl-platform`、`cobot-web`，主工作区均在 `/data/LFT-W02_data/jiaan/jiaan/projects/`。需要专题对话时仍共享所属项目，不因此重复建立业务仓库。

## 下一步

先登记并验证当前 RLT warmup 5k 模型及现有评测的完整来源，做固定输入的离线加载对照；不启动新的在线学习。

旧位置、验收条件和切换／清理规则见迁移记录。

来源：2026-09-27 用户确认的项目划分、机器职责与逐批迁移方案；本轮范围仅初始化。

## 保留事项

现有 warmup 现场评测记录为 58 次有效结果、22 次成功（约 37.9%），是保留的历史基线，不是新框架本轮复测结果。EXPO-FT 环境已有准备，数据／动作适配及训练仍按原项目记录推进。

2026-09-27 归属更新：独立 ops 项目已取消；本次仅修正协作与 runtime 归属，不代表本项目旧业务资产已迁移。
