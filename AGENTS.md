# 项目入口

先读 `/data/LFT-W02_data/jiaan/jiaan/agent-guide/AGENTS.md`，再读本项目 [README.md](README.md) 与 [迁移记录](docs/MIGRATION.md)。按用户当前任务执行，保留已有成果和其他对话的改动。

项目：rl-platform。目标：参考 FluxVLA 的模块化方式组织配置、数据／Replay、采样、学习、模型发布与评测；先接入 RLT，再按现有进度接入 EXPO-FT。

当前已开始业务迁移，以 docs/MIGRATION.md 的具体验收／切换记录为准；不得把目录存在或源码 clone 视为运行验证。跨项目问题按项目说明交给对应领域，证据与进展写回所属项目。

## 2026-09-30 按项目接管与并行对话

这是长期项目入口，不再处于“仅初始化”阶段。先读最新记录并核对代码/运行状态；历史旧路径、PID 和未完成描述不能当作当前事实。

- 负责：RLT / EXPO-FT 方法、Replay、采样、训练、Actor 发布、Session 与 RLT 在线执行；可分调研、代码/数据诊断、实时部署、异步加速专题。
- 深入阅读：README.md、docs/RUNBOOK.md、docs/EXPERIMENTS_20260930.md、docs/ANALYSIS.md、docs/MIGRATION.md。
- 实现入口：methods/openpi_rlt/cobot_adapter/、methods/openpi_rlt/experiments/、configs/deployment_models.json、integrations/cobot_runtime/；固定上游 third_party/openpi-rlt/。
- 当前事实：MC30 与 async_rtc20/30/40/50 为可选候选。2026-09-30 修复暂停旧时钟误报及录制 HTTP 占用推理预算，保留 Stage1 的现场重建通过；连续真机 50 Hz 与成功率仍待验收。
- 下一步：定位抖动、真实 deadline 分布和 Learner 争用；区分推理/发布/录制延迟。MC30、采样/奖励、RTC、频率分别做可回退对照，注明是否继续学习。真实 Replay 不作离线实验输出，EXPO-FT 保留独立算法和环境。
- 边界：共用 RTC/滤波交 VLA；页面恢复交 web；录制/mask 交 dagger；硬件交 control。RLT 不反向 import web Python，录制通过 HTTP。

用户会在同一项目开多个终端/对话并 fork。fork 不隔离工作树、GPU、端口、模型、Replay 或机器人。

1. 接管先读 git status/diff、git worktree list、实际进程/录制状态及 /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/coordination/ 下已有任务说明（存在时）。先说明范围和共享资源。仅要求“读取目录 MD，了解项目”时先汇报，不自动训练/重启或执行所有旧待办。
2. 并行任务各在上述 coordination 下维护一个可读主题名 MD，登记负责人/对话标识、时间、分支/worktree、基线提交、计划文件、GPU/端口/输出和状态；自己的说明自己更新，结束标记完成。临时协调信息不入业务 Git，有用结果写正式文档，不替别的任务认领/完成工作。
3. 调研默认只读；分析用独立输出/只读快照，不改生产 Replay/权重。并行代码改动用独立分支/worktree，放 scratch/rl-platform/<可读主题>/，先核对依赖根/环境，不为 worktree 改生产路径。不在别人工作树切分支、reset、clean、stash 或全量提交。
4. 同文件/接口交叉先明确归属，独立开发后审核合并和调用方；合并、push、发布串行。共享 main、环境和机器配置不是并行试验区。只提交本任务改动，不 force push，不静默覆盖；合并前重查远端及未提交变化。
5. Cobot 同时只有一个启停/部署负责人，跨项目共用 /data/LFT-W02_data/jiaan/jiaan/scratch/cobot-web/coordination/cobot-live.md 说明（存在时先读）。未明确接管时仅只读/离线工作，不因模型“暂停”就抢 GPU、重启服务或切数据目录。进程锁只提供互斥，不是运动授权；硬件重启/运动前核对现场条件。
6. 新算法、采样、RTC/异步/EMA/频率使用可选配置、独立实验输出及明确回退。研究可并行；生产模型默认/参数/Replay 修改与运行负责人协调。
7. A6000 开发验证、所属仓库提交 push 后按清单校验同步 Cobot；区分源码已同步与运行已切换。结果写所属项目及 guide/projects/rl-platform/README.md 事实摘要；不改 guide 治理、不提交/推送 guide Git。

主仓库 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform；现场副本 /home/agilex/jiaan/project/rl-platform。数据/权重通常引用 Getea1/jiaan/{data,model}，实际登记配置优先（如 NVMe Stage1 路径）；不擅自搬资产。
