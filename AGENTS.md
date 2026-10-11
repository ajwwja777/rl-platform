# 项目入口

先读 `/data/LFT-W02_data/jiaan/jiaan/agent-guide/AGENTS.md`，再读本项目 [README.md](README.md) 与 [进度与经验](docs/README.md) 和 [迁移记录](docs/architecture/MIGRATION.md)。按用户当前任务执行，保留已有成果和其他对话的改动。

项目：rl-platform。目标：参考 FluxVLA 的模块化方式组织配置、数据／Replay、采样、学习、模型发布与评测；先接入 RLT，再按现有进度接入 EXPO-FT。

当前已开始业务迁移，以 docs/architecture/MIGRATION.md 的具体验收／切换记录为准；不得把目录存在或源码 clone 视为运行验证。跨项目问题按项目说明交给对应领域，证据与进展写回所属项目。

## 模型权重存放

- 需要加载推理的权重放本机 NVMe：`/home/agilex/jiaan/data/<模型族>/...`，目录结构与 USB 上一致（如 `data/rlt/plug_insertion/reference_4999`）。USB 盘 `/media/agilex/Getea1`（NTFS/FUSE、机械盘）只作归档，原件不删。原因：15 GB 的 Stage-1 从 USB 恢复约 385–424 s，本机约 7 s（2026-09-29 冷缓存实测）。
- 保留原始 float32，不转换精度；只复制当前在用的 checkpoint。复制后逐文件 sha256 与原件比对一致再切换配置。
- RLT session 运行中不切换路径（learner 在写 USB 的 `online/`）。改了 Stage-1 路径后，先 `rlt_v3_down.sh` 再 `rlt_v3_up.sh`，否则模型服务复用检查会因路径不一致拒绝启动。
- plug_v3：manifest 的 `stage1_root` 指向本机 Stage-1，在线 RL 的 actor/learner/warmup/replay 仍在 USB。

来源：2026-09-30 同步校验发现现场另一对话已有此约定，原文合回 A6000；本批不搬迁资产。

## 2026-09-30 按项目接管与并行对话

这是长期项目入口，不再处于“仅初始化”阶段。先读最新记录并核对代码/运行状态；历史旧路径、PID 和未完成描述不能当作当前事实。

- 负责：RLT / EXPO-FT 方法、Replay、采样、训练、Actor 发布、Session 与 RLT 在线执行；可分调研、代码/数据诊断、实时部署、异步加速专题。
- 深入阅读：docs/README.md → 对应模块 README → 本任务 TASK/ANALYSIS/RESULT；操作见 docs/execution/RUNBOOK.md，迁移见 docs/architecture/MIGRATION.md。
- 实现入口：methods/openpi_rlt/cobot_adapter/、methods/openpi_rlt/experiments/、configs/deployment_models.json、integrations/cobot_runtime/；固定上游 third_party/openpi-rlt/。
- 当前事实与待办统一维护在 docs/README.md；模块 README 记录职责、稳定认识及任务入口。接管时按证据日期和实际状态核对，不沿用旧摘要。
- 研究与执行分开验证；可选方案须有独立输出与回退。具体下一步维护在对应 TASK.md；不自动执行历史待办。
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

## 按模块组织文档与自主推进（2026-10-11）

用户已确认本框架。代码、配置、资产和 outputs 布局不变；文档在 docs/{architecture,stage1,warmup,online,execution}/ 内按实际职责组织。docs/README.md 是唯一全项目进度与关键经验总览；各模块 README 保存本模块职责、现状、可复用经验和任务链接。旧路径兼容页不再添加正文。

- 问题归主要模块的 tasks/YYYY-MM-DD-问题名称/；跨模块只设一个主任务，其他模块引用。小任务只需 TASK.md，复杂后再拆 ANALYSIS.md 和 RESULT.md，不建立空模板目录或全局任务池。
- TASK 写问题、验收标准、允许/禁止范围、资源预算、停止条件、基线/证据入口、负责人/worktree、状态和唯一下一步。恢复对话先读任务进度，再核对实际 Git/产物；不要重复已完成实验。
- 自主推进：核对事实 → 提出可证伪假设 → 选择信息价值高的最小验证 → 隔离修改/实验 → 检查回归和副作用 → 达标交付或明确失败/证据不足。已授权且必要的步骤自行推进；受阻记录缺口并继续独立工作，不越过资源、生产发布和现场运动边界。
- ANALYSIS 记录数据身份、代码/配置、命令、产物、对照及验证边界，区分事实/假设。RESULT 记录修改、验证结果、使用/回退和未解决问题；不能把测试通过、文件存在或代理指标改善当作模型能力通过。
- RLT 结论用已验证/失败/证据不足；数据划分区分 TRAIN、重复选型 DEV、独立 TEST，自主/辅助成功分开，按完整 Episode 计算不确定性。夹爪/左臂固定的任务不以非活动维预测误差声称物理退化。Q 范围按实际 reward/discount 推导。
- 任意 agent 均可维护所属任务；各自登记 coordination、独立 worktree，公共入口与模块摘要合并时重查 diff，串行合并/push。不得把其他 agent 未合并草稿的假设升级为正式结论，不替其关闭任务。
- 完成后将有依据的“现象 → 原因/边界 → 解决办法 → 证据”简写到模块 README；仅关键跨模块经验进入 docs/README.md。guide/projects/rl-platform/README.md 只同步摘要和入口，guide Git 不在本项目提交/推送。

通用范本：`/data/LFT-W02_data/jiaan/jiaan/agent-guide/pipelines/project/DOCUMENTATION.md`。本次整理记录：docs/architecture/tasks/2026-10-11-documentation-framework/TASK.md。
