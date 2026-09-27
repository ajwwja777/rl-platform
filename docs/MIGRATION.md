# 统一 RL 实验平台：迁移记录

日期：2026-09-27。当前批次：**入口初始化**，尚未开始业务迁移。

## 已有位置与成果

以下是已读项目记录与前序目录核验的入口清单，不表示本轮重新完整验证每项资产。执行某批迁移前须核对实际目录、符号链接、Git 状态和使用者。

| 机器 | 旧位置／已有依赖 | 保留事项 |
|---|---|---|
| Cobot | `/media/agilex/Getea1/jiaan/projects/rlt` | RLT 现有部署、模型、环境与运行记录，顶层与嵌套 Git 分别核验。 |
| A6000 | `/data/LFT-W02_data/jiaan/projects/proj-20260904-cobot-realworld-rl` | RLT 适配、审计、数据契约及实验配置；保留未提交修改。 |
| A6000 | `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/runs/plug_v3_yyshadow/warmup_20260925_trials` | 有价值的 warmup 对比产物，不能按临时文件直接清理。 |
| A6000 | `/data/LFT-W02_data/jiaan/jiaan/projects/expo-ft` | 已存在的独立仓库，作为固定版本依赖接入；本次不移动或提交其工作区。 |
| Cobot | `/media/agilex/Getea1/jiaan/data/evaluations` | 现有模型现场评测记录，不能误作具备完整 action replay 的训练数据。 |

## 首个候选验收范围

先登记并验证当前 RLT warmup 5k 模型及现有评测的完整来源，做固定输入的离线加载对照；不启动新的在线学习。

验收要求：权重与预处理校验一致，恢复运行所需 actor／learner 版本明确，现有评测记录完整可读；忠实复现参数不因框架接入而静默改变。

## 逐批迁移约定

一次只处理一个明确范围，记录来源、目标、依赖、版本／校验值和回退入口；先复制与验证，再切换，最后清理对应旧文件。未验收不切换，仍被依赖或缺少可靠备份的原件不清理。共有目录按文件实际归属处理，保留其他对话的未提交修改及共享资产。

迁移批次记录至少包括：范围、来源与目标、验证结果、切换状态、可清理清单及实际清理结果。初始化完成仅表示入口与 Git 可接管，不代表运行环境或业务功能已验收。

本轮没有迁移／删除旧文件，没有安装项目运行环境、启动训练、加载模型或控制机器人，也没有变更当前网页服务。

## 初始化发布记录

- 2026-09-27：项目目录与维护入口已建立，基础提交已 push 并核对远端 main 一致。
- 仓库：https://github.com/ajwwja777/rl-platform（独立仓库，非 GitHub fork）。
- 首次发布提交：`3063ae62a7043184e7331d877d1e89bb067b18cc`。
- 本记录在首次发布验证后追加并单独提交；最新版本以 main 为准。
- 运行状态：源码／文档基础已发布，业务迁移、环境安装及新位置运行验收尚未开展。

## 2026-09-27 RLT 业务迁移（进行中）

本批已取得迁移授权，guide Git 由另一个 agent 管理，本会话不提交／推送 guide。

来源为 Cobot /media/agilex/Getea1/jiaan/projects/rlt 和 A6000 旧 proj-20260904-cobot-realworld-rl。两边原件、差异与复制核验置于 outputs/migrations/20260927-rlt/。生产冲突以现场版本为基础；A6000 独有工具和测试保留。当前 cohort、模型计算、奖励、数据比例和 warmup 预算不变。

新运行布局见 README。代码主工作区在 A6000，通过 Git 发布后同步 Cobot；新模型、Replay、环境和数据均使用 Cobot /home/agilex/jiaan。模型历史归 A6000，现场只留当前所需模型。旧文件通过完整验证前不得清理。

代码调整：路径与所有权拆分；推理环境隔离；输出目录可配置；加载前要求已完成 warmup 的 checkpoint；Python 3.8 网页契约导入兼容；补充 Trace 仅在终止时 fsync（HDF5 和 Replay 提交语义不变）。A6000 可选 Actor pinning / 动作平滑保留，但当前 profile 不启用，默认动作裁剪保留现场公式。

实时进度与验收证据以后续“验收／切换”记录为准，不能将准备工作视为完成。
