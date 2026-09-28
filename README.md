# RL 平台

本项目维护 RLT 的配置、Replay、采样、学习、模型发布与评测边界。A6000 是代码和 Git 主工作区；Cobot 运行现场推理、采样和在线学习。

- A6000：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`
- Cobot：`/home/agilex/jiaan/project/rl-platform`
- 数据：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/`
- 操作手册：[RUNBOOK](docs/RUNBOOK.md)
- 本轮验收与清理：[MIGRATION](docs/MIGRATION.md)
- 2026-09-28：正式网页已使用新路径；现场加载、独立在线恢复和数据读取通过。本批历史归档、环境备份及对应旧目录清理已完成，详见迁移记录。
- 仓库：<https://github.com/ajwwja777/rl-platform>

## 目录与职责

| 目录 | 用途 |
|---|---|
| `configs/rlt/plug_v3_yyshadow/` | 当前在线／冻结配置和 Stage 1 发布清单 |
| `methods/openpi_rlt/` | Cobot 适配、Session/HIL/录制、训练工具和历史方法 |
| `third_party/openpi-rlt/` | 固定版本的上游子模块，自有独立仓库，保留作者历史 |
| `scripts/` | 当前启动／状态／释放入口和迁移核验 |
| /media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/ | Stage 1、固定 warmup 5k、可更新在线权重 |
| `outputs/rlt/plug_v3_yyshadow/` | Replay、学习指标、日志、推理服务登记和诊断 |
| `models/history/` / `data/history/` | A6000保存历史模型与训练数据，索引见 configs/assets/legacy_rlt_models.json |
| `outputs/migrations/` | 原件、差异、校验和切换证据 |
| `configs/environments/` | 现场包版本清单；不等同于跨机器环境重建保证 |
| `outputs/environments/` | 现场冻结环境备份及哈希 |
| `envs/online` / `envs/stage1` | Cobot 的 Python 3.10 学习／3.11 推理环境 |
| `.venv` / `uv.lock` | A6000 CPU 开发测试环境，不替换现场冻结环境 |

模型、大数据和运行输出不进入 Git。模型格式、环境版本、来源和校验结果进入文档／发布清单。

## 从哪些源码开始看

当前网页入口使用plug_v3_yyshadow。根目录scripts负责已登记部署，methods保留Cobot适配和历史方法，third_party保留固定上游实现。历史plug_v2及旧脚本保留用于追溯，不与当前发布入口混用；旧Task2/Task5名称在ROS/HTTP协议和历史适配中保持兼容。

| 想审核的内容 | 源码入口 |
|---|---|
| 路径、预检与进程启动 | [scripts/rlt_up.sh](scripts/rlt_up.sh)、[scripts/preflight.py](scripts/preflight.py) |
| 当前学习参数与Replay位置 | [online_rl.yaml](configs/rlt/plug_v3_yyshadow/online_rl.yaml) |
| 冻结Stage1加载、预处理、固定输入验证 | [serve_stage1.py](methods/openpi_rlt/plug_v3_yyshadow/serve_stage1.py) |
| 上游角色启动及Cobot补丁边界 | [online_role.py](methods/openpi_rlt/scripts/online_role.py)、[online_runtime.py](methods/openpi_rlt/cobot_adapter/online_runtime.py) |
| 右臂7D动作与反馈 | [right_arm_env.py](methods/openpi_rlt/plug_v3_yyshadow/right_arm_env.py) |
| Session、暂停、终止及HIL状态 | [session.py](methods/openpi_rlt/cobot_adapter/session.py)、[task2_runtime.py](methods/openpi_rlt/cobot_adapter/task2_runtime.py) |
| 录制HTTP及收尾合同 | [task5_client.py](methods/openpi_rlt/cobot_adapter/task5_client.py) |
| 上游网络、学习更新、Replay | [rlt_online_rl](third_party/openpi-rlt/rlt_online_rl/src/rlt_online_rl) |
| 无机器人恢复学习验证 | [validate_online_resume.py](scripts/validate_online_resume.py) |

采集/评测共享进程的用途选择在同级cobot-web的app/backend/cobot_console/shared_model_env.py，真实硬件仲裁在同级cobot-control。阅读跨项目调用时沿这些边界查看，避免将网页按钮行为误认为算法实现。

## 当前方法

plug_v3_yyshadow 使用右臂 7 维动作、三相机、20 Hz 控制、10 帧动作块。当前 warmup 基线是 learner 5000 / actor 2500；迁移不重新训练、不调整奖励、BC/Q 权重、探索强度或数据比例。

Stage 1 Reference、固定 warmup 5000、最新冻结 Actor、最新在线 Actor 是不同使用方式。网页采集和部署共用模型进程；只有选择在线模型后进入采集，新增合格 Replay 才驱动在线更新。评测隔离 Replay，保存未标注不等于成功／失败，放弃不保留本轮记录。

历史 58 次评测、22 次成功（37.9%）保留为既有基线；本轮的加载／接口测试不构成新的真机成功率。

## 项目接口

`cobot-control` 管理硬件、CAN、ROS、示教控制权、归位和恢复；`cobot-web` 提供录制 HTTP 接口、共享模型管理、网页与终端兜底；`rl-platform` 负责 RLT 算法和模型状态。调用关系明确，不复制另一项目的运行实现。

当前录制库是 cobot-web 的 `capture_core` / `segmented_capture`，后续若迁到 cobot-dagger，再做单独接口验收。EXPO-FT 仍按自己的既有记录推进。

## 开发检查

```bash
uv sync --frozen
./.venv/bin/python scripts/audit_migration_config.py
```

跨项目测试需要 cobot-web 位于同级目录。现场环境不是通过这份 CPU 开发依赖重新安装；迁移保留现场现有版本与加载顺序。

## 数据和权重的单份归属（2026-09-28）

用户确认：采集原始数据、现场评测和当前部署checkpoint长期只在Cobot；训练中间checkpoint与停止部署的历史模型只在A6000。同一场景供多个模型训练，不为每个模型复制原始数据。

当前Cobot数据仍在/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/（示范52.82GiB、warmup81.49GiB、online0.31GiB）；模型在/home/agilex/jiaan/project/rl-platform/models/rlt/plug_v3_yyshadow/（约14.45GiB，共享Stage1占14.37GiB）。当前Replay、日志在同项目outputs/rlt/plug_v3_yyshadow/，不是原始数据的替代。

A6000历史模型在本项目models/history/（49.17GiB），历史转换数据在data/history/legacy-rlt/（0.57GiB），历史实验原件在outputs/migrations/20260927-rlt/legacy-rlt-source/runs/（45.69GiB）。新A6000 models/rlt/plug_v3_yyshadow/仍有14.44GiB部署迁移副本，旧scratch也有原始数据/权重/实验副本，最新单份规则尚未全部落实；盘点不等于已删除。

完整路径、大小和重复项见同级cobot-web/docs/STORAGE.md；原始盘点位于outputs/storage/20260928-inventory/。场景根目标为Cobot /home/agilex/jiaan/data/plug_insertion/，当前尚未改名，不混并旧相机/动作定义不同的批次。


## 2026-09-28 Getea1 存储切换

Cobot 数据与模型统一在 /media/agilex/Getea1/jiaan/data/ 和 /media/agilex/Getea1/jiaan/model/。数据按场景分、模型按项目/模型分；本轮不新增 A6000 权重备份。代码、安装环境、运行日志与 PID 留在 /home/agilex/jiaan/project/<项目>/。完整路径与批次状态见相邻 cobot-web/docs/STORAGE.md。

当前现场路径以以下配置为准：
- 模型：/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/{reference_4999,warmup_5000,online}。
- rollout：/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/{warmup,online}/three_camera_v3。
- 专家示范：/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/demonstrations/three_camera_v3。
- Replay：/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt/replay_clean_v1/replay_journal.pkl。

上文“当前Cobot数据”“场景根目标”等旧盘点段落是迁移前快照；本批复制/切换结果见 docs/MIGRATION.md 最新节。奖励、归一化、动作合同与 5,000 步 warmup 不变。
