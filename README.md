# RL 平台

本项目维护 RLT 的配置、Replay、采样、学习、模型发布与评测边界。A6000 是代码和 Git 主工作区；Cobot 运行现场推理、采样和在线学习。

- A6000：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`
- Cobot：`/home/agilex/jiaan/project/rl-platform`
- 数据：`/home/agilex/jiaan/data/rlt/plug_v3_yyshadow/`
- 操作手册：[RUNBOOK](docs/RUNBOOK.md)
- 本轮验收与清理：[MIGRATION](docs/MIGRATION.md)
- 2026-09-28：正式网页已使用新路径；现场加载、独立在线恢复和数据读取通过。历史归档／旧目录清理仍以迁移记录为准。
- 仓库：<https://github.com/ajwwja777/rl-platform>

## 目录与职责

| 目录 | 用途 |
|---|---|
| `configs/rlt/plug_v3_yyshadow/` | 当前在线／冻结配置和 Stage 1 发布清单 |
| `methods/openpi_rlt/` | Cobot 适配、Session/HIL/录制、训练工具和历史方法 |
| `third_party/openpi-rlt/` | 固定版本的上游子模块，自有独立仓库，保留作者历史 |
| `scripts/` | 当前启动／状态／释放入口和迁移核验 |
| `models/rlt/plug_v3_yyshadow/` | Stage 1、固定 warmup 5k、可更新在线权重 |
| `outputs/rlt/plug_v3_yyshadow/` | Replay、学习指标、日志、推理服务登记和诊断 |
| `outputs/migrations/` | 原件、差异、校验和切换证据 |
| `configs/environments/` | 现场包版本清单；不等同于跨机器环境重建保证 |
| `outputs/environments/` | 现场冻结环境备份及哈希 |
| `envs/online` / `envs/stage1` | Cobot 的 Python 3.10 学习／3.11 推理环境 |
| `.venv` / `uv.lock` | A6000 CPU 开发测试环境，不替换现场冻结环境 |

模型、大数据和运行输出不进入 Git。模型格式、环境版本、来源和校验结果进入文档／发布清单。

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
