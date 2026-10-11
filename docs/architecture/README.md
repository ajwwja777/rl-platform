# 架构、接口、环境与迁移

[全项目进度与经验](../README.md) · 更新：2026-10-11（文档整理，无新增模型能力验证）。

明确项目职责、实现入口、机器/环境和资产归属；记录迁移与文档协作，不承担算法能力验收。

## 现状与边界

- 已验证（本次文件核对）：主仓库、固定上游入口与独立 worktree 均在 A6000；本次只改 MD。框架实施见 [本次任务](tasks/2026-10-11-documentation-framework/TASK.md)。
- 历史迁移按 [MIGRATION](MIGRATION.md) 的日期与回执理解；旧路径/PID/“尚未完成”不得直接沿用。
- EXPO-FT 保留独立方法和环境，接入状态由实际配置/对应项目证据判断。

## 经验

先复制、按收据核验，再清理；旧归档不能覆盖新文件的验证。独立 worktree 不是独立 GPU/环境/服务。guide 只存简短交接，详细证据归本项目。来源：[迁移记录](MIGRATION.md)。

## 专题入口

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [MIGRATION.md](MIGRATION.md)
- [MIGRATION_CONTEXT.md](MIGRATION_CONTEXT.md)
- [COBOT_GAP_ANALYSIS.md](rlt/COBOT_GAP_ANALYSIS.md)
- [HPC_MIGRATION_MANIFEST.md](rlt/HPC_MIGRATION_MANIFEST.md)
- [LOCAL_REPRODUCTION.md](rlt/LOCAL_REPRODUCTION.md)
- [PLUG_V2.md](rlt/PLUG_V2.md)
- [UPSTREAM_MANIFEST.md](rlt/UPSTREAM_MANIFEST.md)
- [WORKFLOW.md](rlt/WORKFLOW.md)

## 任务

日期任务保留当时证据；归档不是重新验收。新任务按问题建立，跨模块只存一份正文。

- [2026-10-11-documentation-framework](tasks/2026-10-11-documentation-framework/TASK.md)
- [2026-09-21-online-learning-diagnosis-and-framework-review](tasks/2026-09-21-online-learning-diagnosis-and-framework-review/TASK.md)
- [2026-09-19-upstream-reproduction-reset](tasks/2026-09-19-upstream-reproduction-reset/TASK.md)
- [2026-09-18-plug-v2-hpc-training-preparation](tasks/2026-09-18-plug-v2-hpc-training-preparation/TASK.md)
- [2026-09-17-tracking-conflict-recovery](tasks/2026-09-17-tracking-conflict-recovery/TASK.md)
- [2026-09-17-18-daily-summary](tasks/2026-09-17-18-daily-summary/TASK.md)
- [2026-09-16-daily-summary](tasks/2026-09-16-daily-summary/TASK.md)
- [2026-09-14-daily-summary](tasks/2026-09-14-daily-summary/TASK.md)

## 既有产物入口

以下仅核对路径并建立索引，正文/图/manifest 保留原位置；目录存在不代表本次复核了结论。跨模块产物以主要问题归属，按需互相引用。

- [deployments](../../outputs/deployments)（目录）
- [environments](../../outputs/environments)（目录）
- [migrations](../../outputs/migrations)（目录）
- [retired-cobot-rlt-20261009](../../outputs/retired-cobot-rlt-20261009)（目录）
- [rlt](../../outputs/rlt)（目录）
- [storage](../../outputs/storage)（目录）
- [verification](../../outputs/verification)（目录）
