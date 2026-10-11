# RL 平台：当前进度与关键经验

最新实际核对（2026-10-11 夜）：[第三批探索数据](online/tasks/2026-10-11-exploration-collection/BATCH3.md)累计29完整Episode（11自主成功/2辅助成功/16失败），1068条Replay与strict回执逐数组一致；固定Actor3640、训练batch0。用户中途移动插孔，前后结果不可直接归因。含HIL窗口仅19条/1.78%，下一步为匹配离线纠正BC与加Q诊断；尚未执行新训练。缓存新运行56hits/62requests，构建9.017秒。

2026-10-11 晚：[探索采集收尾/目录/末端往返](online/tasks/2026-10-11-exploration-collection/FINISH_AND_MOTION.md)：已定位test历史竞态与269/564重复特征引用，修复CPU验证通过；新计划Reference在末端每.25s交替跳变，纯Stage1/RTC/探索归因仍需单因素证据。

更新：2026-10-11。本次为文档框架整理，技术状态引用已有证据，不代表重新查询现场、复跑实验或得到新的成功率。

## 当前重点

Online 新经验能否改善自主能力仍是主要问题。最近正式交付为 HIL8 可选分批 Online 候选；已有离线软件一致性和局部模仿改善证据，失败终点 Q 残差、位置 OOD、独立 TEST 与自主持续收益仍不足。下一步见 [Online 当前问题](online/tasks/2026-10-11-online-learning-evidence/TASK.md)，不因文档整理自动训练或部署。

| 模块 | 当前进度与未完成项 | 入口 |
|---|---|---|
| 架构/环境/迁移 | MD 模块/任务整理完成；代码与资产原位置保留 | [architecture](architecture/README.md) |
| Stage1 | 数据/输入/原生数值链已有历史审计；独立泛化与 RL 表征能力仍不足 | [stage1](stage1/README.md) |
| Warmup | 5k/20k 选择与后续候选有记录；重复 DEV 不是独立 TEST | [warmup](warmup/README.md) |
| Online | HIL8 3seed 局部拟合改善；接管截断/末端配额未通过保持；自主持续收益未证实 | [online](online/README.md) |
| 执行 | 发布准备/trace/反馈修复有交付证据；当前进程与闭环表现须另查 | [execution](execution/README.md) |

文档整理本身的进度/验收：[任务](architecture/tasks/2026-10-11-documentation-framework/TASK.md)。另一对话研究草稿保持未合并，入口在 Online 模块；不将草稿主张升级为事实。

当前执行交付：[固定 HIL8 探索采集](online/tasks/2026-10-11-exploration-collection/RESULT.md)。A6000 101项/现场52项 CPU 合约与回归通过，新入口已同步且网页可见；固定权重、Learner关闭、独立数据目录。历史第二批：用户已采11个完整Episode（6失败5自主成功、无HIL；新增4轮同位置均成功），探索/Actor身份有实际trace；模型目录自动匹配已同步现场，77项前端回归通过，实际history可读11条。通用HDF5 action存在双臂联合有效性限制，右臂coordinator字段与独立Replay须分开核对。[保存位置纠错](online/tasks/2026-10-11-exploration-collection/STORAGE.md)。Online收益仍未证实。

## 关键经验

- 先闭合数据身份、真实训练输入、更新和 served 权重，再讨论能力；配置值、文件存在、测试通过不是当前运行证据。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- TRAIN、重复 DEV、独立 TEST 分开；自主/辅助成功分开；完整 Episode 是不确定性单位，重叠窗口不是独立试验。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- 同时看损失/排序、同状态动作偏好、物理动作幅度与闭环结果。低 TD loss、Actor Q 上升、末段 AUC 或模仿改善，均不能单独证明 Online 越来越好。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- 历史失败、反例、错误实验标签和回退也保留；资源/生产边界与算法假设分开。恢复要验证完整状态与采样轨迹，不能只看步数。[来源](online/tasks/2026-10-10-online-credit-repair/TASK.md)
- 文档按模块归属，每个问题一个主任务；详细证据归任务和原产物，guide 只保留摘要。多对话接管先查进度、Git 和 coordination。[约定](../AGENTS.md)

## 如何继续工作

从模块 README 找任务，再读 TASK.md 的范围、验收、进度和下一步；复杂任务再读 ANALYSIS.md/RESULT.md。已验证/失败/证据不足必须有证据入口。具体任务进度只在 TASK 维护，模块记录稳定认识，这里只汇总关键变化。

代码导航见 [项目 README](../README.md)，现场操作见 [执行手册](execution/RUNBOOK.md)，历史批次见 [迁移记录](architecture/MIGRATION.md)。旧路径页仅供兼容；新增正文写对应模块。outputs 中原始报告、图、日志和 manifest 不移动，各模块列出所属入口。
