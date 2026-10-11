# 固定 Actor 探索采集交付

状态：101 项 CPU 验证通过；准备提交推送和现场限范围同步。负责人：当前执行对话。

用户授权：完成探索采集入口，之后交给用户加载与采集；不替用户运动机器人。

范围：固定 HIL8 7281 / Actor 3640，50Hz RTC EMA，右臂六关节相关有界探索，夹爪固定；Learner 禁用，独立 Replay/录制/trace；旧入口与默认、上游、训练资产不改。另一个 online-review worktree 保留。

基线：c731184；分支 feature/exploration-collection-20261011；工作区 /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/exploration-collection-20261011/rl-platform。

验收：冻结与评估隔离；实际执行动作与探索来源可追溯（含 RTC 保留前缀/HIL）；私有目录；禁用 Learner；原路径回归；提交推送后仅在现场空闲且持锁时同步与 SHA 校验。所有测试为 CPU 假 I/O，不加载模型、不占 GPU、不运动。

停止条件：现场忙、源文件非预期、权重身份不符、动作或 trace 合约失败；保留旧入口回退。

下一步：CPU 合约/回归验证 → 提交推送 → 空闲现场限范围同步 → 交付网页操作与采集边界。
