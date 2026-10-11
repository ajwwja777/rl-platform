# 固定 Actor 探索采集交付

状态：开发、CPU 验证、提交推送、现场源码同步完成；待用户真实采集。负责人：当前执行对话。

用户授权：完成探索采集入口，之后交给用户加载与采集；不替用户运动机器人。

范围：固定 HIL8 7281 / Actor 3640，50Hz RTC EMA，右臂六关节相关有界探索，夹爪固定；Learner 禁用，独立 Replay/录制/trace；旧入口与默认、上游、训练资产不改。另一个 online-review worktree 保留。

基线：c731184；分支 feature/exploration-collection-20261011；工作区 /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/exploration-collection-20261011/rl-platform。

验收：冻结与评估隔离；实际执行动作与探索来源可追溯（含 RTC 保留前缀/HIL）；私有目录；禁用 Learner；原路径回归；提交推送后仅在现场空闲且持锁时同步与 SHA 校验。所有测试为 CPU 假 I/O，不加载模型、不占 GPU、不运动。

停止条件：现场忙、源文件非预期、权重身份不符、动作或 trace 合约失败；保留旧入口回退。

唯一下一步：用户在数据采集页加载 plug-v3-hil8-explore，先完成3–5个完整Episode；执行方核对 trace/Replay 身份、实际偏移、HIL与终局后继续累计20–30集。Learner关闭。

结果与使用/回退：[RESULT.md](RESULT.md)。A6000 101项、现场52项CPU检查通过，代码121f0f8已推送并同步；服务、权重和旧Replay未变。实际推理/运动由用户操作，尚无本入口自主收益结论。

## 保存目录纠错

用户已采3个完整Episode，但探索固定目录覆盖了网页选择，已定位并完成A6000修复/15项CPU验证；活动模型未重启，现场补丁待空闲同步。当前下一步：用户可选择真实探索目录查看3条已有记录；结束Session/释放后再同步路径修复。详细事实及UUID见[STORAGE.md](STORAGE.md)。
