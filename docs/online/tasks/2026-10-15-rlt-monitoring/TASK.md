# RLT Online 四图监控（2026-10-15）

用户授权为数据采集页增加 RLT Online 四图，并保持指标读取独立于模型加载。代码位于方法适配层与 Web，固定上游和训练目标未修改。

## 实现口径

- 训练趋势直接读取 `online/metrics/learner_metrics.jsonl` 的 `global_step/q1_mean/target_q_mean`；Critic loss 保留原生双 Q MSE 之和。只显示超过 `warmup_required_updates` 的 Online 更新，恢复后的未来日志不显示；不使用 Actor 更新次数作横轴。
- 支持普通 Online、独立分支、runtime_profile 及 supported_runtime 的登记路径隔离。MC/混合目标模式标为实际训练 target，不能误称纯 TD target。
- 自主成功率使用真实 collection/online 终局和 HUMAN/MIXED 步数，20 轮窗口；接管成功只进分母。Warmup/evaluation 分离；放弃、缺标签、空轮次或缺 HIL 信息不进分母。历史缺口不推断。
- `q_monitor.py` 在 EnvDriver 创建独立 spawn CPU 子进程（最多两 CPU affinity，32 条有界队列）。控制线程仅复制小型已有 numpy 特征/动作并非阻塞入队；异常、队列满不改变控制状态。评分读取当前配置指定的本地已保存 Critic checkpoint，每轮固定；Actor 与 Critic 版本分别记录。
- 动作 Q1 在 `fresh_plan` 成功或 RTC `complete_inference` 之后采样，RTC 已提交前缀按实际采用值评分，丢弃/过期结果不采样。使用原生动作归一化与 TwinCritic；supported held-gripper 模式保留同一 Critic 投影。语义为执行插值/滤波/clamp 前的已采用动作块，不声称完整动作块已被执行。
- 回合 reset 返回后记录真实 UUID/时间，仅 latched collection/online 开启采样；暂停/HIL/队列缺样处断线。读图不启动模型、Learner、ROS 或 GPU。

## 验证与边界

CPU 验证覆盖原生数值/归一化、固定 Critic、真实 spawn 子进程、接管统计、队列溢出、异常隔离、过期 plan 排除及 RTC 前缀；Web 验证覆盖四卡、读取路径隔离、离线历史、切模型响应隔离、EXPO 回归。详细数量与现场同步在交付记录补充。

当前另一任务已按用户要求暂不登记 `realtime-expo-rlt-online`，只开放 Reference 与专家 7k Frozen。本任务不恢复其训练入口、不启动训练或机器人；现有其他 Online 入口可使用监控。新运行采样代码在下次正常加载 Online 后生效；旧日志无指标不补值。真机回合与性能验收仍由现场执行。
