# 2026-09-20 plug_v2 真机在线 RL、r13 批次修复与 actor8445

## 结论

plug_v2 因果 RTC 候选完成了当日真机部署、固定策略观察、HIL/终局录制和多轮在线更新。用户现场观察到动作已能较稳定、较平滑地接近目标孔，并至少出现过一次无 HIL 自主插入成功；样本仍不足，不能据此宣称稳定成功率或单调在线提升。

当日生产链从 `rtc-causal-passive-v1-step5000` 经门控发布依次到 5420、5775、6155、6715、7070、7435、7765、8185，最终到 `8445`。最新发布为 `rtc-online-20260920T223823-666775`，最后一批使用5个新 UUID、52条新 transition、260次更新，离线发布门全部通过。固定门会同时检查部署匹配动作误差、命令加速度、early-policy/terminal Q 的 AUC 与 separation，以及 actor/critic 有限值；被拒候选不会替换当前真机版本。

## 实时控制与现场故障处理

- Runtime r12 保持30 Hz异步 RTC、delay6、0.10 rad/s速度、0.9 rad/s²加速度、0.04 rad tracking和Piper关节限制。
- `rtc_deadline_missed` 只允许保持最后安全指令、丢弃迟到 future 后重新 d0 锚定，单 episode 最多3次；tracking、workspace、硬件和 recorder 故障继续 fail-closed。
- 被动左臂与双夹爪继续锁存 episode 起点的 Stage1 d0 action-space 命令；deadline重锚不能重写。
- Task5 start HTTP 422 已增加服务端异常类型/堆栈和客户端有限 detail。一次旧前台终端显示 `cycle_exited_1` 是运行代码指纹更新时旧 learner 的 fail-closed 退出；新 supervisor 接管后网页继续工作，不是同一代进程矛盾。

## r13 在线批次调度修复

定位到 `rtc_online_cycle.pending()` 的集合错误：它以“至少5条尚未尝试”为触发条件，却把所有未接受 UUID 返回给训练，导致拒绝批次随新数据重复形成5、10、15条候选。修复后：

- 每次只由恰好5条从未尝试的新 UUID 触发；超过5条也只取最早5条。
- 已尝试批次不会再次作为触发批次或扩大更新步数；原始 episode、标签和审计不删除。
- 历史 episode 仍属于累计 off-policy replay；只有独立数据质量审计判定损坏、标签或时序非法时才排除。发布门拒绝的是候选权重，不等同于判定源数据无效。
- 定向测试先复现旧实现失败，修复后4项通过；新不可变 runtime descriptor 为 `rtc-online-20260920T212442-557885-runtime-r13-20260920T214732`，模型权重当时保持7070不变。

7070 后首个5条候选的 early-policy Q AUC 从0.9545降到0.8864，超过允许的0.05降幅，因此拒绝；后续重复批次的 separation 又从0.10282降到约0.0812–0.0819，低于80%保持门。r13消除了重复触发后，后续独立五条批次仍可被正常训练、拒绝或发布，最终已推进到8445。

## 当日收尾快照

2026-09-20 22:40 CST 只读核验：

- Cobot 根：`/media/agilex/Getea1/jiaan/projects/rlt`。
- backend：`ready_disarmed`，supervisor PID 663351；Stage1、actor、Session和cycle子进程仍登记运行，PID属于动态快照。
- Session：`stopped`、`policy_paused=true`、终局 aborted、无 fault；当前 actor/learner 8445。
- unified UI：8015；数据根 `/media/agilex/Getea1/jiaan/data/rlt/plug_v2/online`。
- online ledger：最后一次更新 accepted；另有1条从未尝试的新 episode，尚不足下一批5条。
- 模型仍预加载。下次开始前重新核验进程、端口、current release和机械臂现场状态；结束全部 RLT 工作时才使用 `rlt_down.sh` 释放模型。

## 下一步

1. 固定相机、插头夹持、排插位置和复位协议，先用8445做小样本纯自主评测，单独统计成功、失败、HIL和基础设施中断。
2. 每5条有效 success/failure episode 检查一次 operation、release audit和 actor version；拒绝时记录具体门，不把版本未变化误认为未训练。
3. 将“触发批次”和“累计 replay”在网页上分开展示，避免 attempted/quarantined 字段被误解为删除训练数据。
4. 在线改善趋势必须使用固定场景、同一版本分组和足够样本确认；当天一次自主成功和用户主观平滑反馈仅作现场证据，不作最终成功率结论。
