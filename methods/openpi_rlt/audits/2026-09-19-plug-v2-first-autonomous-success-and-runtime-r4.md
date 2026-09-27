# plug_v2 首次自主成功与 Runtime r4 韧性修复

日期：2026-09-19

## 现场证据

固定候选 `rtc-corrective-r1@2000` 在 Runtime r3（Piper 关节限位与抗 windup）后取得首条无 HIL 自主成功：episode 28，UUID `d1877ed8-d74d-44bf-9af7-9d0cc7855b61`，320 policy frames、33 inference chunks、最大跟踪误差约 0.007826 rad。随后三个有效终局为失败；当前同一固定策略的有效自主结果为 1 success / 3 failure，即观察成功率 25%。样本量不足，不能宣称成功率稳定提升。

另有多轮因运行时故障放弃，不计入任务成功率：三次 `rtc_deadline_missed`，随后出现模型 WebSocket 关闭后被永久复用的 `sent 1000 (OK); then received 1000 (OK)`。r3 后未再观察到原 J5 `tracking_bound_exceeded`。

## Runtime r4

Cobot 源码：`/media/agilex/Getea1/jiaan/projects/rlt/methods/openpi_rlt/plug_v2/`。

- 6-step RTC deadline 偶发错过时，迟到计划因 generation 失效而丢弃；控制保持最后安全 setpoint，完成后从当前观测以 `prefix_length=0` 重新规划。迟到计划不会补发。
- Task5 recorder 健康查询移到独立单线程 worker，避免 HTTP 状态调用阻塞 30 Hz 控制循环。
- 模型 RPC 的连接关闭可重连一次；recv 超时会丢弃旧 socket，下一次 fresh request 建立新连接。
- 保留 30 Hz、RTC delay 6、0.04 rad tracking bound、Piper 物理关节限位和固定候选；没有放宽安全门限。
- 页面状态增加 `rtc_deadline_recoveries`、`model_rpc_reconnects`；前台终端显示 revision 与累计恢复次数。

证据：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/candidates/rtc-corrective-r1/runtime-validation-r4.json`。`py_compile` 通过；RTC queue 6项、RPC reconnect 2项通过；对仍驻留的 Machine A metadata 握手通过。没有启动新 Session 或机器人动作，r4 仍需现场验收。

## 下一步验收

当前网页 Session 已 stopped、policy paused，模型保留。下一次在 `cobot-platform` 运行普通 `./scripts/rlt_demo.sh` 即可替换旧 Session runtime并加载 r4，不需要 `--restart`。先保持固定排插位置、eval/no explore/no HIL完成至少5条干净自主轮次；任务成功/失败正常标注，运行时故障轮次放弃并从统计中排除。确认不再因单次 deadline/RPC 抖动自动结束后，再累计到至少10条有效自主结果并讨论位置扰动或在线RL；当前不更新actor。
