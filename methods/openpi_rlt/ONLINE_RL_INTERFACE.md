# openpi-RLT × Cobot 在线闭环接口

## 推荐边界

首轮保持 upstream 的算法和服务分工，不把 Task5 改造成 RL learner，也不让上游
ROS 2 单臂脚本直接控制 Cobot：

- **Machine A**：加载冻结的 R1 Stage 1 checkpoint；每个 observation 返回随图像/状态变化的
  `z_rl` 和冻结 VLA 的 `ref_chunk`。
- **Machine B**：沿用 upstream actor、critic、learner、replay manager 和原子 actor snapshot。
- **Cobot bridge**：项目私有 ROS 1/Task2 adapter，负责三相机、14D state、策略输出、逐臂
  takeover 和 raw trace；策略只能发布 `/task2/policy/joint_left/right`，Task2 coordinator
  继续独占最终前臂 command。
- **Task5**：可选的数据/界面旁路，负责 recording、sidecar 和人工 terminal outcome；网页未开
  或未 recording 不阻止 policy rollout。

这个边界最大程度复用公开实现，同时保留已经现场验证过的 Task2 安全与接管语义。任何
Machine A/B、Task5 或网络异常都不得绕过 coordinator 直接发布动作。

## 单个 episode 的数据流

1. bridge 原子采样三相机与 14D 当前状态，发送给 Machine A。
2. Machine A 返回 `z_rl (2048,)` 与绝对 14D `ref_chunk (10,14)`。
3. warmup 或 base 模式执行 VLA reference；online 模式由 actor 根据
   `z_rl + proprio + ref_chunk` 产生 refined chunk。
4. 任一后臂示教按钮触发 Task2 接管：策略立即暂停、旧 chunk/request/version 失效；bridge
   继续按 control tick 记录实际执行动作和左右臂接管 mask。
5. 最后一个接管释放后，从最新 observation fresh replan；不得续播接管前 chunk。
6. 每个 tick 先进入项目 raw trace；success/failure/done 尚未提交时 episode 为 `pending`，
   不进入正式 replay。
7. 操作员在 Task5 UI 或等价的独立人工信号中提交 terminal outcome。success 仅在最后一个
   transition 给 reward 1；failure/done 为 0，不能把 episode 标签广播到所有帧。
8. episode 完成后再生成 upstream replay windows，并发送 replay manager。learner 异步更新；
   新 actor snapshot 只在 episode 边界、hash/schema/version 验证通过后切换。

## HIL 字段映射

| Cobot 原始事实 | upstream replay | 规则 |
|---|---|---|
| VLA 当前参考 | `ref_chunk` | 始终保留，不被人工动作替换 |
| 实际前臂 command/执行目标 | `action_chunk` | 策略或人工实际执行值 |
| 左右臂 takeover `(T,2)` | `source_chunk (T,)` | 无=`BASE/RL`；单臂=`MIXED`；双臂=`HUMAN` |
| 左右臂 takeover `(T,2)` | raw sidecar `expert_mask` | 原样保留，供审计和未来逐臂 loss 使用 |
| episode success | `rewards[-1]=1`, `success=1`, `done=true` | 只在明确提交后写入 |
| failure / unlabeled done | 全零 rewards, `done=true` | failure 与 done 分别保留在 raw metadata |
| 未结束或未标注 | 不写正式 replay | 留在 recovery/pending 区 |

upstream learner 把 `HUMAN/MIXED` step 的 BC target 设为 `action_chunk`，把 `BASE/RL`
step 的 target 设为 `ref_chunk`。因此单臂接管时把该 tick 标成 `MIXED` 是兼容映射：未接管臂
的 executed action 本来就应等于当时策略实际执行值；原始逐臂 mask仍保留，以免丢失信息。

## 14D online action contract

- 顺序固定为 `[left joint 1..6, left gripper, right joint 1..6, right gripper]`。
- online actor/action stats 使用 14D 容器；左右 6 个关节均为相对当前 state 的 delta，两个
  gripper 均为 absolute。
- upstream 公开实现只对 `:6` 做 delta，不能直接用于双臂。项目私有
  `cobot_adapter/online_runtime.py` 在进程启动时只替换 action representation 与 learner 的
  JAX inverse；fixed upstream 文件保持 clean。
- Stage 1 的 14D norm stats 可作为首轮 online action stats 候选，因为它已按相同
  joint-delta/absolute-gripper 变换生成；部署包仍需用最终 checkpoint assets 和 manifest hash
  绑定，不能依赖中转机绝对路径。

## 验证阶梯

1. 纯函数：三相机/14D、双臂 delta round-trip、per-arm mask → source、terminal reward。
2. upstream fake Machine A + project Machine B wrapper：一条 episode 写 raw trace、finalize replay、
   learner update、actor snapshot save/reload。
3. `legacy40-v2.1` 离线 observation/action replay：不发布 ROS，不连接机器人，验证实际数据、
   feature payload、14D stats 和 replay shape。
4. Cobot 真实 observation 的 zero-publisher shadow：读取相机/关节、调用 Machine A/B、执行所有
   schema/latency/version/safety gate，但禁止创建 command publisher。
5. Task2 fake ROS：覆盖 Enter 前示教、rollout 中逐臂 takeover、最后接管释放后的 fresh replan、
   迟到 response 丢弃、terminal pending/commit/recovery。
6. 现场重新核验 CAN、急停、相机、夹爪、工作区并取得明确动作授权后，才进入 warmup rollout。

## 当前未决项

- Task2 topic/service、三相机/14D schema 和按臂 mode 语义已由 snapshot 与自动测试固定；仍需在
  现场 `--shadow` 下验证真实 payload、相机同步和连续 20 Hz latency。
- Task5 facts 已证明包含 `control_source_left/right`、intervention id、valid mask 和 terminal labels，
  但首轮部署不依赖 Task5；人工 terminal ROS service 与网页如何统一显示仍待现场验证。
- 首轮使用公开 Agilex 20 Hz、RL chunk 10；整个 `in_the_pot` 作为 actor-refined critical phase。
  这与论文 50 Hz 不同，必须作为 adaptation 报告。
- warmup 固定为 replay 600 条，再做 20,000 updates；历史 HIL 是否预载 replay 尚未决定，不能
  未经 manifest 混入。
- Machine A/B 已在 Cobot 4090 上以约 18.0 GiB 完成共存 smoke，首轮采用本地拓扑。自动视觉
  terminal、自动安全 reset 和 warmup→online 的真机边界切换仍待验证。
