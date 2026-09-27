# openpi-RLT → Cobot `in_the_pot` 差距分析

## 复现定位

- 论文：arXiv `2604.23073`，使用 π0.6，Stage 1 在单任务 demonstrations 上同时训练 RL token、可选微调 VLA；随后冻结 VLA 和 token encoder，在线训练小型 actor/critic。
- 本项目：使用官方仓库公开的 π0.5/Agilex 实现和现有 π0.5 base。它是官方代码路径上的 Cobot adaptation，不是论文 π0.6 checkpoint 的逐字复现。
- RLT token 是 observation-dependent representation；冻结的是 VLA 与 token module 参数，而不是冻结 token 数值。

## 论文/代码已对齐机制

| 项目 | 论文 | 公开代码 | Cobot 判断 |
|---|---|---|---|
| Stage 1 | `Lro + alpha * Lvla` | `rlt_alpha=0/1`，reconstruction 对 VLA embedding stop-gradient | B1/R1 必须从同一 π0.5 base、同一数据和预算出发 |
| 在线状态 | RL token + proprio | `z_dim=2048`、`proprio_dim=7` 默认 | 改为双臂 14D 前需 fixture；是否加入速度/EE pose 需明确 |
| 动作 | VLA H=50，RL C=10 | `chunk_len=10`，actor 参考 VLA chunk | Cobot 需要 14D×10=140 维 actor 输出 |
| reward | episode 结束人工 success=1，否则 0 | manual success/failure/done bridge | Task5/操作员提交 terminal outcome；pending episode 不入正式 replay |
| HIL | 人工动作覆盖 actor，并替换 replay reference | source/intervention trace 已有组件 | Task2 是底层控制权机制，仍需 RLT event/trace adapter |
| replay | stride 2，混合 VLA/RL/human；UTD 5 | 默认 stride/UTD 配置存在 | 必须保留 policy/checkpoint/source/takeover 的版本证据 |
| actor | Gaussian，fixed std；reference dropout 50%；BC constraint | online YAML 提供相应超参 | 先按官方默认 canary，再以安全范围和任务数据调参 |

## 必须实现和验证的适配

1. **单臂 7D → 双臂 14D。** 上游 `LerobotAgilexBagImageDataConfig` 默认 `output_action_dim=7`，delta mask 仅 `(6,-1)`；online YAML 的 `action_dim/proprio_dim` 也是 7。Cobot 必须固定 `[left 6 joints, left gripper, right 6 joints, right gripper]` 的 14D，并使用 joint-delta + absolute-gripper 的双臂 mask `(6,-1,6,-1)`。
2. **14→32 模型容器。** π0.5 内部 state/action 是 32D、H=50。已验证 norm 在 14D 上进行、随后 padding 槽为零；但固定上游的 flow loss 会平均全部 32 维，零 padding 仍参与监督，且 LeRobot 的 horizon `action_is_pad` 在 repack 后未进入 loss。首轮保持官方行为，后续应将 action-dim/horizon mask 作为显式方法变体评估。输出只取已验证的 14 个物理维，不能只因 shape 通过就接受。
3. **三相机物理映射。** 上游别名为 global/fisheye/depth，模型键为 base/left/right；Cobot 必须用固定视觉 fixture 证明 top/head、left wrist、right wrist 没有互换。
4. **频率协议。** 论文在线控制 50 Hz、C=10；Cobot 历史 wrapper 多为 20 Hz，legacy40 为 30 FPS。需选择训练/在线时间尺度和 action resampling 规则，不能把 10 steps 当成相同物理时长。
5. **terminal 与 HIL 事件。** Task2 pause/resume/takeover 负责安全控制权；RLT adapter 负责生成 raw trace 的 source、intervention interval、executed/reference action 和 episode terminal outcome。Task5 网页 recording 仍可选，不作为部署门禁。
6. **真实数据 loss。** `legacy40-v2.1` 可先用于公平 B1/R1；additional/rollout 若加入，必须验证 per-arm `expert_mask` 如何作用于 VLA loss 和 RLT reconstruction/online warmup，不能沿用目录名推断。
7. **checkpoint 发布。** Stage 1 checkpoint 包含 VLA/RLT；Stage 2 还需 actor/critic/target/replay/version。actor 只加载原子发布、hash/schema 一致的版本，并能回滚。
8. **ROS 与控制桥。** 上游真实机器人说明要求 ROS 2 Humble，并通过其 Agilex `pika_sync_ros.py`/manual bridge 控制单臂；当前 Cobot 是 ROS 1 Noetic + Task2 coordinator。不能原样启动上游 robot rollout，必须保留上游 Machine A/B、replay 和 learner 语义，只替换方法私有的 Cobot observation/action/HIL bridge，并先做无 publisher 测试。
9. **online action representation。** 上游 `ActionRepresentationAdapter` 与 learner 的 JAX 反归一化都把 joint group 写死为 `:6`，只能正确处理单臂 6 joints + gripper。项目私有 `online_runtime.py` 已以测试固定双臂索引 `[0:6, 7:13]` 为 delta、夹爪 `[6, 13]` 为 absolute，并保持 fixed upstream 不变；接入 Machine B 入口和完整 fake replay 仍待完成。
10. **逐臂 HIL 到上游逐步 source。** Task2 可独立接管左右臂，而上游 `source_chunk` 每个 tick 只有一个标量。当前 adapter 固定：无接管保留 `BASE/RL`，单臂接管记 `MIXED`，双臂接管记 `HUMAN`；VLA `ref_chunk` 不被人工动作覆盖，实际动作进入 `action_chunk`，原始 `(T,2)` `expert_mask` 继续留在 sidecar。

## 官方 online 默认与 Cobot 起点

- 上游 base runtime 示例是 50 Hz，但公开的 `agilex_ethernet/online_rl.yaml` 实际使用 20 Hz、chunk 10、`action_dim=7`、`proprio_dim=7`、`fixed_std=0.002`、replay warmup 600、warmup updates 20,000、每轮 5 个 gradient updates。
- 因此“论文 50 Hz”不能覆盖“公开任务配置 20 Hz”；Cobot 首轮可把 20 Hz 作为有代码证据的适配起点，但双臂 14D、Task2 接管时序和 `in_the_pot` reset/terminal 仍需专门验证。
- 上游人工接管在每个 control tick 记录最新 human action，不会把原始 teleop event stream 直接塞入 replay。这与 Task2 原始事件/按臂 mask 的边界兼容，但转换器尚未完成端到端验证。
- `legacy40-v2.1` 的 40 个 episode 长度为 654–878；在 H=50 的逐帧采样下共有 49,000 / 1,469,150（约 3.335%）horizon action slots 属于 episode-end padding，而固定上游没有使用对应 `action_is_pad`。此外 18/32 action dimensions 是零 padding。二者都应进入正式 run 风险清单，但首轮不能静默改 loss 后仍称为忠实复现。

## 当前可用数据与限制

- `legacy40-v2.1`：40 episodes / 29,383 frames / 30 FPS / 三相机 / 14D，首轮 B1/R1 canonical 候选。
- `additional-expert-v1`：29 episodes / 32,983 frames；expert-mask coverage `0.999166`。
- `rollout-1-v1`：15 episodes / 21,908 frames；overall expert-mask coverage `0.153003`，左右臂不同。
- 44 个 Task5 引用的 raw HDF5 缺失，但 converted/labels/facts 已保留。它不阻塞 legacy40 Stage 1；若作为在线 replay 血缘则必须明确限制。

## 进入 HPC 前的硬门

- 14D camera/action/state fixture、round-trip、官方真实 batch 和 dummy joint backward 已通过。
- B1/R1 冻结同一 dataset manifest、π0.5 base hash、seed、batch/update budget。
- 真实 π0.5 base 单 batch forward/backward 的显存、loss 分解和有限性通过；当前仅 dummy full path 已通过。
- 选定在线控制频率与 resampling 规则。
- 远端目标路径、磁盘、GPU/allocation 和 π0.5 base 位置核验完成。
