# 中转机复现记录

**状态：** upstream + real-data + π0.5 full-model RLT-only save/resume verified  
**日期：** 2026-09-04

## 运行边界

- 上游 clone 保持 clean，不在其中开发 Cobot adapter。
- 环境、cache、fake 数据、checkpoint 和日志均位于 `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/`。
- 本阶段已只读核验 HPC canonical 数据，并在中转机使用其 checksum 等价副本；trainer π0.5 base 已只读复制到 scratch 并完成 checksum。
- 未申请 HPC GPU、未连接 Cobot，不运行 ROS 或机器人程序。

## Online runtime

- 环境：`scratch/cobot-realworld-rl/envs/rlt-online-py310`，Python 3.10.18；实际占用约 5.2 GiB。
- 安装：editable `packages/openpi-client` + `rlt_online_rl[dev]`，JAX CUDA12 0.5.3。
- 测试：`python -m pytest -q rlt_online_rl/tests` → `43 passed, 1 warning in 25.89s`。warning 是根 pytest 配置中的未知 `exclude-dependencies`，未影响测试。
- fake Machine A：上游默认 8000 被另一用户服务占用，未触碰该服务；运行时把模块常量覆盖为 18081，在不修改源码的情况下完成 WebSocket round trip。
- payload：metadata 为 `fake-machine-a/nudge-no-return`；`z_rl=(2048,)`、`proprio=(7,)`、`ref_chunk=(10,7)`，首末 chunk 不同。
- 上游缺口：正式 `MachineAFeatureClient` 先访问同端口 HTTP `/healthz`，但 `launch/fake_machine_a.py` 只提供 WebSocket。raw WebSocket 可用，fake server 不能原样覆盖正式 health handshake。

## Stage 1

- 环境：`scratch/cobot-realworld-rl/envs/rlt-stage1-py311`，Python 3.11.12；按 `uv.lock` 安装，约 7.5 GiB。
- dataloader：两种 debug 配置均为三相机；模型 state `(B,32)`、action `(B,50,32)`。从 stdin 调用默认 multiprocessing worker 会因 `<stdin>` 不可重导入而失败；`num_workers=0` 的交互 smoke 通过，正式文件入口仍需单独做 worker 吞吐测试。
- `debug_rlt`：RTX A6000 单卡、batch2、2 steps；`loss/rlt_loss` 为 `15.0628`、`19.1557`，finite；峰值观察显存约 36.7 GiB。checkpoint step 1 原子完成。
- resume：step 1 在约 2.82 秒完成恢复，读速约 557–620 MiB/s，继续得到 step 2 loss `15.0651` 并保存成功；Orbax 报告 metrics item 缺失，但 train state/params 恢复成功。
- `debug_rlt_joint`：RTX A6000 单卡、batch1、1 step；`rlt_loss=12.2203`、`vla_loss=2.2084`、`total_loss=14.4287`、`grad_norm=85.4656`，finite。checkpoint 原子完成。
- joint checkpoint 第一次写出约 4.8 GiB 逻辑数据、耗时约 64 秒；实际目录约 3.0 GiB。正式训练不应使用过密保存周期。
- 运行结束两卡均回到驱动占用、0% utilization；没有留下训练进程或临时 checkpoint 目录。

## `legacy40-v2.1` 真实数据

- HPC canonical：164 files、421,394,882 B；40 episodes、29,383 frames、30 FPS、120 路 AV1 episode videos。中转机副本完成后 `rsync --checksum --dry-run` 无差异。
- canonical 源与中转副本的逐文件内容树 SHA-256 均为 `4bd2987f61900da803c030615b8c612b6547dab8c8219c116a475250aae835a0`。
- 官方 LeRobot loader + PyAV 可直接读取本副本的 AV1；这不改变旧 Cobot deployment 中 `decord==0.6.0` 不能读取 canonical AV1 的既有结论。
- raw sample：state `(14,)`、action chunk `(50,14)`、三相机均为 `(3,480,640)`；正式 transforms 后 batch 为三路 `(2,3,224,224)`、state `(2,32)`、action `(2,50,32)`。14:32 padding 精确为零，所有值 finite。
- episode 0 共 782 frames；action 与同帧 state 的关节维平均绝对差约 `0.01215 rad`，与 next-state 约 `0.00973 rad`，支持“原始 action 是近似 absolute joint target、训练时仅关节转换为 delta”的解释，但这不是全数据动力学证明。
- 以 512 frames 计算的临时 stats 已用于 canary，仅用于健康检查。
- 完整 29,383-frame stats 使用 batch 32、workers 8、CPU 在约 6 分 17 秒内生成：`runs/stage1-production-preflight/assets/openpi-rlt/cobot_rlt_pi05_joint/legacy40-v2.1/norm_stats.json`，3,462 B，SHA-256 `0e86c542fe27de2018a031c72d493494821697ed9f3a9c976511ee9ddfb1a2e8`；state/action 的 mean/std/q01/q99 均为 finite 14D。正式 run 只使用该完整版本或其 checksum 等价副本。

## 真实数据 joint canary

- 数据：`legacy40-v2.1`；模型：dummy π0.5 + dummy RLT；单卡 RTX A6000 GPU1；batch 1、workers 2。
- `step_0`：`loss=21.2537`、`rlt_loss=19.2825`、`vla_loss=1.9712`、`grad_norm=129.3100`，finite；checkpoint 原子完成。
- 上游 `initialize_checkpoint_dir` 明确把 steps `()` 和 `(0,)` 都判为“尚无可恢复 checkpoint”，因此只含 step 0 时 `--resume` 会重新开始。这是官方行为，不能把该次运行作为恢复证据。
- 生成 step 1 后再次 `--resume`，日志明确 restore step 1（约 5.63 秒），仅运行 step 2：`loss=16.5626`、`rlt_loss=14.6270`、`vla_loss=1.9356`，并原子保存 step 2。save/resume 闭环通过。
- 项目 overlay tests：加入 PyAV backend 回归后为 `45 passed in 47.73s`，ruff 通过。

## 上游 loss 语义审计

- norm stats 在 14D 上计算，随后 `PadStatesAndActions` 补零到 π0.5 内部 32D。
- `compute_loss_with_prefix` 对 32 个动作维直接求平均，padding 14:32 虽目标为零，但没有 action-dimension mask，仍进入监督。
- LeRobot raw sample 存在 horizon `action_is_pad`，但上游 repack 后没有传入 flow loss。首轮复现保持固定上游行为；若修改，必须作为单独方法变体和消融，不得静默称为忠实复现。

## 真实 π0.5 base RLT-only canary

- base：`scratch/cobot-realworld-rl/assets/pi05_base`，29 files、12,441,749,581 B；10 个登记大文件 SHA-256 全部匹配 trainer manifest，`rsync --checksum --dry-run` 无差异。
- 配置：`cobot_rlt_pi05_only`，真实 `legacy40-v2.1`，batch 1、workers 2、`alpha=0`、RLT hidden size 2048；RTX A6000 GPU1。GPU0 上存在其他用户进程，未触碰。
- step 0/1 的 RLT loss 分别为 `18.9727`、`19.4356`；grad norm 分别约 `130.8905`、`137.1891`，均 finite。base restore 约 7.82 秒。
- 从 step 1 明确恢复耗时约 13.75 秒，继续得到 step 2 loss `18.8834`、grad norm `130.8226`，并成功保存 step 2。
- 首次 checkpoint 保存约 318 秒阻塞、350 秒总耗时；第二次约 61.52 秒阻塞、93.09 秒总耗时。最终 step 2 目录约 21 GiB，逻辑 params 约 9.3 GiB、总 train state 约 15.3 GiB，无 tmp 残留。
- JAX 观察占用约 46.48 GiB，接近单张 A6000 上限；该结果仅证明真实 full-model 的 RLT-only 训练与恢复，不支持在本机尝试 joint full FT，也不代表任务效果。

## 结论

官方源码在中转机上已证明 online 组件测试、fake WebSocket、双臂 14D 真实数据加载、dummy joint VLA+RLT，以及真实 π0.5 base RLT-only 的 forward/backward/checkpoint save/resume 链路可用。该结论仍不包含真实 base 的 joint full FT、HPC 多卡、online Cobot adapter 或真机效果。
