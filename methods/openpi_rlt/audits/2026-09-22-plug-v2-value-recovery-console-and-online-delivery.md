# 2026-09-22 plug_v2 在线 RLT 价值恢复、统一控制台与现场交付

## 结论

本轮没有执行机械臂动作、相机启动或真机 rollout。Cobot 上发布了新的离线验收候选 `rtc-recovery-mc-critic-v1-20260922`：learner step `16135`、actor version `7067`。本轮只重新校准 critic，actor 权重保持上一稳定版本不变，因此首次冻结 rollout 应复现既有平滑基线；后续在线更新才会以低学习率、小 Q 权重和固定发布门逐批修正 actor。状态仍是 `offline_validated_onsite_pending`，不能把离线指标称为真机成功率提升。

统一网页 `http://10.7.165.64:8015/` 已整合普通采集、RLT Session、三相机同步预览、Episode 回看、目录搜索、训练诊断和固定白名单设备操作。系统状态、训练进度、拒绝原因、loss、Q 曲线和 actor 修改量均通过同一页面动态刷新。

## 问题定位

此前约 300 条在线 episode 没有形成稳定改善，主要不是单一超参数造成，而是四个相互叠加的问题：

1. 原 TD critic 只在终局得到奖励。即使继续训练 20k critic-only，早期自主 chunk 的成功/失败 AUC 约 `0.398`、separation 仍为负，成功信号没有可靠传到插入前的决策。
2. 原“stratified”实际 batch 的 HIL 比例约 50%，并非预期 20%；actor/critic 容易被辅助成功样本主导。
3. resume 时 burn-in 使用累计 global step 比较请求步数，导致已经训练过的 checkpoint 直接跳过新的 critic-only 校准。
4. 发布门主要检查动作拟合和平滑，没有固定检查早期 Q、终局 Q 与 actor Q 的成败排序；逐父版本比较还能累积退化。

输入分布漂移仍是独立风险：相机位姿、插头夹持深度与排插位置变化会让同一 actor 表现明显变化。新网页会提示相机静止、过期、不同步或不可用，但无法替代现场固定复位。

## 训练与在线更新修复

- critic target 改为 episode Monte Carlo success return，用终局成功标签直接监督整条可验证轨迹，避免仅靠长距离 TD 传播。
- 重新进行 2000 次 critic-only 校准；actor 保持 version 7067，避免把尚未可靠的价值梯度直接写进策略。
- replay 每个 128 batch 固定 26 条 HIL（20.3125%）；剩余 policy 样本严格平衡 success/failure。
- actor 的 Q 梯度只作用于原始 terminal 或 HIL transition；BC 与平滑仍使用完整 batch。
- 在线配置固定为：actor lr `1e-5`、BC weight `10`、Q weight `0.01`、delta weight `100`、每 5 条全新且已验证 episode 触发一次、每条新增 transition 5 updates、单批最多 2000 updates。
- 一批被拒绝后只隔离该批 UUID，不会阻塞后续五条新 episode；历史事实仍保留审计。
- 新候选只在下一条 episode 开始时切换，单条 rollout 内 actor 固定。

选中 checkpoint 的离线价值指标：

| 留出组 | 指标 | AUC | separation |
|---|---|---:|---:|
| autonomous early | Q(data) | 0.5648 | +0.0274 |
| policy-terminal early | Q(data) | 0.5980 | +0.0505 |
| autonomous terminal | Q(data) | 0.6961 | +0.2273 |
| autonomous | Q(actor) | 0.6019 | +0.0740 |
| policy-terminal | Q(actor) | 0.6176 | +0.0828 |

这些值说明 critic 已不再系统性反向排序，但样本量只有 6 条自主成功与 18 条自主失败，仍属于可上线小范围验收的基础，不是充分的泛化结论。

## 统一网页交付

- 三相机按同一 generation 原子换帧，最大允许时间偏差 120 ms；单路冻结、过期、不同步和断线均显示明确告警并自动退避重连。
- loss 图使用真实 training step 为横轴，分别显示 actor/critic loss、Q1/Q2/target Q、weighted BC/Q/delta 和 replay 样本构成。
- Episode Q 图显示 autonomous success、autonomous failure、HIL success 随 episode chunk 位置的中位曲线和 AUC/separation。
- actor 服务每 10 个 decision 在后台线程计算 actor Q、reference Q、horizon/joint delta；控制 RPC 不等待 critic。日志超过 2 MiB 自动保留最近 500 行。
- 更新进度显示等待 episode、replay 构建、训练百分比、固定 gate 评估、接受/拒绝及原因。
- 视频下方时间轴支持拖动、靠近节点吸附和点击节点查看关键帧。
- 普通采集与 RLT 目录支持最近路径、前缀搜索、子目录候选与不存在目录的显式创建。
- 系统页只允许固定脚本：CAN、机械臂节点、相机、Home、Recover、reference/frozen/online RLT、Session stop 和 down；确认 token 60 秒、一次性、参数白名单。状态由实际 CAN/ROS/process 只读探针决定。

## 制品与指纹

- release：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/rtc-v5/releases/rtc-recovery-mc-critic-v1-20260922.json`
- checkpoint：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/recovery-20260922-mc-terminalq/checkpoints/step_16135.pkl`
- value audit：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/recovery-20260922-mc-terminalq/evaluation_16135.json`
- release audit：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/recovery-20260922-mc-terminalq/release_audit.json`
- current pointer SHA-256：`a48d3e52a4a36f7e60fef48bd6aa6819b593800fb5bf6ae7da41906165db3b67`
- runtime patch：`r15-mc-terminal-hil-conservative-telemetry-20260922`

不可变选择器已核验 learner step 16135、actor 7067、critic target `mc_success`，代码、checkpoint、norm stats 和 Stage 1 manifest 指纹均匹配。

## 验证

- RLT pure replay/target/gate/cycle：21 passed。
- learner core：13 passed。
- Cobot platform Python 全套：431 passed。
- Node 前端：23 passed（Cobot 自带 Node 10 对 `node:test` 使用兼容测试 runner）。
- shell 语法、`git diff --check` 通过。
- CPU actor 恢复、推理与异步 Q/delta telemetry 通过；测试 telemetry 已删除。
- 统一网页主页及 config/cameras/devices/diagnostics HTTP 均为 200。
- 实测只读状态：Task2 五臂 CAN 5/5 up；机械臂、相机、RLT 未启动。网页测试后已停止。
- 没有执行机器人动作或现场成功率测试。

## 现场流程

在 `/media/agilex/Getea1/jiaan/projects/cobot-platform`：

1. `./scripts/ui_up.sh`，打开 `http://10.7.165.64:8015/`。
2. 用系统页或既有命令完成 CAN、机械臂、三相机和 plug Home；必须确认三相机均为绿色且画面持续刷新。
3. 先执行 `./scripts/rlt_demo.sh --no-record`。这是当前 release 的冻结 actor，不录数据、不在线更新。固定排插、插头夹持和相机位姿做 3–5 条安全验收。
4. 网页结束 Session 或终端 Ctrl+C 后，执行 `./scripts/rlt_up.sh`。初始仍是同一 actor 7067，记录并启动五条批次在线更新；不需要 `--restart`，除非遗留 Session 明确存在。
5. 成功/失败按真实终局标注。HIL 只在“策略已经偏离但仍可恢复”时介入；无需人工凑 episode 比例，sampler 会在 transition batch 中固定约 20% HIL。
6. 每五条新 validated episode 后，页面会显示训练进度与 gate 结果。accepted 候选从下一条 episode 生效；rejected 候选不会覆盖当前 release，也不会阻塞后续新批次。
7. 初始阶段不加 `--explore`。Ctrl+C 或 `./scripts/rlt_stop.sh` 只结束 Session、保留模型；关机前使用 `./scripts/rlt_down.sh` 释放全部 RLT 进程和显存。

## 验收与后续 TODO

- [ ] 冻结 actor 7067 在固定场景做 5–10 条纯自主基线，记录成功、偏孔方向、自动暂停、deadline/reanchor 和 HIL。
- [ ] 在线模式做三批、每批五条；每批至少包含真实自主失败，HIL 只救可恢复偏差，成功/失败标签不人为配平。
- [ ] 每批核对 actor version 是否变化、candidate accepted/rejected 原因、early/terminal Q separation、actor/reference Q 和 delta。
- [ ] 用下一批冻结 5–10 条比较 actor 7067 与最新 accepted actor；只有固定场景纯自主成功率或终点误差改善才称在线学习有效。
- [ ] 若连续三批均被 gate 拒绝，先查网页显示的具体门限，不调大 Q 权重或探索噪声。
- [ ] 若 gate 接受但真机无改善，优先检查输入分布和终点观测可辨识性；再决定补左相机/夹持标定或重新训练 Stage 1。
