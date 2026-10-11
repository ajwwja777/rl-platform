# 固定 HIL8 探索采集交付

状态：已交付可选探索采集入口；等待用户加载并进行首批采集。不是持续 Online 学习放行。

代码提交 `121f0f8b0d69f067a28d201b58831e0db6e8edce` 已合入 A6000 main 并推送。Cobot 8 文件逐 SHA 同步，现场 CPU 检查 52 项通过，实际配置原生解析通过；网页 `/api/collection/model` 已显示新入口 available=true、training_enabled=false。11 个受保护资产 SHA 与监测服务 PID/start ticks 未变。安装不启动服务或加载模型。回执：`outputs/exploration-collection-20261011/field-install.json`。

## 做了什么

新增可选入口 `plug-v3-hil8-explore`，显示「插孔 · HIL8 探索采集（固定权重／不训练）」。固定 HIL8 Learner7281 / Actor3640，基础 Actor 确定性；仅 latched collection 加探索，evaluation 不加。Learner 与 phase controller 禁用，旧入口/默认参数不变，未合入 AWBC W2。

探索沿用另一对话原型的思路，重新实现为规划线程中的无文件 I/O 偏移：六关节 sigma=0.15*(q99-q01)/2，rho=0.5，截断 ±2sigma；每段 plan 同偏移，暂停/接管 epoch 后重置相关状态。关节2 sigma≈3.429mrad，上界≈6.859mrad；不是 TCP 位移承诺。夹爪不加探索、保持原夹爪锁定。保留逻辑20Hz/C10、物理50Hz、RTC 与 EMA，既有限速/拒绝发布保护继续生效。

补齐了原型没有的部署合约：冻结权重与探索开关独立；Actor/stat SHA 固定；Actor来源/版本错误即停止；禁止训练 branch 继承；专用 Replay/录制/日志；不在规划热路径同步写 JSONL。探索来源随队列行传播，RTC 保留前缀保留旧 request/offset；记录规划前 Actor、偏移、实际物理发布与执行后反馈。纯 HIL 没有虚构 Actor 反事实。

## 证据与边界

- 已验证：A6000 101 项相关 CPU 检查通过；最后附加代码身份后 11 项专项复核通过。包括新探索 native EnvDriver→真实 Replay 转换，并以极小随机网络检查原生 Replay schema 可用于 CPU train_step（非生产权重训练）。
- 已验证：既有 model registry 每一旧行完全保持；现场预检源码与基线 c731184 一致，同步前后部署 offline/无Session、录制 idle。该时点状态不替代日后实际状态查询。
- 失败（已解决）：现场隔离测试首次 2 项配置相对路径错误、50 项通过。修正安装测试目录的 configs 只读链接后 52 项全通过，生产实现未因此改动。初次失败和复测日志均保留。
- 证据不足：新探索入口尚无真实加载/真机采集数据，未证明自主成功率提升；没有独立 TEST。单元测试及重叠窗口不能当 Episode 成功率样本。
- 既有问题：AWBC W2 的重复 DEV 保持尚不稳定，本次不发布其权重；本次固定模型采集用于获得不同动作与结果证据，再离线检查 Critic 动作偏好和单因素训练候选。

完整 CPU 命令/日志：A6000 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/exploration-collection-20261011/`。工作副本和同步脚本：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/exploration-collection-20261011/`。源原型保留原 online-review worktree，不替另一对话提交。

## 资产与输出

- Cobot 只读 Actor：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_hil8_clip_20261010/actor_snapshot/actor_snapshot.pkl`；SHA256 `020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5`。
- 对应 stats：同候选目录 `action_norm_stats.json`；SHA256 `4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c`。
- Cobot 新录制：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/exploration-hil8-7281-v1/`。
- 新 Replay：`/home/agilex/jiaan/project/rl-platform/outputs/exploration-collection/hil8-7281-v1/replay/collection.pkl`。
- 新 trace/指标：同 `hil8-7281-v1/` 下 `traces/`、`online/metrics/`。不会写入旧生产 Replay 或发布新 Actor。

## 给采集负责人的操作

1. 刷新网页，在「数据采集」选择「插孔 · HIL8 探索采集（固定权重／不训练）」（ID `plug-v3-hil8-explore`），加载模型。不是 HIL8 分批 Online、也不是 supported MC30 7000 Online。
2. 保持 50Hz＋RTC＋滤波，启动采集 Session；本批固定模型，不启用训练或发布候选。加载后核对 Actor3640、learner_enabled=false 与探索 sigma；如加载或身份检查报错，保留报错给执行方，不切回 Online 继续试。
3. 先采 3–5 个完整 Episode 核对链路，再累计20–30个探索 Episode。记录插孔位置类别及幅度；成功/失败照实标记，偏离时正常接管，保留 HIL 标签，辅助成功与自主成功分开。不要为了成功率只保存成功轮。
4. 这批全部是后续训练/开发候选，不命名独立测试集。新数据先做执行来源/偏移/HIL/结果核对，再做 Critic 配对诊断及离线候选对照。独立冻结验收另留未参与选型的数据。
5. 明显异常偏移、持续触碰/堵转、控制或 trace 超时则暂停并保存故障证据；用户掌握停止与接管。本交付不自动运动。

回退：结束当前 Episode、释放新入口，选择既有 `plug-v3-hil8-7281`；新采集隔离保存。若需源码回退，使用安装回执的原文件备份且重新核对现场空闲。不得把新 Replay 并回生产以实现“回退”。

## 2026-10-11 用户首批采集后的纠错

已有3轮真实完整录制，Actor3640及探索元数据可见，但本入口固定录制目录错误地覆盖了网页选择。A6000补丁改为网页显式选择优先（无选择才用上述专用默认）；15项CPU验证通过，当前活动现场未更新。详见[保存位置核对](STORAGE.md)，不要把A6000修复当当前进程已生效。旧数据无需移动。
