# 2026-09-20：原仓库 RLT + RTC 真机验收候选

当前交付是 **offline_validated_onsite_pending**，不是已证明真机成功率提高的模型。此前 corrective/IQL 候选继续撤销，不恢复其入口。用户先固定评估，再运行在线更新。未发机器人命令、未重启 CAN/机械臂/相机、未访问 HPC。

## 交付入口

在 Cobot：

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_up.sh
./scripts/rlt_demo.sh
```

使用统一网页 8015，等待 ready/disarmed，现场确认后 arm → 开始 Session。固定候选 `rtc-upstream-r2-5000-linear`，日志应显示 step 5000、runtime `r9-upstream-rtc-smdp-20260920`；网页 actor_version=5000。此数为 critic/global updates；actor 实际更新 2500 次，不与旧 actor5000 混为同一模型。

先 1 条小范围自主验收，再 3–5 条固定场景记录。确认方向、平滑性、暂停和 HIL 接管正常后结束 Session/终端 Ctrl+C，再：

```bash
./scripts/rlt_up.sh
```

默认在线、无探索。每 5 条新增已验证成功/失败 episode，在 waiting_scene 等空闲边界持学习锁更新；每条新训练 transition 对应 5 次更新，单批上限 2000。沿用全部历史训练 replay 与原 stratified sampler，不是只训练这五条。固定 UUID 留出不训练。放弃/shadow/no-record 不参与。初始已有 86 条 rollout UUID 已登记消费，避免一启动就重复更新。

通过前后同输入动作/平滑性回归检查才发布；失败保留当前权重与数据，等待至少五条额外新数据再尝试。通过后**下一条 episode**加载新版本；当前 episode 不换参。`--frozen-actor`/`rlt_demo.sh` 整个 Session 固定选定版本。未宣称每次更新必然提高真机成功率。

对照：`./scripts/rlt_up.sh --reference`。纯评估可加 `--no-record`，不进入 replay。首次验收不开 `--explore`。Ctrl+C/网页结束只停止 Session，保留 Stage1；新终端 `./scripts/rlt_stop.sh` 可结束遗留 Session；普通已结束后无需 --restart。全部结束/关机用 `./scripts/rlt_down.sh`。

安全界限保留：30Hz、右臂关节、左臂/夹爪固定；.10rad/s、.9rad/s²、tracking .04rad、既有工作空间包络、每轮360条自主命令（12秒）。达到预算或保护边界仍会暂停，并显示具体原因；不是自动判定失败。结束成功/失败后的既有 all plug 归位由现场流程触发，本次未执行。

## 本次实质修正

1. 回到 pinned `Yyshadow/openpi-RLT` `c1e40ac360185778c98cf20da2820e22d2d415e7` 的 direct Gaussian actor、twin critic、BC−Q+delta actor loss、target/actor每两步更新及 stratified sampler。上游 checkout clean；没有改上游源码。撤掉自创 IQL/residual 作为本候选的学习核心。
2. RTC 适配是显式扩展：99维状态包含原14维state、已承诺prefix相对state、delay；右臂7维放在前面以满足原 action delta adapter。Critic TD 使用真实 decision duration、gamma^duration、可靠终局；断裂边 censored，不伪造 reward。原C10条件两步 params/metrics/RNG 与上游对照通过。
3. 重新调用冻结 Stage1 生成准确 RTC prefix 对应 z/ref；不是只改旧cache的context。169文件、168源UUID、4777行，3859条 train TD有效；保留两条历史自主成功于训练集。一个无可靠终局的失败数据保持 censored。float32 factual replay 避免绝对关节命令 fp16 量化。
4. RTC队列只提交 actor 负责的10条命令，删除混入的 reference 尾段。请求tick改为0、4、14、24…，提前6步推理；guard只检查即将执行的 actor C10，而非将被替换的尾段。暂停/HIL generation 丢弃旧计划，控制时钟不追赶突发发令。
5. 部署端对 actor-reference 修正做时间线性投影，再使用原共用限速/加速度滤波。这是明确标注的部署平滑扩展，**不是论文原生模块**，无放宽安全界限。评估同时包含 old warmup 加同样投影，避免把纯滤波收益冒充学习收益。
6. CPU专属 actor worker 与 JAX learner 分离；Stage1常驻。checkpoint、norm stats、运行源码、Stage1 manifest 均指纹核验。训练完整恢复 actor/critic/targets/optimizer/RNG，在线固定norm stats。发布descriptor不可变，pointer原子切换。

## 参数与证据

Stage1继续已训练的三相机训练时RTC checkpoint `projects/rlt/checkpoints/plug_v2/bf16/3999`，本次没有重新训练VLA/encoder/decoder。新 warmup：5000 global /2500 actor updates，batch128，lr1e-4，gamma .99/控制tick，std .002，ref dropout .5，tau .005，BC/Q=10/.1；online=5/.1。delta_weight=300 是明确的离线平滑单因素调整（上游默认10），不是声称原论文参数。5k是目前验收候选，不声称全局最优预算或20k一定更差于此99维模型。

所有数字为复用UUID回归集，同输入反事实，不是独立闭环评估：

| 指标 | reference | old warmup（同线性投影） | 新候选 |
|---|---:|---:|---:|
| d0 同滤波目标MSE | 3.429e-5 | 4.119e-5 | 3.157e-5 |
| d6 同滤波目标MSE | 9.035e-5 | 9.237e-5 | 4.060e-5 |
| d0 实际human命令MSE | 3.086e-4 | 2.937e-4 | 2.955e-4 |
| d6 实际human命令MSE | 2.175e-3 | 2.173e-3 | 2.102e-3 |

同滤波目标改善约8%/55%，但对实际human命令改善小，d0仍略差于旧warmup。d6滤波后加速度RMS比reference约高10%，不能说所有指标都更平滑。真实部署限速与加速度边界检查通过。

Q：验证11条后续HIL成功/8失败最初三条policy decision均值 .199/.026；训练46成功/21失败 .189/.050。**没有独立自主成功留出**，成功中包含后续人救回，不将该分离称为自主成功概率。最后batch human_mask_ratio=.547、bc_human_penalty=.0300、bc_ref_penalty=.00476；human含专家示教，原stratified池重叠，不代表真实HIL恰占20%。

验证：34 pytest +3 upstream unittest；实际Runtime.infer +真实Stage1 +CPU HTTP actor，在4保存场景的理想命令跟随fixture共416命令通过，热推理约97–109ms，低于200ms预留；没有ROS publisher/实机动作。HTTP与直接CPU推理46例完全一致；CPU与GPU batch浮点最大差 .000247rad（阈值.0005rad），不称位级一致。

完整五条历史episode隔离在线批次：97条train TD transition，485 updates，5000→5485，训练与评估22.14秒；动作误差门槛通过，norm stats SHA不变，发布仅指向testing pointer。**不是新在线效果，不部署该5485测试权重**。下一轮真机数据准备耗时另计，不能承诺每批22秒。初始交付仍5000。

## 资产与后续

Cobot方法根 `/media/agilex/Getea1/jiaan/projects/rlt`：
- selector `runs/plug_v2/learning/rtc-v5/current.json`；descriptor `releases/rtc-upstream-r2-5000-linear.json`。
- checkpoint `runs/plug_v2/learning/rtc-upstream-v4-delta300-20260920/checkpoints/step_5000.pkl`，SHA `a6273d48183df852863c3e7d44676e7592414a9cf20c0f097a20909fbf0023da`。
- 同run的 config/norm_stats/metrics/evaluation；完整模型并非A6000文档副本。
- `runs/plug_v2/diagnostics/rtc-upstream-20260920-v4/`：manifest、candidate-comparison（逐episode Q曲线）、runtime-audit、actor-parity、delivery-manifest、validated-source.tar.gz、offline-diagnostics.png。
- 在线状态 `runs/plug_v2/learning/{operation.json,rtc-v5/online_cycle.json}`；每批独立rtc-online目录，审计/父版本/数据UUID可追溯。
- 代码来源/哈希见本项目 manifests/2026-09-20-upstream-rtc-delivery.json。

剩余由现场验证：固定5000能否稳定纠正偏孔；同复位分布 reference/5000的无HIL成功率与终点误差；之后分版本记录在线更新趋势，区分自主成功/HIL成功/失败/保护暂停。若异常，不继续盲采或放宽guard，保留本轮trace。离线通过并不能保证未知场景安全或学习持续提高。
