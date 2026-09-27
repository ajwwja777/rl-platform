# 2026-09-20 因果 RTC 被动动作契约修正与在线 RL 真机验收交付

## 结论

当前发布 `rtc-causal-passive-v1-step5000` 已通过数据、训练、发布指纹和无动作真实 Runtime 离线验收，具备小范围真机验收条件。状态仍为 `offline_validated_onsite_pending`：尚未声称真机插入成功率提高，也未声称在线更新一定单调改善。旧 corrective/IQL 与此前动作契约错误的候选不恢复。

Cobot 当前指针：

- release：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/rtc-v5/releases/rtc-causal-passive-v1-step5000.json`
- release SHA-256：`57c2cb01b4fc0d1dfe989042ad1b124422608e95b7c51dc6aa29c1e14af3b3dd`
- checkpoint：`runs/plug_v2/learning/rtc-causal-passive-d300-b1000-20260920/checkpoints/step_5000.pkl`
- Runtime revision：`r11-passive-command-latch-20260920`

## 根因与修正

此前部署把实测的左臂和夹爪 state 直接复制进 action prefix。plug_v2 示教中的左夹爪 action 基本为 `0.0002`，而实测 state 为 `0.0003`；该维训练分位数退化为 `q01=q99=0.0002`，按既有 `+1e-6` 归一化后，state-as-action 约为 199，形成严重分布外前缀。离线复现中，错误前缀的 d6 边界跳变中位数为 `0.1277 rad`、p90 `0.1820 rad`。

Runtime 现在在每轮第一个 d0 Stage1 proposal 中锁存左臂和双夹爪的**动作空间命令**，后续 d6 prefix 与最终命令始终复用；新 episode 才重置。右臂继续保留 `0.1 rad/s` 速度、`0.9 rad/s²` 加速度、`0.04 rad` tracking 和工作空间保护。修正后 d6 边界跳变中位数 `0.00359 rad`、p90 `0.00740 rad`，与 teacher 的 `0.00345/0.00763 rad` 接近。

训练 replay 同步改为：

- decision clock 为 d0@start、d6@start+4+10k，不再混入错误的 4/6 交替 TD 跳数；
- policy chunk 使用实际下发命令，human chunk 使用人工命令；
- policy/HIL 特征均使用 d0 被动动作锁存；
- terminal chain 最多向前平移9帧，使最后一个完整 chunk 精确结束于终局，不做动作 padding；
- 小于等于1秒的 HIL handover gap 以真实 wall-time duration 连接 SMDP credit；更长暂停保持 censored；
- 不为缺失终局、空片段或断裂边虚构 reward/next state。

最终 factual 数据集含158个 episode、1825行、1810条有效 TD、1349条 human、476条 policy、151个终局行、131个正奖励行。固定留出含11条 warmup success、4条 warmup failure；1个空专家片段与7个无可靠终局的 success 仅作 censored/BC 证据。数据审计 `passed=true`。

## 训练与离线证据

采用 pinned upstream RTC actor/critic 结构，训练5000 global updates；前1000步 actor 只做 BC+smooth、critic 建立价值，后4000步联合 BC−Q+delta，`delta_weight=300`，stratified sampling。Stage1 三相机 checkpoint `bf16/3999` 保持冻结。

固定同输入留出结果：

| 指标 | Stage1 reference | 旧 warmup | 新发布 |
|---|---:|---:|---:|
| d0 deployment-matched MSE | 6.715e-5 | 8.365e-5 | 5.136e-5 |
| d6 deployment-matched MSE | 3.567e-5 | 3.841e-5 | 2.667e-5 |
| d0 acceleration RMS | 2.458e-4 | 3.051e-4 | 2.771e-4 |
| d6 acceleration RMS | 4.061e-4 | 4.081e-4 | 4.084e-4 |

新发布相对 reference 的同口径 MSE 改善约23.5%/25.2%，相对旧 warmup 改善约38.6%/30.6%。早期 policy Q 的11成功/4失败留出：AUC `0.9091`、均值分离 `0.1498`；terminal AUC `1.0`。这些是复用的 HIL/失败固定留出，不是独立自主成功率。

真实 `Runtime.infer + Stage1 + HTTP actor` 在5个保存场景各执行104条理想跟随命令：请求 tick 均为 `[0,4,14,...,94]`，推理约97–115ms，均低于200ms deadline；被动维恒定，右臂最大速度不超过0.1rad/s，审计 `passed=true`。在线 resume smoke 从5000更新到5032时发布门禁通过，早期 policy Q AUC从0.909升到0.932；该 smoke 未切换生产 actor。

最终新鲜验证：38项轻量 pytest、10项在线更新 unittest、19项 replay unittest、9项 Torch RTC核心 unittest及 `compileall` 全部通过。发布重算验证19个生产源码哈希、release/checkpoint/norm 哈希均匹配。

## 在线更新合同

当前75个既有 validated warmup UUID 已全部写入 `consumed_uuids`，pending为0。正常 `rlt_up.sh` 不会启动后立即重训旧数据。每累积至少5条新的 validated success/failure episode，只在 `waiting_scene/stopped/offline/ready/disarmed` 边界运行；rollout/HIL/finalizing 时不扫描 replay、不哈希 checkpoint、不写 ledger。

每批更新数为 `min(2000, 5 × 新增事实 transition)`。发布门禁同时检查：固定留出 deployment MSE、command acceleration、早期 policy Q AUC/分离、terminal Q AUC/分离及 actor/critic finite。拒绝时保留当前 actor；接受后只在下一 episode 边界加载新版本。固定 norm stats，不在当前 episode 热切换。

## 现场验收流程

在 Cobot，先按既有现场流程确认 CAN、机械臂节点、三相机、plug 初始位姿与急停；本次 agent 未执行这些动作。

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_up.sh

# 第一步：固定当前发布，只看方向、平滑、暂停/HIL；不写入训练数据
./scripts/rlt_demo.sh --no-record

# 结束固定验收后：真实在线RL，默认确定性，不加 --explore
./scripts/rlt_up.sh
```

在线首批建议5条：若自主仍基本失败，做2条自主失败、3条较晚但仍可恢复的HIL成功；避免在已经不可恢复后才介入，也避免过早接管。第5条固化后在网页等待 update `accepted` 或 `rejected`：accepted 后下一轮 actor_version 应大于5000；rejected 则保持5000。不要仅凭 loss 判断，记录每个 actor_version 的自主成功/HIL成功/失败、终点偏差和保护暂停。

`Ctrl+C` 只结束当前 Session，保留预加载 Stage1；普通已结束后再次运行不需要 `--restart`。关机或释放GPU才运行 `./scripts/rlt_down.sh`。首次验收不开 `--explore`。

## 仍需现场回答的问题

- 固定发布在当前 camera/夹持/插排复位分布下是否稳定接近插孔并保持平滑；
- 五条新数据后的发布门禁是否接受，以及下一版本同场景自主成功率或终点偏差是否改善；
- HIL介入时机、物理接触随机性与复位一致性是否限制 Q 的可分性。

本次只执行离线训练、推理、测试、发布与文档更新；未启动 ROS/CAN/相机、未 arm、未归位、未发机器人命令，未访问 HPC。
