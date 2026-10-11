# 第三批探索采集：29 Episode 数据链与下一步

日期：2026-10-11。状态：本批只读核对完成；新离线训练对照尚未执行，未放行持续 Online 学习。用户确认中途移动过插孔，移动起始 Episode、方向和幅度未知，不补造分组。

## 结论

| 主张 | 状态 | 直接证据与边界 |
|---|---|---|
| 新经验已保存并进入私有 Replay | 已验证 | 29 个完整录制，1068 条 journal；1068 个唯一 native Episode/step，全部与 strict 输入回执逐数组 shape/dtype/SHA 相符，无不匹配 |
| 已发生 Online 学习 | 失败 | 当前 plug-v3-hil8-explore 是固定 Actor3640，Learner disabled，实际训练 batch 为 0；本次成功不能归因于学习 |
| 两条 HIL 的纠正段进入 Replay | 已验证 | 原始录制20/25，79/81个独立逻辑步全部被窗口覆盖；共19个含人类动作的窗口，只是2个独立Episode |
| 位置泛化或自主成功改善 | 证据不足 | 中途移动目标且位置标签缺失；没有固定位置、固定执行的匹配比较，也没有独立TEST |
| 收尾精确缓存实际生效 | 已验证 | 用户重载后的native39：56 hits、62实际feature requests，构建9.017秒；这不是相同输入A/B速度实验 |
| 末端往返原因已修好 | 证据不足 | 本轮未改执行/模型；新数据仍有Reference J2切换跳变，成功失败均出现，不能唯一归因于探索、RTC或纯Stage1 |

## 身份、配置和范围

- A6000代码基线：e0f4a0df33897eb6e3cff5ec5bcd4e11d9a6ff3a；缓存源码提交5517f2575262ae422002f518cbc3674c03909cc5，网页源码c2c93d5ba540667b4937c5fbf391fff2f27b03d8。此前同步证据见FINISH_AND_MOTION.md和finish-history/field-sync.json。
- 模型入口plug-v3-hil8-explore；固定HIL8 Learner7281来源的Actor3640。Actor文件：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_hil8_clip_20261010/actor_snapshot/actor_snapshot.pkl`，SHA256 `020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5`；配套stats SHA256 `4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c`，身份承接既有交付收据，本轮未重新加载权重。
- 实际执行逻辑20Hz/发布50Hz，RTC开启/replan5，EMA tau0.08；六右臂关节相关有界探索rho0.5、2sigma截断，J2 sigma3.429mrad/上界6.859mrad；夹爪固定，基础Actor确定性。Stage1 key42，reference_4999。Learner关闭。
- Cobot原始录制：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/exploration-hil8-7281-v1`。
- Cobot运行根：`/home/agilex/jiaan/project/rl-platform/outputs/exploration-collection/hil8-7281-v1`；Replay为`replay/collection.pkl`，证据为`traces/`、`traces/replay_inputs/`、`online/metrics/`；解析配置为`unused-checkpoints/online_rl_config.yaml`。
- 最后录制29 UUID：37ecdb21-3e4c-409e-8d63-7064ade12dbc；新Session c1e55230-6ba5-45de-881d-82e133fa8015，快照时waiting_scene/policy_paused。录制ID和原生Replay Episode ID不同；以下跨ID对应由时间核对，不能冒称原生UUID连接。
- Replay读取期间大小/mtime稳定；10,373,484 bytes，SHA256 `919ff1adb31b6fcae0f3b3380ca80dc2e881b7f2501566d7042237674d0dcad8`。只读原始字节核对，未调用ReplayManager写入/重存。

## 数量与分布

| 集合 | 自主成功 | 辅助成功 | 失败 | 合计 |
|---|---:|---:|---:|---:|
| 前次完整分析的1–11 | 5 | 0 | 6 | 11 |
| 新增12–29 | 6 | 2 | 10 | 18 |
| 当前1–29 | 11 | 2 | 16 | 29 |

此前目录修复已看到27条历史，本轮相对该UI快照只新加28失败、29成功；“新增18”以此前完成行为分析的11条为基线。总2883帧，新增18条1677帧。11次中断/放弃且无Replay提交的尝试单列，不算完整失败；其中一次中断含117个human步，不混入本批有效HIL。

数据为后续TRAIN/DEV候选，尚未锁定拆分；没有独立TEST。以下只是当前数量，不以重叠窗口计算成功率区间；位置变化未知，不拟合前后学习趋势。

![数据构成](../../../../outputs/exploration-collection-20261011/batch3/batch-overview.png)

| 私有Replay按完整Episode结果分组 | transitions | 占比 |
|---|---:|---:|
| 自主成功Episode | 532 | 49.81% |
| 辅助成功Episode | 46 | 4.31% |
| 失败Episode | 490 | 45.88% |
| 总计 | 1068 | 100% |

成功Episode合计578/1068=54.12%；与失败约54:46，不是80:20。新增611条。这个buffer都是本次探索采集，没有专家条目，不等同旧Warmup/生产训练buffer。

动作来源是另一维：policy-only1049、mixed3、human-only16。因此HIL-containing19/1068=1.78%；辅助成功Episode不是每个动作都人工，也不是每个动作都正确。若将来在该buffer有放回均匀采batch128，期望2.28个含HIL窗口，约10.05%的batch没有HIL；这是采样假设下的计算，实际训练batch目前为0。原生success字段只有13个终端为1，其余1055个为0，不能据此计算失败占比。

录制标签failure同时带episode_quality=bad、keep_for_training=false，而该16条失败已实际进入RLT私有Replay。录制筛选标志与RLT入库不是同一个口径；不要用这些标签静默过滤RL失败样本。未在本次改标签。

## HIL、末端动作和时序

- 原始20/native25：单段HIL79步，约4.086秒；25条Replay为policy16/mixed1/human8，覆盖全部79步。
- 原始25/native35：单段HIL81步，约4.180秒；21条Replay为policy11/mixed2/human8，覆盖全部81步。
- HIL动作数值有限；实际hil_target_mode=feedback，缺少hil_command_receipt不是本模式下单独的缺证故障。接管期间陈旧Actor计划不能当作同状态反事实Actor输出。
- 两段HIL的整段右臂J2反馈净变化约+64.752/+71.887mrad，包含完整人工推进过程；不可把它解释为应瞬时增加的探索噪声或最优纠正幅度。腕部图像与接近/插入过程相容，实际成功来自用户标签，未通过图像独立验证插入深度/接触力。
- 当前HDF5通用action受双臂联合有效性置NaN；诊断使用rollout/coordinator_command右臂7:14及coordinator_right mask，状态为observations/qpos右臂；Replay来自数值trace，不直接取该NaN字段。录制10Hz、逻辑20Hz、发布50Hz不能混为同一时基。
- 新完成轮次未发现新增超过40ms的发布间隔，末轮最大30.647ms；不等于证明所有现场过程无故障。新数据中Reference末3秒J2大跳变在成功和失败轮次均有，尚无唯一因果结论；见hil-motion.json。此前raw10的每.25秒交替跳变见FINISH_AND_MOTION.md。
- 新模型日志`/home/agilex/jiaan/project/cobot-web/runtime/deployment/model-20261011T214804.log`，COBOT_REPLAY_BUILD native39：hits56、requests62、hash0.236s、inference8.492s、build9.017s、transitions59。标签到rollout指标9.088s，录制结束到指标12.405s。仍需等待，不能说已到1秒。不同Episode窗口/输入不同，不能用旧20.68秒直接算受控加速比。

## 下一步建议与判定边界

当前已足够启动首轮离线诊断，不必为凑总量继续无差别采集；两条HIL只能支持局部纠正拟合/方向诊断，不能支撑独立泛化结论。新实验本轮尚未执行。

1. 固定本次数据身份和现有Actor，对HIL20/25有效状态重算Reference、当前Actor、人类反馈动作的配对Q1/min-Q及TD目标，区分即时奖励、bootstrap、动作支持程度。只在同一有效状态比较；历史人类动作不冒充全局最优动作。
2. 先建立冻结Actor基线，再在相同数据、初始化、采样、预算、seed与保持损失下比较纠正BC与BC+Q，唯一改变Q项；锚定/lr对照另做，不混在一次实验。采样按Episode记录真实batch身份，同时报告HIL、近数据和成功/失败重叠比例，避免误把buffer占比当训练占比。
3. 评估纠正误差/方向/幅度与旧场景保持，检查Q是否引导偏离有效纠正。两条HIL可轮流留一条做诊断，但只有2个独立Episode且反复选型，只算DEV，不称独立TEST。位置标签缺失单列。
4. 离线通过仅产生冻结候选；当前冻结权重保持回退。真实受控验收再比较自主/辅助成功，之后才决定分批Online学习；不靠更高ActorQ、低TD或AUC单项自动晋升。若BC有效而加Q变差，支持优先修Critic/关闭该Q指导；若两者都无效，先查覆盖/观测和动作对齐，不立即加大探索。

## 复核命令与产物

所有本批分析输出位于A6000：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/exploration-collection-20261011/batch3/`。

只读采集命令（在A6000执行；重跑会观测新现场，避免覆盖本快照）：

```sh
ssh agilex@10.7.165.64 python3 - < snapshot.py
ssh agilex@10.7.165.64 /home/agilex/jiaan/project/cobot-web/.venv/bin/python - < reduce_field.py
ssh agilex@10.7.165.64 /home/agilex/jiaan/project/cobot-web/.venv/bin/python - < reduce_hil.py
ssh agilex@10.7.165.64 /home/agilex/jiaan/project/cobot-web/.venv/bin/python - < check_replay.py
python3 plot_summary.py
```

产物：snapshot.json、data-analysis.json、hil-motion.json、replay-verification.json、summary.json、operator_context.json、batch-overview.png/jpg、wrist-ep{20,25,28,29}.jpg；脚本同目录。分析脚本修正None探索字段处理、旧matplotlib/PIL导出兼容；没有修改生产代码。未启停服务、加载模型、占用现场GPU、运动、改权重或Replay；未在笔记本保存新文件。

## 完整录制身份（标签来自用户，非独立成功检测）

| 录制 | UUID | 结果 | HIL帧 | 帧数 |
|---|---|---|---:|---:|
| 1 | 164b2bd2-4cee-47ca-89e8-a8b3c5a4e99a | failure | 0 | 82 |
| 2 | bcaef048-d842-47ce-8b55-ab11fc95f397 | failure | 0 | 98 |
| 3 | 01187a85-44df-4b52-9ec8-b8380d3d34a3 | failure | 0 | 126 |
| 4 | ebd35e5b-05c4-4f0e-b13b-d914f609e927 | failure | 0 | 79 |
| 5 | d20b5155-1756-4777-ad98-4e8c52bb5fed | failure | 0 | 96 |
| 6 | cb3b0c13-3a8e-43e9-a512-50d7a7a6850e | failure | 0 | 57 |
| 7 | e48ed143-ca1e-4299-8930-4563c96af464 | success | 0 | 123 |
| 8 | f154fff4-c772-4055-9351-56bd7e675921 | success | 0 | 110 |
| 9 | 14060abb-12bd-4fe3-ae3d-a74ff26ff90c | success | 0 | 177 |
| 10 | 56da005e-d27f-4cea-9dd5-dea937cb24d4 | success | 0 | 142 |
| 11 | 8319b3c8-426a-44b0-984e-9e74350cfa67 | success | 0 | 116 |
| 12 | 177ea14c-c2aa-43b0-9fc4-3c9d4e4fd004 | success | 0 | 198 |
| 13 | 51d2301a-387c-4d66-95df-0447cba98c15 | success | 0 | 111 |
| 14 | 78c25ec9-335f-4ede-ad62-3cc4cd4bf669 | success | 0 | 73 |
| 15 | 2db91deb-9943-4615-9b40-c41e9efae484 | failure | 0 | 78 |
| 16 | 128ee0aa-f5dc-4efa-b770-7824baec6d5d | success | 0 | 97 |
| 17 | 086bc75d-ec03-4cb3-accc-0cc1e9c856ff | failure | 0 | 92 |
| 18 | 77d558a7-bb82-4434-aeaa-77b9b14d9741 | failure | 0 | 91 |
| 19 | 34ba67d6-8e64-40e3-bd65-6dd92809eb68 | failure | 0 | 63 |
| 20 | 7981f8bc-52b6-49a5-a474-fdb44f011229 | success | 41 | 88 |
| 21 | 4d4c6f77-2cb3-4ff2-ac21-d17282083cfd | failure | 0 | 92 |
| 22 | 7605e740-8704-4a54-81c8-320de26b31c9 | success | 0 | 93 |
| 23 | 1e524a0b-b060-47c9-9238-b300935ae8f5 | failure | 0 | 97 |
| 24 | 122f5fc9-5d24-473b-a79e-aacf00acbece | failure | 0 | 91 |
| 25 | 3231a3ca-8eaa-4047-8468-6dbbf0410fe7 | success | 42 | 78 |
| 26 | 3f46e530-885e-4127-8ad1-e25bb5bf8777 | failure | 0 | 45 |
| 27 | bb947a84-eccd-48af-b024-6df20181bf4c | failure | 0 | 49 |
| 28 | ce341664-8162-49ea-8421-c420122f2193 | failure | 0 | 87 |
| 29 | 37ecdb21-3e4c-409e-8d63-7064ade12dbc | success | 0 | 154 |
