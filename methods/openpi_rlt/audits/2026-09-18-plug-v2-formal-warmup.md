# plug_v2 首轮正式 warmup 完成（2026-09-18）

## 结果

当前任务明确授权 Cobot 4090 手动训练。11:57:25–12:12:56 UTC（北京时间19:57:25–20:12:56），完整流程931.24秒，优化器阶段174.79秒。完成20000 critic / 10000 actor updates；独立留出综合选优发布 actor_version=500（critic累计编号，对应250 actor updates），不是最后20000步。所有6个检查点均通过门槛，500步综合最优。仅离线验证，未启动机器人动作，未访问HPC，未提交/push。

## 数据与参数

82个专家原UUID/83片段、55 HIL成功、20自主失败；放弃20条不入训练。真实replay v2共158文件/7652行，其中teacher7165、policy487。训练6568行、验证1084行，134训练UUID/23验证UUID，交集为空。训练片段计数 expert75/success44/failure16，expert的两片段属于同一原UUID。冻结Stage1 Pi05/token encoder/decoder。batch128、Adam LR1e-4、critic:actor=2:1、physical gamma=.99、tau=.005、专家/成功/失败=.3/.4/.3、BC10/Q.1/smooth10、reference dropout=.5、右六关节 bounded residual=.05rad、30Hz执行条件化与限速；真实prefix/时间/BC mask使用v2契约。

## 固定留出指标

| 指标 | 未warmup基线 | 选中500步 |
| --- | --- | --- |
| heldout BC（平方关节误差） | 0.0002541116 | 0.0002144832 |
| smooth | 0.0000081816 | 0.0000087687 |
| critic TD | 0.2883037 | 0.0229599 |
| max command step（rad） | 0.0066666603 | 0.0066666603 |

BC降低15.6%；smooth略增加约7.2%，低于允许的30%门槛，不能声称更丝滑或真机成功率提高。该数据无自主成功，主要正向监督来自专家/HIL；真机应先做固定actor验收，避免把HIL成功计作自主成功。

## 制品与证据

执行根 `/media/agilex/Getea1/jiaan/projects/rlt`：

- 模型 `runs/plug_v2/learning/warmup/actor.pt`，36943083 bytes，SHA256 `17053265aceafac30c83c2dc176673ae6793fe78178f7421d4b3d5c330cb251f`。
- `learning/warmup/ready.json`、`learning/operation.json` 均accepted；selected500，learner_final_step20000。
- `runs/plug_v2/learning/formal-replay-validation-20260918.json`：真实特征finite、完整时间契约、UUID隔离通过。
- `runs/plug_v2/learning/formal-actor-validation-20260918.json`：23验证制品推理finite/形状正确、左臂与双夹爪输出保持reference、Stage1 checkpoint与stats匹配，优化器/target状态保存，未发布机器人指令。
- 日志 `runs/plug_v2/learning/warmup-20260918T115725Z.log`，末尾 `RL_ACTOR_READY 500`；launch凭据同目录 `launch-20260918T115725Z.json`。
- Stage1固定 `/media/agilex/Getea1/jiaan/projects/rlt/checkpoints/plug_v2/bf16/3999`。

训练PID1342702已退出，learning lease释放。Session仍stopped/policy_paused/reference(-1)，新actor尚未现场部署。模型继续驻留，显存18908MiB/free5306MiB，GPU idle；未down避免重复加载。

## 下一步

现场确认机械臂/相机与专家一致后，在 `/media/agilex/Getea1/jiaan/projects/cobot-platform` 执行 `./scripts/rlt_up.sh --frozen-actor --no-record` 做固定warmup测试（页面期望actor_version500）；希望采在线数据与每5条有效轮次更新时用 `./scripts/rlt_up.sh`。两者都需现场arm/start，agent未执行。详细过程见 [现场SOP](../PLUG_V2_ROLLOUT_SOP.md)。

## 首次现场反馈与动作影响复查（12:20 UTC以后）

用户反馈固定warmup与未warmup无明显差别。只读查询确认Session `1f3f3ef6-b258-4cc8-b840-b77d722040a3` 的 actor_mode=warmup、actor_version500、learner_version20000、recording_enabled=false；该次Session已stopped/paused、无fault。53个chunk日志显示actor500。三类留出样本通过实际8021 actor service无动作请求与保存模型逐值比对，差异不超过6e-8；排除该次旧actor未加载，但不能用版本号声称性能改善。

在全部固定留出缓存上量化（非刚才实时视觉状态）：失败组33行，原始actor与reference平均绝对差0.002731rad；经30Hz真实执行条件化后仅0.0003443rad（约0.0197度），92.14%有效关节动作差低于0.001rad。成功episode组612行（含policy/HIL）平均条件化改变量0.003577rad，专家439行0.003017rad。失败状态改变量被执行条件化大幅压低，是无明显纠偏的一项证据，不据此换算末端厘米或证明单一根因。

同状态min(twin Q)比较：成功episode组录制动作相对条件化reference获得更高Q仅44.28%，专家54.67%；条件化录制动作分别36.76%/46.92%。此为离线动作偏好诊断，不是成功分类准确率，表明尚无可靠的动作优劣辨识证据。失败reference采集动作与相同上下文条件化reference的Q一致，符合真实baseline动作契约。

当前目标对失败/非监督动作仍施加BC权重10的reference_anchor，Q权重0.1；候选选优是heldout BC+10*smooth，不能代表任务成功。下一阶段应针对失败状态纠偏、可执行动作下Q偏好及参考约束/选优目标做小规模离线消融，保留物理限速；先不盲目增加训练步数/在线rollout或通过explore放大噪声。未改训练参数/执行控制，未启动新训练或真机动作。完整诊断 `/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/warmup-effect-audit-20260918.json`。
