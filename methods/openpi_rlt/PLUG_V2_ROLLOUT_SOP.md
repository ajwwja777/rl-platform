> 当前禁止把 supported-online-delivery 当作有效真机交付：现场失败，候选已拒绝。正在按原仓库重建诊断基线；尚无新的验收命令。

# 当前入口：2026-09-19 保守在线版本

以 [本次交付](audits/2026-09-19-supported-online-delivery.md) 为准。先 rlt_up.sh --frozen-actor 验收，再结束Session后 rlt_up.sh 在线学习。初始actor500，r7；旧corrective保持封锁。此版是IQL式保守更新变体，非原样RLT；尚未验证现场成功率/在线趋势。

# plug_v2 三视角 RLT 现场流程（2026-09-18）

唯一现场入口 /media/agilex/Getea1/jiaan/projects/cobot-platform/scripts，公开网页8015。以下动作由现场人员执行，agent交付仅无动作验证。

## 当前优先：固定 RTC 纠偏候选 r1

离线合同/输出/Session验收通过，仅具备固定模型小范围现场验收条件，未宣称成功率改善。关闭在线学习/探索，不覆盖旧warmup/latest；d0 reference/d6纠偏。

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_up.sh --profile plug_v2
./scripts/rlt_demo.sh --restart
```

8015→等ready/ROS camera ready→现场arm→开始Session。核验终端fixed-candidate name rtc-corrective-r1与sha，版本2000。默认录制但不自动训练，仅demo加 --no-record；拒绝 --explore。Ctrl+C/rlt_stop保留模型，rlt_down释放。[完整验收审计](audits/2026-09-18-plug-v2-fixed-rtc-corrective-deployment.md)。

## Reference rollout

先按原流程配置CAN、启动机械臂/三相机；仍需ROS节点，包装脚本只是自行source。新模型只覆盖已持插头、靠近插孔后的插入片段。保持专家采集时左相机视角，不用origin改变相机姿态。

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_up.sh
./scripts/rlt_up.sh --reference
./scripts/rlt_status.sh
```

不再传4000数字，manifest唯一选择 checkpoints/plug_v2/bf16/3999，对应4000训练更新。当前reference是joint Stage1 VLA分支、无RL warmup；没有额外训练独立pureSFT。预加载则复用，cold load仍需等待。

VS Code转发8015，打开 http://127.0.0.1:8015/，选择RLT rollout，等后端ready_disarmed及ROS/cameras ready，现场arm→开始Session。模型预加载不等于Session/硬件就绪。若需到专家准备位姿，由现场确认安全后执行 ./scripts/home.sh all --pose plug。

每轮自主rollout，必要时后臂按钮HIL；网页暂停后HIL也记录有效数据。release后保持暂停，几秒后点成功/失败可行，等待不训练；需要继续同轮才点继续，旧chunk丢弃、fresh replan。判定终局先暂停、释放示教按钮，再选成功/失败/放弃；固化后all --pose plug、保持暂停，现场复位物体后下一轮。EndSession暂停并以放弃收尾未完成本轮，不归位/不继续动作。

reference默认进 data/rlt/plug_v2/warmup；可在网页RLT录制目录选择其他data子目录，点击“检查/新建目录并使用”，未建自动创建，页面显示实际保存路径。仅本轮结束/非活动阶段可改，选择不用重载模型；warmup和online分别记忆路径，历史登记的同版本目录仍被对应训练读取。网页条数只计当前目录保留的成功/失败，放弃回减且下条连续；内部身份不复用，同页简洁历史列结果/HIL/actor。

2026-09-18错误中臂相机的32次尝试已按用户要求删除，删除当时warmup有效条数0，82专家保留；随后已在确认视角后采集55 HIL成功/20自主失败并完成首轮warmup。拍不参与训练demo用 --reference --no-record：保留小型终局/HIL统计，无图像/正式replay；shadow亦无动作/训练/归位。

默认eval，VLA噪声固定seed42、actor不加探索。显式 --explore 才加右臂chunk级std.001rad扰动；条件化/限速不变，不能修改锁定prefix。

## 手动 warmup

新相机尚无warmup actor，不能用旧7160等代替。2026-09-18已采55 HIL成功/20自主失败，75条转换完成，warmup replay v2修复、真实特征与正式训练验收已通过。20000 critic/10000 actor已完成，固定留出选优发布actor_version500；完整15分31秒，优化2分55秒。[正式训练审计](audits/2026-09-18-plug-v2-formal-warmup.md)。无需重复运行warmup，先固定actor现场验收。入口至少3有效成功+3失败，评测HIL与自主分开；不会达到transition阈值自动训练。

结束Session，保持模型内存，**不要先rlt_down**：

```bash
./scripts/warmup.sh
```

冻结Pi05及token encoder/decoder，只训actor/twin critic。专家30%/新成功40%/失败30%，原UUID固定留出（新rollout初始20%，专家原74/8 UUID）；暂停、坏帧、跨generation不拼接。policy直接复用真实accepted token/ref/已承诺prefix，专家/右HIL用cold d0，禁止未来教师prefix。teacher stride2、policy按真实十步decision；截断不bootstrap、不伪造失败或奖励。HIL成功自主片段不用正向BC，失败仍用于Q/reference约束；未执行padding不入BC/Q/平滑。使用独立runs/plug_v2/replay/v2，旧缓存不训练。[修复证据](audits/2026-09-18-plug-v2-warmup-replay-repair.md)。

最多20000critic/10000actor更新，batch128/LR1e-4，按固定留出BC/QTD/平滑及物理限速选优，未必用20k。4090正式实测优化174.79s/20k，完整特征准备+优化931.24s；缓存绑定Stage1。学习时start/resume/next锁定，候选被拒则不部署。

看 projects/rlt/runs/plug_v2/learning/warmup/{ready.json,actor.pt} 与 learning/operation.json、命令stdout。actor_version/global_step是critic累计更新编号，actor更新次数约一半，不能混同Stage1的3999。

成功/失败平铺HDF只暂存，后台官方LeRobot转换/全视频/数值/完整事实/SHA验证后才删；留下phase/lerobot/<uuid>、rlt/labels/source_facts/training_mask/summary。放弃归档小事实后删图像，不训练。网页专家条数82保留，不靠HDF存在计数。

## 固定 warmup 与在线更新

结束当前Session后切模式：

```bash
./scripts/rlt_up.sh --frozen-actor
# 只评估，不录正式数据：
./scripts/rlt_up.sh --frozen-actor --no-record
# 最新在线：
./scripts/rlt_up.sh
# 明确选择受控探索：
./scripts/rlt_up.sh --explore
```

新warmup存在后默认latest，有accepted online则加载它，否则从新warmup开始。每episode actor固定；success/failure/HIL存online，转换完成后每5条新有效episode，轮间/暂停一次训练：UTD5×实际新增train replay transitions，单批最多20k。holdout/历史不计新增预算，训练仍混专家/历史成功失败。通过才发布，下一轮用新actor；拒绝留旧版，数据仍在replay。固定warmup/no-record不在线更新，不再每轮固定32步。

核验 learning/online_cycle.json（UUID/updates/去重）、operation.json及online/ready.json（接受及留出指标）与Session actor_version（实际部署），不要用主观感觉当更新证据。更新不保证成功率单调上升，用相同固定场景独立自主测试reference/frozen/latest，HIL成功分开。

## 退出与关机

rlt_up现在保持终端前台，不立即返回提示符。Ctrl+C先暂停策略、放弃并收尾当前未完成轮，不归位；等待写盘结束后终端退出，模型继续驻留。网页结束Session同样使监控终端退出。正常写盘队列最多等待90秒，界面收尾期间不要开始新轮；停止动作先于写盘等待。
如果网页关了，在运行rlt_up的终端按Ctrl+C即可。如果终端也关了，从新终端执行：
~~~bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/rlt_stop.sh
./scripts/rlt_up.sh --reference
# 或一条命令：结束旧Session，再准备新的（仍需现场arm/start）
./scripts/rlt_up.sh --reference --restart
~~~
rlt_stop只结束Session并恢复已结束的失败recorder占用，保留失败数据以供排查；下一次准备目录会将已关闭且明确error/身份匹配的incomplete隔离到本目录/.failed/<uuid>，不计条数、不训练，文件编号保留不重复。未知/正在写入的文件仍会阻断。重复执行可行。重开rlt_up复用已加载模型。退出信号可捕获会做同样处理，但断网/强杀不保证能通知远端；现场危险用物理急停，不依赖HTTP/终端。
全部结束并释放预加载模型/显存才用：

```bash
./scripts/rlt_down.sh
./scripts/ui_down.sh
```

先RLT再UI。down等注册转换、收尾并停自己登记的Session/cycle/actor/model、释放显存；若转换仍未完成明确拒绝快速关闭，不强杀数据。CAN/机械臂/相机launch独立，由现场按原流程结束并支撑后臂/断电，这两个down不是机器人断电或系统关机。

当前offline-validated，真机效果及新warmup/在线提升待现场。[训练交付证据](audits/2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery.md)。

## 2026-09-18 固定候选对照

候选100/500/2000/5000/10000/20000已保留。[离线对比](audits/2026-09-18-plug-v2-checkpoint-comparison.md)显示20k并未增强失败纠偏；真机对照前结束Session，执行 `./scripts/rlt_select.sh 20000` 再 `./scripts/rlt_up.sh --frozen-actor --no-record --restart`，网页核验actor_version20000。返回500同样先结束Session再select500。选择会备份旧warmup完整制品，不重载Pi05；不切换活动episode，不覆盖online actor。
