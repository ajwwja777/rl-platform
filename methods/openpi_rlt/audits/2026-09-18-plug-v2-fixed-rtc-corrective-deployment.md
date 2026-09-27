# 2026-09-18 固定 RTC 纠偏部署前离线验收

用户要求先离线验证，再交付最佳部署方法与文件。推荐独立固定 `rtc-corrective-r1` 小范围现场验收，而非继续旧20k在线Q优化。没有执行机械臂动作、HPC访问或真实Session启动。

## 机制与数据

此前新critic的episode排序改善，但factual HIL方向偏好仅52.5%，失败state减速Q偏高；episode AUC1不证明Q指导纠偏有效。追加inverse BC、完整reference输入、command loss、constant/affine temporal、past-command监督及三折集合对照，负结果均留在Cobot runs/plug_v2/learning对应20260918目录/operation.json，不发布这些候选。

过去两条已发送SDK命令（仅t-2,t-1）与当前state生成合法hold/linear prefix6，未来teacher只作BC target。137文件1364行，支持范围内train专家797/HIL378，val85/104；原UUID train74专家/44HIL、val8/11，fragment与三角色（teacher/factual/autonomous-boundary）统一划分，零训练UUID泄漏。另42train/10val真实自主输入及随后HIL label作为反事实BC，不是实际执行的RL transition，不造reward/TD。

原train三折OOF选择2k/有限门控族，固定raw radius.2、brake_beta.75、gain .3–.7、prefix位移threshold.015/temp.005rad。该门控量是计划prefix位移，不是实测tracking lag。随后将门控纳入训练forward，与部署处理相同；inverse EMA/velocity projection仅派生BC目标，原SDK label不改。loss=.1 raw inverse MSE +1 conditioned feasible-target MSE +.01 command acceleration；速度不提高。BC batch128含56专家label/72HIL或自主输入后HIL label，不冒称作者human20%设置；此前critic隔离实验20.3125% HIL独立配额是另一阶段。

冻结Stage1/token encoder/decoder与contract step2000 critic，只微调actor。版本2000是候选标识与选定训练段步数，不等于总actor累计更新或Stage13999。d0专家全纠偏退化，故明确发布范围d6；d0首段/暂停/HIL fresh replan返回reference原值。此为执行相容BC纠偏，不是宣称标准RLT warmup/在线学习已经有效。

## 原UUID留出动作结果（episode macro，rad²）

|组|UUID/窗口|候选|reference|最强简单baseline|候选/baseline|
|---|---:|---:|---:|---:|---:|
|专家过去命令RTC|8/85|0.00082629|0.00099605|0.00083369|0.9911|
|HIL过去命令RTC|11/104|0.00046111|0.00109794|0.00049411|0.9332|
|真实自主输入BC反事实|10/10|0.00011073|0.00027619|0.00011383|0.9728|

相对reference MSE改善约17%/58%/60%，不是成功率。相对最强简单基线优势约0.9%/6.7%/2.7%，小样本结论需现场验证。外层val参与多轮机制诊断，虽无UUID训练泄漏，不能当完全独立最终统计确认。

Q_actor_mean对15条最初3自主d6窗口AUC1（11HIL辅助成功/4失败）；跨UUID条件RL token置换AUC约.555，reference/context仍含视觉，不能视为完整视觉因果验证。无自主成功类，不能推出自主成功分类可靠；同state动作排序不足，所以此候选不使用critic在线优化actor。失败有效heldout无done终局，终局校准证据支持也有限。

## 实际执行合同验收

CPU服务明确输出legacy .05 core，固定wrapper重建训练的.2 residual/gain/brake，fingerprint/version必须吻合。200窗口（199d6+1d0）实际HTTP与CorrectiveActor直接forward误差<=2.38e-7rad；真实CommandFilter逐值匹配训练filter，prefix完全不改，左臂/双夹爪固定实测state。30Hz/.2rad/s/.006667rad step/tracking .04rad保持；CPU actor P95约2ms。

真实Stage1 RPC+真实Runtime/HTTP+FakeIO（替换ROS模块，无publisher）7项通过：暂停立即停执行/清队列；暂停HIL进入及退出保持暂停；学习锁拒绝resume；明确fixed模型选中；异步RTC；success/failure/abort收尾无home；Session结束服务idle。另旧generation结果拒绝、tracking冲突fail-closed通过。完整chunk约96–114ms，六帧200ms预算内。

生命周期captured-command测试确认复用既有CPU core、Session corrective、不启动online cycle、exploration off。shell语法与控制解释器manifest预检通过，当前合同Q评估器也实际执行（正式结果formal-q-evaluation-final）。固定制品只含actor/诊断critic/provenance，不保留不匹配optimizer或target恢复状态，禁止用作resumable learner。

## 部署（由现场人员执行）

按原流程配置CAN/启动机械臂/三相机，维持专家左相机视角及已持插头/近插孔的插入初态。

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_up.sh --profile plug_v2
./scripts/rlt_demo.sh --restart
```

VS Code仅转发8015，打开http://127.0.0.1:8015/，选RLT rollout，等待ready_disarmed且ROS/camera ready，现场arm→开始Session。固定eval/无探索/无在线cycle；预加载模型复用。默认在网页所选online目录保留新rollout但不自动训练；只拍不留图像加 --no-record，无动作检查加 --shadow。此入口拒绝 --explore。

终端 `[fixed-candidate]` 应显示name rtc-corrective-r1及精确sha/version2000；仅2000数字不足以证明身份。先同场景5次独立自主对照reference，HIL成功另计，不充当自主成功。暂停/HIL release保持暂停；正常终局沿公共流程all --pose plug归位，现场确认安全与复位后才下一轮。

Ctrl+C或网页结束Session仅收尾/保留模型，另一终端 ./scripts/rlt_stop.sh 结束遗留Session；释放显存再 ./scripts/rlt_down.sh。不要重跑warmup或rlt_select覆盖原actor。reference对照仍 ./scripts/rlt_up.sh --reference。

## 文件/证据/当前动态

执行根 `/media/agilex/Getea1/jiaan/projects/rlt`：

- `runs/plug_v2/learning/candidates/rtc-corrective-r1/{actor.pt,manifest.json}` 独立固定制品，必须通过wrapper部署。
- `methods/openpi_rlt/plug_v2/{fixed_candidate.py,corrective_policy.py}` 合同实现，runtime/cli仅增加corrective分支，原reference/warmup/latest保持。
- `runs/plug_v2/learning/rtc-corrective-consistent-training-20260918/{train_inverse.py,results.json,deployment-gate.json}` 训练与留出。
- `runs/plug_v2/learning/rtc-corrective-cross-validation-20260918` 原train OOF选择与负结果。
- 候选目录 `{validate_offline.py,offline-validation.json,validate_session.py,test_fixed_session.py,lifecycle-validation.json,session-offline/tests/session_e2e_validation.json,formal-q-evaluation-final/report.json}` 可复验及raw episode Q CSV。

最后只读真实Session stopped/policy_paused/actor20000/fault null；8015未监听，需用户ui_up，不能称网页ready。model8020预加载保留，诊断锁空闲，11221/11226测试服务已关闭，正式replay0。未访问旧trainer/HPC根，provenance路径只是历史。完整轻量证据与执行SHA见 `../plug_v2/fixed_rtc_corrective_delivery.json`。

最终候选SHA256 `a197ab49db46bf7089740eb868400df57861051c34eeeaee625394d713a44918`；原production SHA256 `c6395e9d44139326625cdc2c894fe56b5d8c0196fcb4d722267a0eb24715a9b3`。
