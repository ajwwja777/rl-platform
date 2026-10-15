# EXPO 5 回合批量更新与双卡 Learner

2026-10-15，负责人 /root。用户批准减少逐回合训练/回传，同时固定当前模型测试成功率。实现位于 feature/expo-sync-alignment-20261014，A6000 主仓库及 Cobot 安装状态以末尾验收记录为准。

## 当前状态

已安装为网页可选入口；184 CPU、102前端回归、真实双卡20 Critic/1 VLA、批量跨进程恢复及两方向跨GPU数恢复通过。已有Online11在双卡恢复后189个导出Actor叶身份保持，Critic220/VLA11、Replay5225、本批0/5。通过Cobot实际受限SSH启动GPU1,0的Learner并验证ready，再正常关闭连接；0新增更新、0机器人命令。当前现场仍为原RTC-SFT2000/Runtime1172181，未加载新模式、未重启Web或硬件。首轮真实batch5 rollout/HIL及收益由用户测试结束后验收。

## 调度与保留项

沿用提供仓库 `train_pi_robo.py` 的 `update_type=batch / num_batch=5 / num_updates=0 / step_interval=30` 语义。有效成功或失败回合逐轮保存、进入Replay，unknown/aborted不计入批次。每5个热身后的有效回合集中执行累计更新，并仅保存/发布一次Actor。全局batch64、每组20 Critic/1 VLA、成功池VLA采样、奖励、网络与损失保持原配方；不额外减少UTD，也不自动修改RTC/滤波/编辑幅度。

新运行仍先收集10个有效Online热身回合，首个可训练批次为第11–15回合。本次默认入口从已有Online11接续：12有效回合、1924在线transition、Replay5225（含25成功示教3301帧）、Critic220/VLA11、checkpoint2、credit19。已完成热身，下一批从0/5开始，不重做热身。非默认RTC/edit参数按既有规则使用独立训练状态。

本批计数、transition credit、Replay及随机状态在每回合原子提交；重启恢复未满批次。重复结果回执不重复入库/训练。每批仍同步等待训练、checkpoint、传输和本地安装后才开放下一回合；本版本不是后台异步训练。

五回合各150个transition时约25组更新=500 Critic/25 VLA，余数跨批保留。因此批量化减少暂停/保存/传输次数，不代表总梯度计算减少。相同Actor版本只检查manifest身份，跳过已装载权重的整包重复哈希/下载。

## 可选入口和状态

模型ID `expo-plug-left-2cam-edit001-batch5-2000`，方法 `realtime-expo-ft-edit001-batch5`，默认edit0.01/RTC5；原逐回合入口保留。Cobot执行推理/相机/控制/HIL，LFT2仅Learner。选卡优先空闲GPU1、0：两张空闲用两张，一张空闲用单张，均占用则拒绝，不停止其他任务。双卡是原生全局batch64的数据并行，不是把batch改128。

网页显示本批x/5、待执行更新组，以及原有Critic/VLA/Replay/本地Actor版本和实际GPU。当前成功率测试加载的是RTC-SFT2000、Runtime1172181、training_enabled=false；本任务不重载或替换它，不将其成功率归因于Online11。

- A6000接续目录：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-native-remote-20261014/online-edit001-batch5`。
- Cobot运行目录：`/home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_remote_edit001_batch5_20261015`。
- 新录制目录：`/media/agilex/Getea1/jiaan/data/datasets/plug_left_2cam/recordings/expo-edit001-batch5`。
- 回退：原逐回合模型及其独立目录保留；两目录后续训练彼此独立，不自动合并新Replay。

## 实现入口与证据

`sync_schedule.py`负责批量信用；`online_learner.py`负责持久化、更新与发布；`remote_launcher.py`负责选卡/参数；`remote_runtime_support.py`验证批次身份、传输及本地发布；`device_layout.py`、`memory_update.py`、`stream_checkpoint.py`支持双卡donation/CPU EMA与跨布局恢复。`fork_expo_batch_run.py`只从已停止且完整提交的状态生成独立接续目录，检查旧adapter和checkpoint SHA，保留源目录；仅不可变资产硬链接，元数据独立复制。

首次真实双卡暴露CPU EMA闭包持有已donate的actor_tau；已改为在更新前复制独立host标量，专门回归覆盖donation后首次编译。修复后的20/1调用177.51秒，之前单卡调用284.73秒，但编译缓存条件不同，不能当作正式加速比。双卡和单卡不承诺逐位相同的训练轨迹；检查输入身份、数值有限性、计数和完整状态恢复。

独立验证总目录：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-native-remote-20261014/verification/edit001-online/`，本任务用`dual-gpu/`和`batch5/`。合成回合只在verification，未进入真实Replay。真实机器人新模式rollout/HIL及成功率仍由操作员验收。

参考：[原同步代码](https://github.com/pd-perry/expo-ft/blob/main/train_pi_robo.py)、[异步代码](https://github.com/pd-perry/expo-ft/blob/main/train_pi_robo_async.py)、[Real-Time EXPO](https://arxiv.org/html/2609.18207)、[EXPO](https://arxiv.org/html/2605.25477)。本次对齐依据是现场提供的native-source快照，未直接用远端main替换代码。

## 输入与数值核对

单/双卡初始33叶Critic batch、33叶VLA batch的shape/dtype/SHA完全一致，682个agent叶中所有网络、优化器、RNG数组一致；6处Python标量在device_put后显示为int32/float32（单卡host为int64/float64），JAX默认32位计算。更新后RNG一致，但训练输出不逐位相同：最后Critic loss差约0.00606、VLA loss差3.11e-6，最大报告指标差0.556为filter_critic_grad_norm；642叶抽样参数/优化器增量最大差0.0363。未将这一检查称为单/双卡数值等价通过，差异具体来源尚未定位；双卡沿用原生全局batch及损失/更新函数，需独立真机收益验收。原单卡可用性和状态可恢复性另行验证，不能用双卡资源数代替效果验收。

## 安装与交付回执

运行代码RL `b269415`（含e8e1cd8/979cb5f）、Web `5e7a9a5`先提交push，再在模型操作锁内逐项校验旧/新SHA安装11个文件。保留并合入其他对话的Online参数控件和RLT训练文档。新Cobot profile、A6000受限launcher命名profile均已登记；原逐回合run仅做已审计adapter身份迁移，checkpoint2、Replay、优化器/计数不变，原contract留备份。新batch5接续目录独立元数据，Cobot复用经压缩SHA校验的不可变Actor11文件。

真实验证：10有效热身、unknown/aborted排除；3/5提交后退出、重启补第4轮不训练、第5轮20/1并一次导出；再在双卡恢复保持1/20/1。独立双卡训练checkpoint反向单卡恢复后189个Actor原始SHA逐一相同。生产接续仅恢复不训练，189叶与原Online11快照相同。输入/更新差异详见上节，不将双卡数学轨迹称为逐位等价。

`field-install.json`记录安装前后Runtime1172181与started_at不变；`field-catalog-check.txt`确认新方法available/train=true；`restricted-check.json`确认Cobot→LFT2正式受限SSH、参数d5/edit0.01/batch5、实际GPU1,0和状态11/220/11/5225。关闭自己的SSH输入后launcher终止其空闲Learner（SSH返回241对应子进程SIGTERM，为本次主动卸载），不重启当前SFT。前端HTTP实际返回含批次计数的新JS。安装备份位于Cobot `rl-platform/runtime/verification/batch5-20261015/before/`。

用户完成当前固定SFT成功率测试后，释放当前模型、刷新网页，选择EXPO / realtime-expo-ft-edit001-batch5 / 2000并加载、开始Session。沿用成功/失败结束及HIL；未满5轮结束Session会保留批次，下次继续，满5轮仍需等待同步训练/权重安装完成。最后成功率与更新速度需真实批次衡量；本交付不声称学习收益已经通过。


## 2026-10-15 Web readiness regression fix

The first real batch5 Web load exposed a validation-hook bug: Runtime.start_server polls the local inference-only Actor, but RemoteSupport.validate_server_status also enforced the remote Learner episodes_per_update=5 contract. Actor status intentionally has no training batch field, so the default 1 caused a false mismatch. Earlier restricted SSH readiness verification covered the Learner only and missed this local startup path.

Separate local Actor readiness validation from remote Learner validation. Both retain model/RTC/edit identity checks and now check their explicit training roles. Only remote startup and polling enforce the Learner batch contract. No checkpoint, Replay, sampling, or training contract changes. Regression coverage executes OnlineRuntime.start_server through the real inherited Runtime hook for batch1/batch5, and rejects wrong remote batch on both startup and polling. Full EXPO CPU suite: 188 passed (one existing JAX deprecation warning). Actual paused Web load verification follows installation; no robot episode is started by the verifier.

2026-10-15 actual Web load passed after SHA256-verified installation (fix05299bd, integration ef025d3). Failed runtime released and same batch5 entry loaded through Web API. Runtime1252867 paused/model_ready=true, session_active=false; local Actor training_enabled=false/version11. Remote Learner2288698 ready on LFT2 GPU1,0 (2 devices), episodes_per_update=5, collected0/remaining5. Critic220/VLA11/Replay5225 unchanged. No episode, inference resume, or robot motion requested. 188 CPU tests passed. Evidence: outputs/expo-native-remote-20261014/verification/edit001-online/batch5/readiness-fix-field.json and readiness-fix-tests.log; Cobot runtime/verification/batch5-20261015/readiness-fix-install.json. User can start Session on the loaded model; real rollout/HIL quality acceptance remains user-operated.
