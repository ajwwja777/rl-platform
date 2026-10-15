# EXPO 5 回合批量更新与双卡 Learner

2026-10-15，负责人 /root。用户批准减少逐回合训练/回传，同时固定当前模型测试成功率。实现位于 feature/expo-sync-alignment-20261014，A6000 主仓库及 Cobot 安装状态以末尾验收记录为准。

## 当前状态

实现已提交，184 CPU、102 前端回归通过；真实双卡完成20 Critic/1 VLA和checkpoint/Actor导出。批量3/5退出→新进程补齐5/5更新→双卡恢复通过；已有Online11恢复与反向双卡→单卡恢复、现场静态安装仍进行中，尚未切换现场模型。

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
