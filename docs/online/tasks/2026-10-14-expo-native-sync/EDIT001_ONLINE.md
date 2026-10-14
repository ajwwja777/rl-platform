# EXPO edit0.01 Online · 2026-10-14

状态：60项CPU回归、原生完整更新/新进程恢复、更新后Actor本地推理、正式Web暂停加载均通过；0.01 Online已交付，真实rollout/HIL与学习收益待用户验收。

用户确认0.01预览真机几乎不抖、0.05仍抖，要求使用0.01进行Online并先占用LFT2空闲卡。GPU0外部Isaac Sim保持，LFT1及原0.1运行资产不改。

## 独立配方与身份

新入口：EXPO / realtime-expo-ft-edit001-online / 2000；model_id=expo-plug-left-2cam-edit001-online-2000。Cobot本地Actor，LFT2 GPU1 Learner。edit_scale0.01在构建原生Agent之前写入配置，因此动作编辑、Critic目标编辑、Edit Actor损失使用一致尺度。其余25成功TRAIN示教、前10有效在线回合只收集、batch64/UTD20/interval30、RTC5、HIL/50Hz执行保持。原三个模型及0.01/0.05冻结预览保留。

Actor快照身份新增非默认edit_scale；本地/远端状态互查尺度，错误尺度快照拒绝安装。Learner契约包含尺度和adapter源指纹，修改参数不能复用现有目录。原0.1 Actor身份格式保持兼容；原LFT1代码副本、配置和Replay没有修改。

LFT2固定SSH命令只新增配置中明确白名单的`run edit001`；原`run`仍走LFT1。沿用现有专用key及loopback18443/18453端口，不改authorized_keys、不接受客户端路径或任意命令。

- LFT2 Learner配置：outputs/expo-native-remote-20261014/profile-edit001.json；独立正式状态/Replay/checkpoint：同目录online-edit001。
- Cobot配置：runtime/expo-remote/profile-edit001.json；本地journal/Actor缓存：/home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_remote_edit001_sft2000_success25_20261014。
- 录制：/media/agilex/Getea1/jiaan/data/datasets/plug_left_2cam/recordings/expo-edit001-online-sft2000。
- 隔离验证：outputs/expo-native-remote-20261014/verification/edit001-online；复用既有assets，不在LFT2散放文件。原预览回合不自动进入新Replay。

## 当前验证与修复

源码a5c1899实现一致尺度和命名SSH配置；2394339修复并发状态写入。首次隔离脚本启用learner_only却未附带模拟Actor RNG回执，被准入检查正确拒绝，尚未训练；同时测试与后台收尾并发写status.json共用.tmp引发冲突，已改UUID独立临时文件并添加强制并发回归。失败日志保留，不计为通过。

60项CPU回归通过；第二次隔离native-lifecycle-r2验证unknown/aborted不入训练、前10回合只收集、第11回合30transitions完成20Critic/1VLA/edit/temp更新，Actor0→1。独立新进程恢复保留completed11/Critic20/VLA1/version1和edit0.01，全部退出码0。合成数据只在verification，不进入正式Replay。占位进程722388按用户要求预分配约36GB；测试结束后恢复占位，正式启动前仅结束本任务占位，由生产Learner接管GPU1，其他GPU任务不动。

证据：verification/edit001-online/native-test-result-r2.json、native-lifecycle-r2/lifecycle.json与restore-only.json；更新后Actor和正式暂停交付见以下回执。

## 更新后Actor与正式交付

更新后Actor1完整约2.018GB传到Cobot，manifest身份与每叶SHA验证通过；actor51/51、encoder106/106、edit14/14、critic18/18叶均相对初始改变。Cobot独立无ROS推理40次：precache最大145.66ms（500ms预算），select最大10.33ms（50ms预算），有限值/shape与edit0.01通过。传输约402秒、约5MB/s；同步学习后的完整权重回传可能等待数分钟，仍保持更新/发布期间下一回合关闭。记录actor-transfer.json、updated-actor.json；测试权重没有装进正式Actor目录。

生产源码a5c1899/2394339已feature/main提交push，再按旧/新SHA与操作锁安装11源码/注册文件及专用profile；原模型条目不变。现场先确认0.05预览manual_pause=true、capture idle、无active/operation，正常结束Session并卸载，然后完成隔离Actor验证和正式加载；未启动机器人Episode、录制、回零或重启硬件/网页。

正式回执时间2026-10-14 22:48（北京时间）：Cobot Runtime378049、Actor379952；LFT2 GPU1 Learner1070338。Web paused、model_ready/online.ready=true；remote actor_version0=served0，edit_scale0.01，seed25/3301、online0、Critic0/VLA0、warmup_remaining10。50Hz、RTC和滤波开启；Session开始/结束检查通过，最终Session inactive，gate manual_pause=true、published0、runtime_fault=null。SSH Range206/SHA、本地Actor仅推理与远端训练身份验证通过。正式接入记录production-acceptance.json；不是实际机器人闭环/收益验收。

production-storage.json确认独立录制目录为空、next_episode_index1；health ok、三相机ready。GPU1由正式Learner持有，临时占位与隔离Actor均已退出；无自动等待/切模型任务。用户拥有现场Start/HIL控制。真实采集按成功/失败结束后入库，unknown/abort排除；前10个有效回合仅收集，之后按原生30transition信用累计触发每次20Critic/1VLA更新。

原0.1/LFT1 Online和冻结0.01/0.05预览保留为独立入口；本次新目录不自动混入预览回合或原Online历史。参数0.01是明确的可选任务变体，不声称等于上游示例0.1默认。唯一下一步为用户在新入口进行真实rollout/HIL，核对首批有效回合、更新与发布，并评估平顺性和自主成功率。
