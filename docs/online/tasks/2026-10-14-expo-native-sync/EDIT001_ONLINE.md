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

## 2026-10-14 EXPO无HIL回合结果标注修复

用户失败标注报episode_not_found、左侧0条记录。实际UUID7e4bf1d5-cf33-41d4-bca4-0ebb4c62b4e6有188条EXPO独立采样，但分段录制按HIL门控只有首尾6张图、0训练帧；sidecar已提交，HDF5保持.incomplete。旧label_outcome只查完整HDF5，异常阻断EXPO finalized，同时前端未刷新已保存历史。

Dagger8f5e269对经过身份、complete/committed、0帧、无HIL区间/teach_mask验证的节点回合原子保存结果到sidecar，保持HDF5有效性规则；Web29c9c1e在已提交后标注/收尾失败时刷新历史，不自动归位。39CPU和90前端通过，含真实SegmentedCaptureService零HIL失败标注→HTTP→Online收尾顺序、身份/非零/未完成/HIL拒绝和原子落盘失败保持。

按用户明确failure恢复当前sidecar，再通过现存Runtime8044/episode/end完成原回合收尾。188条样本及原HDF5残件SHA保持；远端正式completed1、online188、Replay3489=seed3301+188，warmup_remaining9、Critic/VLA0、Actor0=served0。先前另有aborted回合，所以accepted2不等于有效warmup2。恢复没有伪造完整HDF5、补拍视频或重跑运动。

4源码文件在提交push后按SHA安装；仅网页35055→411400重载激活，Runtime378049/Actor379952/Learner1070338保持。正式历史HTTP返回1条failure，6张首尾JPEG可读，next_episode_index2；当前录制规则不保存无HIL时的连续三路录像，188条RL观察不能冒充原始完整三相机视频。Session因Web重载恢复为inactive，需用户重新开始。

重载后头部相机无帧，独立4秒订阅f0/l120/r119；单路toggle_uvc_camera请求超时。用户同时报告自行重启节点；本对话此前已发出的正式相机stop/start也完成（420957→421799），最终三路ready、health ok。无法把恢复唯一归因于任一操作；无机械臂/ROS master/模型重启或运动，无残留自动恢复任务。

证据主存：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-native-remote-20261014/verification/edit001-online/node-outcome/，含tests.log、frontend-tests.log、source/install/recovery/acceptance/reload-after/camera-restart回执。现场同目录runtime/verification保存原sidecar备份，失败标签及Online采样正式保留。


## 2026-10-14 EXPO Online 连续回放与下一回合修复

后续截图的两个failure实际已接收326条在线采样，Replay3627=seed3301+326；0 frames是HIL训练帧为0，不是Replay为空。旧回合只有首尾图片，无法补回原始完整三相机视频。旧preview/labels仍查询完整HDF5，分别产生episode_not_found和recording_not_complete_or_labels_invalid。

Dagger f4bcde4增加EXPO Online独立review-only录像，3路320×240合成960×240、15fps、CPU libx264单线程；保存到对应.segments/UUID/review.mp4及review-video.json，不打开HIL门控、不写Replay。人工接管前后持续录像；时间戳用于独立回放节点映射，训练frame_index/mask不变。缺失相机用标识画面并计数，编码失败返回error，不冒充成功视频。

Web 7438333仅对expo_online启用该录像，回放/HIL分别计数；旧零HIL回合明确返回unavailable并显示仅首尾图片，结果只读显示，不再调用不存在的HDF5标签。目录准备与历史刷新异常隔离，结束后的finally保证解除busy。未抓到用户浏览器卡住时的堆栈，不能断言其唯一触发分支；已用故障注入覆盖这两个可导致下一轮锁住的异常路径。

44后端/96前端通过：真实录制器和模拟相机连续两轮start→failure→history→next，编号递增到3；视频逐帧解码、HIL切入退出门控不变、旧回放无错误轮询、真实按钮渲染解锁。Cobot只读真实三相机另外录制3秒/46帧，全部解码，missing_camera_frames0；视频首帧目视确认三路画面，未进入真实Replay、未发机器人动作。

7运行文件提交push后按旧/新SHA与模型操作锁安装；仅重载Web，保持相机订阅至新Web健康，再退出订阅。Runtime378049/Actor379952启动时间不变，Learner1070338仍在LFT2 GPU1；模型paused、Online ready、completed2/online326/warmup_remaining8、Critic/VLA0、版本0=served0，目录下一条3。两个历史preview均HTTP200 unavailable，失败结果保留，三路新鲜、ROS readiness ok。Web重载后Session inactive，用户刷新页面重新开始Session后采集。新版本的真实运动rollout/HIL视频验收仍由用户执行。

证据：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-native-remote-20261014/verification/edit001-online/review-video/，含测试日志、source/manifest、安装/切换/HTTP回执及只读相机MP4；跨机文件逐项SHA核验见receipts.json。
