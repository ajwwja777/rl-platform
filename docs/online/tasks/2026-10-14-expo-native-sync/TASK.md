# EXPO 原仓库同步 Online 对齐 · 2026-10-14

最新：用户报告Online抖动而SFT平滑；已完成配对离线诊断并交付独立edit0.01本地预览，真机待验收，原Online仍0.1且当前加载受LFT1GPU占用阻塞。见[JITTER](JITTER.md)。

历史交付状态：正式Web暂停加载交付完成。Cobot本地Actor/LFT1GPU0 Learner经LFT2中转均ready、版本0一致；Session启停与数据目录检查通过，发布0、无真实Episode。用户负责开始rollout/HIL及真机效果验收。

用户授权按新传入的 train_pi_robo.py / train_pi_robo_async.py 对齐并交付可用于 Web rollout/HIL 的 Online 版本；采用同步 recipe。用户明确确认 realtime_expo 整批示教成功，允许使用 TRAIN 前25回合。

已导出独立 seed_success25_20261014_v2，25回合3301帧，源HDF5 SHA/UUID与 SFT TRAIN manifest核对。选中源ID：1,2,3,4,6,7,8,9,10,11,13,14,15,16,17,18,19,21,22,23,25,27,28,29,30；DEV5未用。6回合尾部一帧POLICY被SFT排除，本次沿用同一人工片段，成功确认与裁剪理由写入独立manifest；原始文件不改。第一次导出的部分目录没有manifest，不可用。

最初的单机内存候选（未交付，已停止）：原仓库示例N32/edit32/filter32/filter_n_edit1/edit_scale0.1、batch64、UTD20、step_interval30、offline_ratio0、25成功示教预填；前10个有效在线回合仅收集，第11个完成后开始训练。保留Cobot左臂7D/双腕相机/H16/r8/delay2/逻辑10Hz。批次完整损失、优化器次数及顺序不缩小；尝试逐小批搬运/激活重计算、原生索引与增强随机数整批生成、目标Actor EMA保持CPU。以上内存候选仍需GPU更新/恢复验证。

证据：scratch/rl-platform/expo-sync-alignment-20261014/progress.json、web-tests.log、a6000-cpu-contracts.log、a6000-native-sampling.log；现场 runtime/verification/expo-sync-alignment-20261014/{native64-platform,mapped64-platform,split64-platform,stream64-platform}。

已确认 native64 申请27.62GiB显存失败；mapped64申请13.56GiB失败；split64因主机MemAvailable保护退出。stream64最后可见进入更新，其最终状态在断联后未知。A6000网页接口12项、快捷键11例通过；实际原生采样函数拆分前后全部数组、terminal/truncation/success选择及RNG逐位一致；分块masked loss/gradient与调度门槛通过。Cobot CPU全合同第一次因合成黑图不满足原生亮度断言失败，修正后的cpu-contracts-v2.log现已取回，4组原生合同检查全部通过。

独立worktrees：scratch/rl-platform/expo-sync-alignment-20261014/{rl-platform,cobot-web}，分支feature/expo-sync-alignment-20261014。当前未合入main。候选登记独立id expo-plug-left-2cam-native-sync-2000，独立Online输出；现有旧Online入口未替换。

唯一下一步：完成本地 Actor / A6000 Learner 的恢复后二次更新、本地延迟检查和最终网页暂停加载；随后交接操作员真实 rollout/HIL 验收。

## 2026-10-14 下午恢复检查

- 旧stream64最后约177秒，GPU14GiB、MemAvailable18GiB，之后日志截断；journal上一启动止于00:08:26，当前启动12:48:42。没有证据将断联归因OOM或电源，旧测试未完成。
- 已将四次GPU实验、完整CPU合同和seed manifest复制到A6000 field-evidence；进度与保护服务身份在progress.json。
- 网页24项选定回归通过；新增native计数/中英文断言后14项DOM回归通过（与24项重叠）。后台12项已通过。首次从错误cwd启动一个旧前端测试失败，按package.json在app/backend运行后通过；不是代码错误被忽略。
- 已准备verify_expo_sync_lifecycle.py：25示教初始化、unknown/abort排除、10采集回合、第11回合1份更新预算、失败回合仍从success seed训练VLA、重复finish、学习时禁止新回合及新进程恢复。该GPU闭环尚未执行。
- -r2验证目录仅隔离脚本已准备；production源码/注册/模型未切换。现场ROS readiness同时报告handover_publisher_unavailable，未来机器人运行还需现场硬件就绪，本对话没有启动硬件。


## 2026-10-14 17:36：第二次主机异常后的只读检查

17:20 Pi05已自行退出、GPU空闲/录制idle后才启动-r2隔离测试。用户报告风扇先响后停、黑屏和SSH断开；17:32:30人工重启。17:34检查无GPU计算进程，CPU39°C/GPU42°C、可用RAM约29GiB；8015未监听，未启动Web/ROS/模型。

上次journal止于17:24:22，无正常关机尾段；已保存内核日志未找到OOM kill、NVIDIA Xid、thermal shutdown直接证据，不能排除未落盘事件。先前实时读取曾见Critic20/215.10秒，本次磁盘恢复仅到Critic19/210.74秒，监测最后219.44秒可用RAM15450MiB、GPU14473MiB/利用率100%；没有Actor完成、checkpoint或exit回执。单次完整更新未通过，不能归因已确定发生在VLA计算内部。

另见15:28:26外置Getea1对应sda/sda2读取I/O错误，比测试早近两小时，不作为本次根因。当前NVMe Composite55.9°C、Sensor2 84.8°C，只保留原始读数，未确认传感器含义或故障时温度。EFI pstore归档存在但无权限读取，sudo -n需密码，未提权读取。网络watchdog最后一次17:21:51正常结束，无其触发重启证据。

完整只读证据：scratch/rl-platform/expo-sync-alignment-20261014/field-evidence/reboot-1732/{diagnostics,additional,final-check,pstore,summary}.json。仅在A6000保存诊断和更新记录，Cobot生产源码、配置、Replay、权重未改；未重启服务、启动GPU计算或机器人。


## 2026-10-14 18:00前后：用户批准远端方案，验证进展与未通过项

- 用户回复“按你推荐的来，最后交付给我work结果的即可”，明确接受A6000推理/学习、Cobot Web/相机/机械臂/HIL。保持同步算法，新增远端可选入口，不替换旧Online。
- SFT包deployment.json记录training_max_prefix_steps=10；远端采用原示例支持的delay5，执行/Replay/训练目标一致，10Hz逻辑与可选50Hz发布保持。原delay2入口保持。A6000推理约245ms，原200ms预算不满足；两机ping约2–4ms，完整HTTP延迟待验收。
- A6000 stream64隔离更新完成Critic20/Actor1/edit/temperature、更新后推理和7.5GiB完整checkpoint；lifecycle5真实25seed/3301帧、unknown/abort不训练、前10回合不更新、第11回合20/1更新和保存通过。均为合成在线回合，不是生产学习或真机验收。
- GPU batch4/UTD2数值对照失败：RNG/步数一致但参数明显不同。进一步同真实图像fixture的GPU增强对照，输入数组相同，left-wrist增强max差0.04565；此前32x32 CPU检查不足以认证GPU等价。当前正在A6000 GPU1测试直接原生整批更新，优先移除不必要的分批增强适配。
- 远端实现RL89b8ef5/Web7c66241已feature分支push，main/现场生产尚未安装。新增SSH固定命令、回合逐文件SHA/不可变传输、断线本地保留、提交回执与新回合门禁；29CPU/14DOM通过。Cobot专用私钥只在runtime/expo-remote，A6000仅授权固定验证启动器及localhost18443转发。未公开HTTP监听、未修改其他SSH授权。
- 两机HTTP验证正在独立runtime/verification/expo-remote-20261014与A6000 verification/a6000/remote-http中运行；不启动ROS/机器人/真实录制，所有在线回合合成。A6000 GPU0运行远端测试，GPU1对照；Cobot无本任务GPU负载。
- 证据位于同任务verification/a6000；首次前端运行缺node路径/jsdom，使用既有nvm Node22和主项目NODE_PATH后14项通过，未安装/修改共享依赖。

## 2026-10-14 最新架构修正与整批验证

用户指出推理应默认本地部署，仅 Learner 在 A6000。本对话已按此继续：本地 Actor 不持有优化器/Replay、不调用训练；远端 Learner 原生整批 batch64/UTD20。每轮结束上传带 SHA 与 rollout RNG 回执的真实执行/HIL journal，远端保存后发布 VLA 可训练权重、视觉编码器、编辑网络、目标 Q 的同版本快照；本地校验、安装、预热完成后才开放下一轮。网页显示远端更新版本及本地已加载版本。

A6000 native64-delay5 单次原生整批更新完成20Critic/1Actor、推理和checkpoint，退出0。两机 remote-http-native 完成10回合热身、第11回合20/1、重启恢复、第二次40/2；远端动作推理延迟未通过50ms保护，不能作为可用现场配置。Cobot 路由确认走 wlp2s0 校园Wi-Fi；推理选择计算约8–9ms，端到端出现50–200ms以上，未放宽保护或改动网卡配置。

最终 hybrid-http 正在验证本地Actor/远端Learner真实SSH链路、未知/放弃排除、热身、权重同步与恢复后再更新。CPU34与前端14项通过；全部为合成回合，不含真实录制/机器人动作。候选代码已feature push，主分支/生产尚未发布；文件实际同步须以最终清单为准。

本地 Actor 首次权重更新已验证：A600020Critic/1VLA 保存并传回快照，本地安装到版本1。51/51个 VLA 可训练叶、106/106个视觉编码叶、14/14个 edit 叶、18/18个目标Q叶的SHA均相对初始发生变化；传输压缩/原始字节双重哈希检查通过。该事实证明参数更新与本地加载，不证明任务成功率提高。正进行两端重启后继续更新。

最终源码候选：RL feature 0c14737、Web feature aa5d24a；CPU36项与DOM14项通过。原压缩浮点快照约1.878GB，仅比原始约2GB小约7%，最终发布使用同一gzip容器的stored模式，并在A6000逐叶搬到CPU后分块写出，减少压缩/逐切片设备调度；仍逐字节无损，不改精度。保留当前与前一完整Actor版本，更新预算持续推进时延长空闲超时，不按墙钟硬截断有效长回合学习。

正式资产已预备在 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-native-remote-20261014/assets/`，272文件/6,659,555,704字节重新SHA全部通过。尚未激活production profile或填入任何合成在线Replay；生产源码/网页尚未切换。

## 2026-10-14 本地 Actor / 远端 Learner 验证完成

原生整批 batch64/UTD20 两次累计40Critic/2VLA更新、完整保存与两端恢复通过。本地版本0→1→2；unknown/abort排除，25成功seed/3301帧、前10有效回合仅收集、后续按30 transition预算更新均通过。第二次更新后大文件传输曾触发旧1秒心跳超时；保存与pending journal完整，修复传输阶段超时/重连后恢复版本2，重复收尾未重复计数或训练。Range断点续传及校验40项CPU回归、14项DOM回归通过。

恢复后50次Cobot本地调用：slow最大161.44ms（delay5预算500ms），fast最大10.73ms（保护50ms）；全部通过。真实机器人运动、相机实时负载、长期成功率仍未验收。恢复报告：scratch/rl-platform/expo-sync-alignment-20261014/verification/a6000/hybrid-http/recovery-result.json。GPU验证使用冻结验证adapter；最终c27fcd4另加入传输恢复/Range与无损导出优化，生产暂停加载将另核对。

最终可选入口expo-plug-left-2cam-native-remote-2000 / realtime-expo-ft-native-remote。仅A6000更新，本地Actor无优化器/Replay训练。同步更新和约2GB权重回传期间下一回合禁用，Wi-Fi下等待可达数分钟；不隐式切异步、不减batch/UTD或精度。原19个模型保留。

## 正式安装回执与剩余步骤

RL728de33/Webaa5d24a已串行合入main/push；保留另一对话a32707f的Stage1文档和代码。21个现场源码文件、226个原生源码快照在操作锁下安装，旧SHA、新SHA、Python/bash语法通过；回退原件在Cobot runtime/verification/expo-remote-20261014/install-before。正式profile已安装，Web35055未重启，/api/site/options显示新入口available。model offline、Session inactive、录制idle、GPU空闲；未启动ROS或机器人。

自动审批拒绝包含安装及authorized_keys调整的组合命令，未返回具体理由；该组合未执行。随后独立源码安装成功。专用EXPO key仍指向已结束的测试启动器，尚未切正式配置；已经向用户明确请求仅修改该key固定命令并收紧permitopen为18443的授权，等待回复。不能将当前文件可用状态描述成正式Online服务已加载。

唯一下一步：获准后切换专用SSH key（不改其他key），用正式profile加载模型并保持暂停，验证seed25/3301、online0、Critic0、Actor0与本地version0、Session启停及published0，再释放现场负责人给用户。真实rollout/HIL仍由操作员验收。正式证据已保存outputs/expo-native-remote-20261014/verification/，没有合成Replay进入正式online目录。

## 2026-10-14 20:04 后：SSH 激活完成，按用户要求等待显卡

用户明确批准专用key切换，A6000固定启动命令已指向正式remote_launcher.py/profile.json，仅允许localhost18443转发；其他key逐行保持。回执outputs/expo-native-remote-20261014/verification/ssh-activation.json。自动审批阻塞已解除。

当前两张A6000均运行qizhou的Isaac Sim评测，每张约15GB；未停止或修改它们。用户明确选择“等评测自行结束”。正式Learner完整更新峰值约33GB，继续保留独占GPU启动保护。

已启动本任务等待器PID3328687（A6000），最多等待6小时：仅当某卡持续60秒没有计算进程、现场模型/Session/录制/GPU均空闲，才选该卡并启动ROS master、请求Web正式模型暂停加载，验证seed25/3301、online0、更新0、本地版本0、Session启停、HTTP Range与发布0。脚本不启动机械臂、相机、Episode、录制或运动；任何冲突/错误停止并留报告。等待器脚本与状态在outputs/expo-native-remote-20261014/verification/{wait_and_activate.py,accept_paused_field.py,activation-status.json,production-acceptance.json}，最后一项仅完成尝试后存在。

当前尚未正式加载，不能开始Online。继续任务先查activation-status.json与真实现场；ready_paused仍只表示软件暂停验收，真实rollout/HIL由操作员完成。不要重复启动等待器；若其他对话需要切模型，先明确取消本任务的等待器。

## 2026-10-14 20:43：LFT1 后端验证完成，Teacher5 保留

用户指定10.12.0.212（LFT1）通过LFT2访问，并要求不要在LFT2随意放文件。实际SSH账户为LFT-W02（显式LFT-W01尝试无权限，未更改任何认证配置）；LFT1 hostname=LFT-A6000-1。GPU0为RTX A600048GB且空闲，GPU1被其他任务使用；仅使用GPU0。先取消原LFT2等待器3328687，旧自动现场加载不再排队。

- LFT1独立目录：`/data/LFT-W02_data/jiaan/expo-online-lft1-20261014/`。assets保存SFT/common/seed/native快照；runtime/{venv,python}为独立环境，3个editable路径和Python软链接已调整；adapter为已发布源码；learner.json为GPU0正式配置；online是正式Learner状态；verification为隔离合成验证。未修改LFT1其他人的环境/模型/任务。
- LFT2只沿用原主项目、任务worktree及`outputs/expo-native-remote-20261014/`，没有新建散落目录或额外复制权重/环境。既有profile.json切为固定loopback SSH中转；原配置备份在该outputs的verification/lft2-profile-before-relay.json。Cobot专用key/目标端口仍是LFT2 localhost18443，LFT2转LFT1 localhost18443；不公开模型HTTP端口。
- 272资产/6,659,555,704字节SHA全部通过。JAX0.5.3/Flax0.10.2与GPU计算通过。初次资产校验SSH等待120秒超时，未作为成功；随后独立后台逐文件校验得到完整回执。第一次环境检测GPU计算已通过，但写回执时verification目录尚不存在；补建后保存事实，不以失败日志冒充全程成功。
- 原生full batch64/UTD20测试完成：25成功示教/3301帧，unknown/abort排除、10有效在线回合仅收集，第11回合20Critic/1VLA更新、完整checkpoint及更新后有限输出推理通过。GPU实测峰值32888MiB。合成测试数据仅在verification/native-lifecycle/run，未用于正式online。
- Cobot仅CPU SSH验证真实三机链路：Cobot→LFT2固定relay→LFT1正式Learner，ready/身份/seed25/3301/online0/更新0/version0检查通过，Actor文件SHA与Range206断点读取通过。验证结束通过stdin EOF退出本任务Learner及双层SSH，GPU0确认空闲、18443/18453关闭；磁盘status.json中的ready是历史回执，不代表服务仍运行。
- 源码576dd1e已feature/main push；9项relay/恢复测试通过。现场只同步未运行的EXPO remote_launcher.py、remote_runtime_support.py及本任务profile的learner_host标识，按旧/新SHA检查并加操作锁。没有加载/释放模型、重启Web/ROS/硬件、开始录制或机器人动作。
- 用户确认Teacher5仍在使用/修复，必须保留。本任务终检仍为Teacher5，Stage1 PID122499保留；runtime PID由另一对话变为165463，状态error，录制idle。未把旧runtime PID不变作为承诺，也未干预修复。EXPO完整Web暂停加载/Session/真机rollout尚未在此后端验收。

正式入口仍为`expo-plug-left-2cam-native-remote-2000` / `realtime-expo-ft-native-remote`。加载时启动LFT1 Learner，Cobot运行本地Actor；同步学习/权重回传等待期间不开放下一回合。唯一下一步：用户结束Teacher5并允许切换后，重新核对LFT1 GPU0和现场状态，执行已有暂停加载验收；不再询问已经批准的专用SSH改动，不重跑已通过的隔离学习实验。

证据（LFT2原outputs/verification）：lft1-native-result.json、lft1-final-verification.json、lft1-relay-result.json、lft1-field-preserved.json、relay-tests.log、relay-install.json、lft1-transfer.log。LFT1详细日志/完整checkpoint留本机verification，不把大文件重复复制到LFT2。LFT1该测试Learner已退出，现场Teacher5保留；Online软件整体最终交付仍受现场模型切换边界限制。

## 2026-10-14 20:59：正式 Web 暂停加载交付完成

用户明确说Teacher5已结束并授权切换。执行前现场Teacher5已由用户释放，model offline、Session inactive、录制idle、Cobot GPU空闲；LFT1 GPU0空闲。未再次释放旧模型，未停止其他评测任务。正式EXPO入口经Web加载：50Hz发布、RTC和滤波开启，逻辑10Hz/RTCdelay5；原已有ROS master复用，未重启硬件/相机/网页。

验收：Runtime228395、Cobot inference-only Actor232063、LFT1 Learner1216617；网页phase paused、online ready，远端actor_version0与本地served_actor_version0一致。25seed/3301帧、online0、Critic0/VLA0、warmup_remaining10；正式Replay未混入隔离测试数据。本地Actor报告training_enabled=false，远端为true。manual_pause=true、runtime_fault=null、published0。Session开始/结束两次状态转换通过，最终Session inactive；未启动Episode/录制/机器人动作。真实HIL与闭环效果仍由操作员验收。

正式链路Actor文件Range206与SHA再次通过；/healthz为ok、无stale_keys，三路相机ready。模型指定录制目录已通过storage/prepare，flat布局、下一回合index1、existing_indices为空：`/media/agilex/Getea1/jiaan/data/datasets/plug_left_2cam/recordings/expo-native-remote-sft2000`。准备目录未录制任何数据。

记录：outputs/expo-native-remote-20261014/verification/production-acceptance.json、production-storage.json、production-acceptance-output.log；现场同名回执在runtime/verification/expo-remote-20261014。正式Learner运行/Replay/checkpoint在LFT1 `/data/LFT-W02_data/jiaan/expo-online-lft1-20261014/online`，Cobot本地journal/Actor缓存沿用已登记native_remote_sft2000_success25_20261014目录。LFT2仅中转和原有项目记录。

交接给用户：网页当前已加载EXPO / realtime-expo-ft-native-remote / 2000，刷新页面后开始Session，再开始推理/采集及HIL；按真实结果结束每轮。前10有效在线回合仅收集，第11回合开始按原生预算学习；更新与权重回传期间等就绪后再下一轮。此为软件闭环与暂停加载验收，不是成功率提高或真实连续50Hz能力认证。
