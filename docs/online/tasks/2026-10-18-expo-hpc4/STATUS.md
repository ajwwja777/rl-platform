# EXPO HPC 四卡 Online：执行事实

截至 2026-10-18T19:29:21（CST），任务进行中，**尚未通过真实 HPC 四卡验收，不是 Online 已就绪交付**。

用户授权立即预约一个四卡节点，完成 EXPO Learner 迁移、训练与 Web 接入测试；Cobot 保持本地 Actor、相机、机械臂及 HIL。没有授权本任务执行真实机器人动作；本任务所有测试数据均隔离。

- Slurm 作业 **734617**，`acd_u`，单节点4 GPU、48 CPU、384 GB，运行时限12小时，从实际分配开始计时；当前 Pending/Priority。预计开始时间由调度器动态调整，不能作为承诺。
- 代码 `9821542` + `9426151` 已提交并推送 main。独立 worktree：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/expo-hpc-online-20261018/rl-platform`。
- 完整恢复起点 **Online/Actor/VLA98、Critic1960、完成27回合、在线4523步、Replay7824、剩余credit8、批次0/5**。保留25条已确认成功TRAIN示教、完整优化器、目标网络及RNG。没有从SFT重新开始。
- 迁移 **6343文件 / 18,371,635,039字节**，源端与HPC逐文件大小/SHA256完全一致。源生产 `committed.json` 和 `contract.json` 未改变。HPC独立副本仅改bundle绝对路径与经审查的适配代码身份；算法、超参数、总更新预算不变。
- 保留5个有效回合触发、global batch64、UTD20、interval30、RTC5、Edit scale0.01、夹爪反馈执行逻辑。四卡使用原有data-parallel mesh。没有声称四卡线性提速。
- 原LFT2的edit001与batch5 profiles固定使用迁移前适配代码快照，签名仍为`6c8a13201630a2793cc544e117f406d73665e49af971f6b7464003abd9094fd2`；旧Replay/检查点可按原代码恢复。
- HPC独立依赖补充层，未修改共享旧环境；native模块导入通过。旧 `/usr/bin/python3` 为3.6，已修复网关/监控改用指定的Python3.11。
- EXPO整套CPU回归246项通过（虚拟四CPU设备）；网关解释器修复后32项针对性回归通过。真实GPU速度与数值验收仍待分配。
- Cobot Actor98缓存 **2,017,658,362字节** 已逐文件SHA验证并在新独立目录复用；隔离加载/预热69.63秒、单次测试推理0.237秒，输出有限值，无ROS发布、无实际机器人动作。测试进程已正常退出。该单次时间不代表50Hz控制验收或成功率。
- Cobot→A6000→HPC的受限SSH只读监控已实际通过；当前返回`allocation_unavailable`。原58点Learner训练历史已保留。
- Web新增入口 `plug_left_2cam / EXPO / realtime-expo-ft-edit001-hpc4 / 2000`，ID `expo-plug-left-2cam-edit001-hpc4-2000`。已按SHA/操作锁安装目录项和启动脚本；当前缺少资格回执，**不可用**。未重启Web或切换其他模型。

## 已登记的后台执行与证据

A6000产物：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-hpc-online-20261018`。

HPC产物：`/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-hpc-online-20261018`。

Cobot回执：`/home/agilex/jiaan/project/rl-platform/runtime/verification/expo-hpc-online-20261018`。

- HPC编排PID3211238，`orchestrator-status.json` / `orchestrator.log`。只在本作业分配后启动Slurm步骤；验证3/5中断恢复、5/5触发两次20Critic/1VLA、保存/恢复、4→2卡恢复与更新后单卡Actor。验证运行及其Actor100均隔离，生产仍从98接续。
- A6000交付PID1748475，`delivery-status.json` / `delivery.log`。只有上述资格通过后才校验Actor100回传到隔离验证目录，并检查实际Cobot全链路HTTP、Range下载及断开网关后Learner保留。全部通过才生成`runtime/expo-remote/hpc4-ready.json`；不自动加载模型、不运行机器人。
- 原始输入、生产HPC副本、合成验证副本彼此隔离。回传验证Actor100不会作为生产权重使用。
- Learner生命周期由Slurm持有，不随Cobot本地模型释放退出。预约结束不自动续订；后台检测本作业Learner退出后撤销本次入口资格。以后再次预约需核验新作业与状态。

关键证据：`asset-transfer-receipt.json`、`cpu-tests-final.log`、`hpc-source-revision.json`、`cobot-installation.json`、`cache-cobot98.log`、`local-actor98-worker.log`。后台实际结果以带时间的`orchestrator-status.json`、`qualification.json`、`delivery-status.json`与`delivery-receipt.json`为准；文件不存在即尚未完成对应验收。

真实四卡更新耗时、Cobot全链路更新后加载、现场Web暂停加载与真机rollout/HIL学习收益仍未验收；不能把CPU测试或预约成功称为Online已就绪。
