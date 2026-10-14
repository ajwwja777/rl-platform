# EXPO 原仓库同步 Online 对齐 · 2026-10-14

状态：实现候选尚未交付，等待 Cobot GPU 空闲。2026-10-14下午已恢复连接；Cobot在12:48重启，旧stream64测试无完成回执，已随旧启动结束，不能标为通过。当前Pi05 PID60095/端口8000占约11GiB，用户明确要求保留正在使用的服务。本任务未停止它、未启动新GPU任务、未同步生产代码或执行机器人动作。

用户授权按新传入的 train_pi_robo.py / train_pi_robo_async.py 对齐并交付可用于 Web rollout/HIL 的 Online 版本；采用同步 recipe。用户明确确认 realtime_expo 整批示教成功，允许使用 TRAIN 前25回合。

已导出独立 seed_success25_20261014_v2，25回合3301帧，源HDF5 SHA/UUID与 SFT TRAIN manifest核对。选中源ID：1,2,3,4,6,7,8,9,10,11,13,14,15,16,17,18,19,21,22,23,25,27,28,29,30；DEV5未用。6回合尾部一帧POLICY被SFT排除，本次沿用同一人工片段，成功确认与裁剪理由写入独立manifest；原始文件不改。第一次导出的部分目录没有manifest，不可用。

候选：原仓库示例N32/edit32/filter32/filter_n_edit1/edit_scale0.1、batch64、UTD20、step_interval30、offline_ratio0、25成功示教预填；前10个有效在线回合仅收集，第11个完成后开始训练。保留Cobot左臂7D/双腕相机/H16/r8/delay2/逻辑10Hz。批次完整损失、优化器次数及顺序不缩小；尝试逐小批搬运/激活重计算、原生索引与增强随机数整批生成、目标Actor EMA保持CPU。以上内存候选仍需GPU更新/恢复验证。

证据：scratch/rl-platform/expo-sync-alignment-20261014/progress.json、web-tests.log、a6000-cpu-contracts.log、a6000-native-sampling.log；现场 runtime/verification/expo-sync-alignment-20261014/{native64-platform,mapped64-platform,split64-platform,stream64-platform}。

已确认 native64 申请27.62GiB显存失败；mapped64申请13.56GiB失败；split64因主机MemAvailable保护退出。stream64最后可见进入更新，其最终状态在断联后未知。A6000网页接口12项、快捷键11例通过；实际原生采样函数拆分前后全部数组、terminal/truncation/success选择及RNG逐位一致；分块masked loss/gradient与调度门槛通过。Cobot CPU全合同第一次因合成黑图不满足原生亮度断言失败，修正后的cpu-contracts-v2.log现已取回，4组原生合同检查全部通过。

独立worktrees：scratch/rl-platform/expo-sync-alignment-20261014/{rl-platform,cobot-web}，分支feature/expo-sync-alignment-20261014。当前未合入main。候选登记独立id expo-plug-left-2cam-native-sync-2000，独立Online输出；现有旧Online入口未替换。

唯一下一步：用户释放Cobot GPU后，重新核对现场任务/录制/进程身份，再启动已准备的-r2/stream64-platform隔离验证。GPU单次更新、数值对照、11回合闭环与全状态恢复通过后，再push/合并、串行校验安装和暂停加载交付。当前禁止以候选存在或CPU检查替代可在线学习验收。

## 2026-10-14 下午恢复检查

- 旧stream64最后约177秒，GPU14GiB、MemAvailable18GiB，之后日志截断；journal上一启动止于00:08:26，当前启动12:48:42。没有证据将断联归因OOM或电源，旧测试未完成。
- 已将四次GPU实验、完整CPU合同和seed manifest复制到A6000 field-evidence；进度与保护服务身份在progress.json。
- 网页24项选定回归通过；新增native计数/中英文断言后14项DOM回归通过（与24项重叠）。后台12项已通过。首次从错误cwd启动一个旧前端测试失败，按package.json在app/backend运行后通过；不是代码错误被忽略。
- 已准备verify_expo_sync_lifecycle.py：25示教初始化、unknown/abort排除、10采集回合、第11回合1份更新预算、失败回合仍从success seed训练VLA、重复finish、学习时禁止新回合及新进程恢复。该GPU闭环尚未执行。
- -r2验证目录仅隔离脚本已准备；production源码/注册/模型未切换。现场ROS readiness同时报告handover_publisher_unavailable，未来机器人运行还需现场硬件就绪，本对话没有启动硬件。
