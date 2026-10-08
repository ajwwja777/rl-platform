# 2026-10-08：再次控制发布超时与同步trace写盘修复

结论：这次确实运行了10月7日的新代码，现场连续运行仍未通过；暂停继续Online放行。第476逻辑步单次迟到135.2ms导致保护停止，上次“普通准备开销累计”不足以解释本次停顿。本批已把控制线程同步trace写盘移到有界后台，并增加缺失分段证据；现场代码同步及CPU检查通过，但没有重启运行或真机复验，不能说故障已彻底消除。同一轮HDF5录制随后因writer_queue_overflow失败，必须由录制领域继续定位，不能把异步trace修复当作HDF5修复。

## 实际证据

|命题|状态|数据/版本/边界|
|---|---|---|
|上次修复没有运行|失败|现场源b04a782b，15:41:13新启动；本次日志含新增[rlt-execution-fault]和对应源码行，证明进入上次修复路径|
|本次发生真实长停顿|已验证|Episode10011，step476，late_ms135.2，C10窗口累计此前22.18ms；不是单纯越过10ms旧门限|
|发布调用本身耗时135ms|失败|故障前1190次已完成调用max .622ms；最后一命令到故障无新命令，不能把该段当一次已完成发送延迟|
|新增训练引起此次停顿|证据不足|Learner从启动到退出持续caught up，Replay2777/global7281/预算0；没有本轮更新。日志不支持训练更新争用归因，也不凭这些日志排除所有共享系统影响|
|控制线程同步写USB/FUSE trace|已验证|record_raw_step→CollectionTrace→AtomicEpisodeTraceWriter每步同步os.open/write/close，目标与HDF5均在/dev/sda2 fuseblk。该线程也负责50Hz发布|
|135ms全部由磁盘造成|证据不足|最后record_written单调时间172831.319072810，下一target prepare172831.473373051，相差154.300ms；历史缺少写入完成、下一sample/accept/request分段耗时，不能排除锁/调度/GC/观测处理。mtime/文件存在不作为耗时证据|
|本轮录制合格|失败|HDF5属性completion_state=error、failure_reason=writer_queue_overflow；状态523 sampled/267 written，文件.incomplete。end_timestamp1791445714.69晚于控制故障，不能反推队列溢出先触发控制故障|
|本轮可入Replay/计失败成功率|失败|未完成、无任务结果、无新Replay；保留pending trace与不完整录制，不重新赋成功/失败/HIL标签|
|补丁消除同步trace磁盘阻塞|已验证（软件）|Event门控写盘阻塞时，实际执行器仍完成10逻辑步/25物理发布，命令顺序/间隔保持；错误/溢出/flush超时拒绝Replay|
|补丁后真机持续发布和录制|证据不足|本批不恢复模型进程、不发机器人命令、不做GPU推理；HDF5队列问题未修复|

## 时序与身份

故障日志：`/home/agilex/jiaan/project/cobot-web/runtime/deployment/model-20261008T154113.log`。trace：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt/traces/online/episode_1791445655535594384.pending.jsonl`，SHA1af98ab5ea890bbe157c9a1b65a324bcb4a1ca6d7f4aea4772a94688183cd1f3，476完整逻辑行/1190物理发布。已完成间隔median21.150/p9522.906/p9927.440/max32.897ms，有效46.749Hz；不包含末次未发送空档，配置50Hz不等于实测50Hz。

录制：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/online/supported_mc30/episode_000010.hdf5.incomplete`，UUID bbf2aa91-1d12-4013-9f6f-5983cf134f3e；文件编号与replay Episode10011不同，不能按后缀直接关联其他历史10。原始文件未修改；只读属性通过现场web环境h5py获得。online/system环境没有h5py的失败探测保留，未安装依赖。

实际模型plug-v3-supported-online，Actor3500/Learner7281/pending3640；RTC max_delay4/replan5，logical20/publish50/EMA.08，关节限速.6、夹爪.08且保持夹爪。退出保存后的checkpoint与前日SHA仍da3dda177f204bd1fced04ca566c715c9e209e36bc0747f9fa34f591f548f45e，Replay仍7ec6da3ac6ffd66a1074b87444400cff46860c58018578997f7d1bad1889d0a5。没有新训练或新Actor发布。

![实际故障时序](../../outputs/recurrent-publication-timeout-20261008/01-actual-timeout-timeline.png)

图为单次未完成运行的系统时序，不是TRAIN/DEV/TEST能力评测，没有Episode成功率区间，也不把1190命令当独立任务样本。

## 确定的结构问题与修复

源基线8e39f6337dc65965ac66df8288fadbc42b6d03e4，修复提交5c46cbb502ff3b8e7b8235aaefa9c0773a576311，分支fix/recurrent-publication-timeout-20261008。新增queued_trace.py，改cobot_ros1.py、cobot_online_env.py、async_execution.py；未改采样/模型/Replay/执行频率/RTC/EMA或固定上游。

仅异步执行器启用QueuedEpisodeTraceWriter。控制线程去掉RGB后复制数值metadata并入FIFO；后台单写者执行原有原子writer。容量64个逻辑记录，pending超过1秒、容量满或写入错误均拒绝后续步骤并停止，不无界缓存/静默丢行。不提高50ms发布保护，不跳步补发。开始新Episode、discard/finalize均先flush；Replay准入在终止暂停后等待trace成功写完，超时/错误不放行。故障先暂停再最多2秒flush留证，不延迟撤销发布权。

新增fault evidence：上一逻辑步trace enqueue/旧同步write耗时、loop sample、plan acceptance、request submission，以及后台accepted/written/pending/oldest age/max write/error。线程锁不跨磁盘write持有；元数据副本防止主线程后续修改改变历史。HDF5三相机录制不由该队列承担，此补丁不能增加其吞吐量。

## 验证、现场同步与回退

A6000 Python3.11 85通过/12.68s，现场实际Python3.10 85通过/17.16s。包括9个新测试：阻塞时发布继续、元数据内容/顺序、真实CollectionTrace生命周期、满队列/年龄/写错/flush超时、真实Replay gate拒绝、实际异步构造接线、暂停优先与故障耗时。保留既有暂停/HIL/限速/200ms阻塞等回归。测试是合成CPU/假ROS，不发送真实归位/动作。

首轮现场测试82通过/3失败，均因测试副本的__file__位置改变导致归位脚本路径断言/文件定位错误，自动还原现场源码。随后使用SHA一致的现场原位置既有测试，补丁实际模块85全部通过；两轮日志保留，不将首轮写成算法失败。同步前后运行进程已退出，模型操作锁独占；没有调用恢复POST。

新async_execution SHA05c97a9fcb81983e45924eac34c72b96f0cd815feea93685666608e674401d70；全部4模块新旧SHA与12项配置/模型/Replay保护见install-status.json。同步前后Stage1 PID3419827/start17242517保持，recorder error/complete/非active；保护资产SHA保持，未重启任何服务。

现场备份：`/home/agilex/jiaan/project/rl-platform/runtime/verification/recurrent-publication-timeout-20261008/before/`，旧3模块保留；queued_trace.py是新增文件。回退需退出使用者、持同一模型锁，恢复清单中旧源并核验SHA；不运行旧权重热回退，不自动恢复旧有同步阻塞配置。

命令和产物：A6000 `outputs/recurrent-publication-timeout-20261008/`，tests-integration-command.json、guarded_install_corrected.py、install-status.json记录真实argv/保护SHA；CPU A6000172–175最多4线程、Cobot20/21最多2线程，CUDA空/JAXcpu。现场`runtime/verification/recurrent-publication-timeout-20261008/`留测试/备份/回执。没有整份数据/权重跨机复制。

## 尚未闭环及录制领域交接

先不继续正常Online。HDF5 writer_queue_overflow须单独测出sample→queue→append→flush的吞吐和最长停顿；需核对同一轮267 written附近是否开始长阻塞、queue深度变化、HDF5/USB/FUSE耗时和已有录制暂停/故障终结行为。设备当前剩余9.2TB且无故障时刻新内核USB错误，不能用“盘没满/无dmesg错误”排除I/O停顿。

按项目边界，录制/mask归cobot-dagger，网页生命周期归cobot-web。本批只修RLT数值trace，不静默改录制路径、增大录制队列或替换存储默认。独立录制诊断可在无机器人/模型的临时输出上做，保持原资产只读；其结果未取得前不能说正常Online已具备条件。交接线索在recording-handoff.txt，本报告和PROGRESS.json保存已完成部分，避免再次从头诊断。

本次源修复的下一步验收还需现场新进程运行；若仍超时，新故障分段可区分trace、控制采样、plan接受和请求提交。根因未证实部分继续标证据不足，不能仅提高超时门限掩盖。

MD完整路径：
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/audits/2026-10-08-recurrent-publication-timeout.md
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/MIGRATION.md
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/recurrent-publication-timeout-20261008/REPORT.md
- /data/LFT-W02_data/jiaan/jiaan/agent-guide/projects/rl-platform/README.md（仅摘要，无guide Git提交）
- /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/coordination/recurrent-publication-timeout-20261008.md
