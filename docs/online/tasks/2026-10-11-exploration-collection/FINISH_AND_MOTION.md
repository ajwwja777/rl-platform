# 2026-10-11：末端往返、历史目录及收尾等待

状态：代码修复及 CPU 回归已通过；现场同步状态另见 outputs/exploration-collection-20261011/finish-history/field-sync.json。缓存缩短实际等待的幅度尚待下次模型运行验证，未宣称已消除末端往返。

## 数据与版本

- 分析快照：专用录制目录 episode_000008–000012.hdf5，UUID 顺序 f154fff4-c772-4055-9351-56bd7e675921、14060abb-12bd-4fe3-ae3d-a74ff26ff90c、56da005e-d27f-4cea-9dd5-dea937cb24d4、8319b3c8-426a-44b0-984e-9e74350cfa67、177ea14c-c2aa-43b0-9fc4-3c9d4e4fd004。均 operator success，HDF5/trace 未见 human 步；不能用累计 Session intervention_count 代替本轮 HIL。原始目录见本任务 RESULT；本批时间与 native12–16 trace 对应，但未将不同 ID 空间冒充统一 UUID。
- 冻结 Actor3640 SHA020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5，norm SHA4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c；Learner=false。逻辑20Hz、发布50Hz、RTC、replan5、EMA .08、J2 探索 sigma3.429mrad/截断6.859mrad。
- A6000修复基线 RLT672eadf7、web36cdc6ee；现场执行 SHA 见 motion-evidence.json 中每行探索元数据。现场原配置 step_trace_stride=0、replay_feature_batch_size=16；后者在当前 on-demand 路径并未实际启用 GPU batch。
- 这些是后续 TRAIN/重复 DEV 候选，不是独立 TEST。用户仅确认此前新增四轮目标位置相同；本次不推断其他轮次位置相同。用户继续采集期间，本报告固定上述五轮，不声称包含之后全部新增数据。

## 已验证：历史列表被普通采集的迟到响应覆盖

initializeCaptureProfile 初始化普通 test 目录，未等待 refreshConsole 确认当前 RLT 模式即 prepareStorage。普通目录请求的迟到响应发出未区分 runtime 的 storage-used 事件，把 historySelectedRoot 覆盖为 test。模型保存位置仍正确，故出现上方专用目录、下方47条旧数据。此前只测目录 API 和隔离模型组件，漏测 app.js 的交错返回，这是前次验证边界遗漏。

修复：等待模式刷新；普通请求跨模式后不应用结果；事件携带 runtime；模型管理目录优先用于历史查询；过时历史请求不能覆盖总数。恢复被 CSS 隐藏的“刷新”按钮。专用目录保持固定，隐藏不适用的浏览/更换按钮并说明用途；普通采集仍可浏览/选择。

已验证：竞态测试在原版2失败1通过；修复后全部80个前端测试通过。测试执行真实 app.js 的对应函数与事件处理，不冒充实际浏览器交互或机器人验收。

## 已验证：收尾存在相同输入重复前向

五轮41/67/54/44/76个Replay窗口，输入回执生成跨度10.947/18.471/14.504/11.842/20.682秒；相邻回执约.274–.279秒。实际代码逐窗口补算当前和next特征，缓存按 observation index 区分。trace保真重索引使相同完整观测出现不同索引。

564次端点引用中只有295个不同输入指纹，269次重复（47.7%）。全部重复的 z_rl/proprio/ref_chunk native dtype、shape、SHA均一致。已验证是可避免的重复计算；既有日志不足以将每毫秒精确分给RPC、GPU和序列化，不将整体时间全部归为GPU。

修复 replay_feature_cache.py：仅已登记 plug-v3-hil8-explore、trace合同、on-demand收尾启用；单轮内按整个观测内容（全部键、RGB、state dtype/shape/value、prompt、RTC prefix）缓存本轮新计算输出，深拷贝防别名污染。仍保留原索引、窗口、奖励、动作、精度及strict输入审计；不跨轮缓存、不复用附近姿态或未经指纹核对的live anchor。固定Stage1使用key42是适用前提，不推广至随机模型。COBOT_RLT_REPLAY_EXACT_CACHE=0可关闭；其他模型及dense批处理路径保持原行为。

已验证：53项相关CPU检查通过，包含真实原生C10+CollectionTrace、RL/HIL、成功/失败、输出逐数组一致，重复输入调用4→3；输入任一关键字段变化均不命中，异常恢复Replay查找方法，live feature provider从不替换，跨轮不共享缓存。日志新增 COBOT_REPLAY_BUILD 的请求数/命中数/哈希、前向及总构建时间。

证据不足：没有本批真实GPU测速，不承诺20秒降到1秒；理论减少的是约一半重复引用，不等同整体时延严格减半。当前在内存中的旧EnvDriver需下一次用户重载后才使用补丁。

## 已验证：末端大幅往返在新计划的Reference目标中已存在

查看右腕及高位相机帧并对齐数值trace。突出案例HDF5 episode10 / native14，末尾12.24–13.79秒每约.25秒切换一次request；J2 Reference变化为 +10.25、-13.67、+21.18、-26.49、+25.41、-14.38、+8.24mrad。七次都对应新的计划来源；同次探索偏移变化绝对值不超过4.37mrad。队列目标、发布目标及反馈出现相应往返，不能把它解释为只在网页画面抖动或收尾GPU忙。

五轮末8秒，扣除记录偏移后的Actor与Reference J2相关 .985–.998，RMSE1.55–2.37mrad；当前Actor主要跟随Reference，未形成消除往返的修正。此处末8秒只是时间窗口，非语义阶段标签；关节变化不是TCP毫米位移。

证据不足：Reference本身已经受当前视觉、状态、RTC前缀与此前探索动作反馈共同影响。代数减去offset不是关闭探索后的反事实轨迹；这些结果不能单独判定纯Stage1、RTC或探索哪一个是唯一根因。也未证明当前训练Critic造成这批往返：本入口全程不学习。

当前最小定位对象已经缩到“同一末端观测下，RTC/前缀导致新计划首个待执行动作如何变化”。后续应保留Actor和执行频率，先离线比较相同原始输入及前缀的计划连续性，再选择单因素候选；本批未盲目改探索幅度、RTC、EMA、奖励或权重，未让机器人自动验证。

## 产物与复现

A6000 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/exploration-collection-20261011/finish-history/：

- extract_motion.py → motion-evidence.json、wrist-ep8…12.jpg、terminal10-cam_high.jpg / cam_right_wrist.jpg；只取小型数值证据与接触图，未复制整份HDF5。
- plot_motion.py → terminal-motion.png、motion-summary.json；terminal-plan-jumps.json为七次计划切换。
- check_duplicate_features.py → duplicate-feature-check.json；只读已有严格回执，未运行模型。
- baseline-web-tests.txt、web-tests.txt、python-tests.txt：基线失败与修复回归；现场同步/资产与PID收据单独保存，不以源码同步冒充运行生效。

命令：通过A6000 ssh agilex@10.7.165.64，用Cobot web/.venv/bin/python执行只读HDF5分析；A6000 python3 plot_motion.py。前端 NODE_PATH=/home/LFT-W02/.hermes/hermes-agent/node_modules /home/LFT-W02/bin/node --test tests/*.test.js；Python使用scratch/rl-platform/runtime-restore-verification/envs/online/bin/python -m pytest，PYTHONPATH指向隔离RLT及固定上游、JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES为空。完整执行文件保存在同输出目录。

## 已验证：同步与HTTP交付

RLT5517f257、webc2c93d5b先A6000提交并push main；6个源码/静态文件在现场 waiting_scene、policy_paused、recorder stopped、无模型operation的操作锁内同步，逐SHA一致。5个进程PID/start ticks与Actor、norm、registry、Replay SHA保持；现场CPU精确缓存/输出隔离/RTC变化不命中验证通过。四个网页资源HTTP内容与安装文件一致，实际combined history接口返回专用目录27条记录（该查询时点；后续可继续增长），不是test47条。未调用加载、启停、运动或GPU推理。

网页刷新即读取新JS/CSS；缓存需下一次用户重载当前模型运行进程，不能将磁盘同步当作旧进程已应用。停止旧进程/加载模型仍由现场用户掌握，本任务未代操作。回退原文件备份 Cobot /home/agilex/jiaan/project/cobot-web/runtime/verification/episode-finish-history-20261011/before/；缓存也可用COBOT_RLT_REPLAY_EXACT_CACHE=0关闭。修复前后真实耗时及末端动作改善均仍待验证。

现场收据：field-sync.json、http-verification.json；时延/重复引用图finish-cost.png；没有后修复GPU实测柱图。
