# 2026-10-12 RLT 忠实复现边界与改动来源

## 当前结论

已验证：当前HIL8/Supported是RLT衍生研究候选，不是原样复现。Stage1训练和Actor/Critic网络保留Yyshadow主结构，训练目标、Actor约束、采样、预算、发布及执行已有实质差异。固定上游文件不改不等于运行算法不改，候选adapter会在进程内替换函数。

本次范围：按用户要求先盘点忠实度，暂停新的防抖/算法实施；无训练/GPU/现场写入/模型或服务生命周期。A6000主项目基线e011a2649534eccefc3c382d3b5b9e30de2c96fb，原作者基线c1e40ac360185778c98cf20da2820e22d2d415e7，本地固定上游8cef77eb7c5211b45382bf9199c9cdf0aaf60a19。现场只读GET /api/rlt/session：plug-v3-hil8-explore，Actor3640，paused/waiting_scene，learner_enabled=false；权重SHA020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5。Online入口配置和当前冻结采集不可混同。

## 基线身份与实际差异

原仓库指Yyshadow/openpi-RLT，不是原论文作者官方训练栈，也不是后续参考的Zhengsw03/RL-Token-Pi05-open。原RLT论文pi0.6/50Hz，Yyshadow插孔任务pi0.5/20Hz；本任务应以reproduction_contract.json锁定的插孔任务配置比较，不能拿base默认std.05/actorLR3e-4/delta0或另一个开源仓库当成Yyshadow插孔基线。

已验证：c1e40ac→8cef77e在固定上游仅3文件57增8删：run_online_rl.py、runtime_logging.py及其测试，属于进程就绪/可关闭Learner/日志位置适配。逐字节比对8个Stage1、Actor/Critic、Replay、EnvDriver、离线trainer关键源文件均等于作者基线。更大算法差异位于本项目methods/openpi_rlt/experiments、cobot_adapter和运行配置，不能用上游diff小来宣称系统原样。

| 项目 | 固定Yyshadow插孔基线 | 当前HIL8候选/探索入口 | 性质、动机及来源 |
|---|---|---|---|
| Stage1结构/训练 | pi0.5、H50、内部32D、1个2048D RL token、2层、alpha1、5000步 | 保留；我们右臂7D/三相机/本地示范，4卡FSDP保持global batch32 | 算法基本保持；硬件/数据适配；Stage1审计与reproduction_contract |
| Reference采样 | serve_rlt_policy每次split RNG | 固定key42；额外RTC推理桥 | 行为改变；可复现输入/特征审计，代价是减少Reference随机性；本地实现及诊断需求，不是论文要求 |
| Actor/Critic网络 | 2层256 MLP、C10×7、双Q、Actor Q1、TD minQ | 主结构及上述Q使用保持 | 未换成EXPO或新模型；held-gripper另列 |
| 原有loss/参数 | online BC5/Q.1/delta10、std.002、dropout.5、gamma.99、CriticLR1e-4、Actor周期2、batch128 | 保留 | 这些本来就是插孔任务配置，不能列为我们新加的约束 |
| Warmup预算 | 20000 Critic更新 | 原Warmup选5000，Actor2500 | 训练预算改变；历史20k DEV排序/拟合变差选择早停；并非严格相同步数复现，也未证明真机更优 |
| 数据 | 原作者自己的示范/rollout | 原5k输入池120专家+84rollout共2567；后续Supported重整2496，HIL8加7Ep281窗成2777/208Ep | 任务/数据适配及旧DEV隔离；不能将不同训练池视为完全匹配实验；原5k脚本内部随机transition验证另有划分 |
| Critic目标 | 原生C-step TD | 70%TD+30%观测完整Episode MC，再clip[0,1] | 实质算法改动；降低自举目标漂移。MC30与截断来自项目假设/匹配实验，不宣称照搬某论文；辅助成功MC仍包含人类救回 |
| Actor教师保持 | 无5k教师项 | 对选定旧TRAIN Episode加teacher保持权重50 | 实质目标改变；抑制遗忘，可能过度锚定；本项目研究经验/保持对照 |
| Actor学习率 | 1e-4 | 1e-5 | 超参改变；10月6日跨seed保持不稳定后降低，可能让适应更慢；不能称普遍更好 |
| 夹爪 | 网络7维直接进Q，按任务执行 | 六活动关节保留；Critic当前/目标/Actor-Q路径的夹爪投影到固定物理值 | 执行约束一致性改动；用户明确夹爪固定，防止利用不可执行维度；不是观察到夹爪真实退化 |
| 基础采样 | Replay默认uniform；上游可选stratified已有.4/.3/.2/.1 | 显式采用近期51/Warmup38/human26/全量13，每批128、有放回；不固定8:2成功失败 | 启用上游可选策略+自有可恢复实现；提高近期/人工曝光；不是新创比例，也不是Episode均匀 |
| Actor HIL8 | Actor/Critic共用基础batch | 仅Actor随机8槽换为近期真实Online human/mixed | 新采样干预；10月10日3seed单条新HIL拟合改善5.32–6.48%，不证明自主收益 |
| 新经验更新预算 | 每新增transition预算5更新 | Supported/HIL8预算1更新，Actor仍每2次一次；当前探索采集为0 | 学习日程改变；降低重复优化/保持风险与争用；不是从论文得出的最优UTD |
| 发布/保存 | 每500更新发布Actor、每1000保存 | 每50保存，写pending，served不自动晋升 | 工程/实验管理改变；可回退及避免未验证权重自动上机 |
| 执行 | 20Hz同步C10，ROS适配可限步长 | 20Hz逻辑、50Hz发布、RTC异步每5步重规划、插值+EMA | 实质执行分布改变；去停顿/平顺性，参考RTC和旧XR部署，非原样执行 |
| 训练动作与发布 | 同步每tick bounded命令写action | 每逻辑步取末次50Hz命令；中间发布保留trace | 表示差异，完整动作一致性仍证据不足；见ACTION_CONTRACT，不直接断言是Q异常根因 |
| 探索 | Actor随机高斯std.002 | 专用入口Actor确定性，额外六关节有界相关偏移rho.5，每规划重抽 | 行为策略改变，固定权重不训练；本地研究为补动作覆盖，非原论文探索原样 |
| HIL监督 | HUMAN/MIXED步BC用实际动作、其他用Reference | 保留该BC语义/标签，增加真实coordinator动作/时间审计 | 数据/机器人适配；没有显式“HIL一定比Actor高Q”排序loss；论文的HIL替换Reference不能和社区逐步mask等同 |
| 阶段门控 | 支持critical/full任务；论文用人工切换难段 | 目前采集关闭phase controller，RL从本轮开始参与 | 当前没有按Q分叉自动门控；未实现最近讨论的精细阶段切换 |
| 基础设施 | 原RPC/ROS/日志 | Web/Session/HIL权限、超时/队列、trace、精确输入缓存、目录与SHA审计 | 现场适配/确定性故障修复，不是算法改良证据 |

## 未进入当前运行的内容

两级EMA：已安装独立smooth模型项，当前session仍旧explore；用户新要求下暂缓优先验收。MC100、接管credit截断、失败末端配额：试验未通过保持条件，没有进入HIL8。AWBC/去teacher/提高LR等另一worktree研究不属于当前交付。Training-time RTC、XR前缀训练、Real-Time EXPO-FT、Ruckig、每集固定J2探索均未接入当前权重/入口。不存在因为读了论文就已修改算法的情况。

## 参考资料与真实用途

1. [Yyshadow原仓库](https://github.com/Yyshadow/openpi-RLT/tree/c1e40ac360185778c98cf20da2820e22d2d415e7)：直接代码/任务参数基线。Stage1/网络/原TD/BC主要来源。
2. [RLT v2](https://arxiv.org/pdf/2604.23073v2)：方法目标、执行动作与HIL、chunk/关键阶段/全任务评测。不是MC30/teacher50/HIL8的直接出处。
3. [Zhengsw03另一复现](https://github.com/Zhengsw03/RL-Token-Pi05-open/blob/main/README.zh-CN.md)：用户提供的横向参考，探索/监控/约束/HIL等对照；不是当前代码基线，不能拿其默认值冒充Yyshadow参数。
4. [EXPO-FT v2](https://arxiv.org/pdf/2605.25477v2)：HIL、初始化、视觉、N-step、固定预算/最终性能消融诊断；未换用其编辑策略/训练VLA。
5. [UniSteer v2](https://arxiv.org/pdf/2605.10821v2)：监督重建、误差尾部、成本、遗忘分析；未引入其noise-steering算法。
6. [VLA-Precision v1](https://arxiv.org/pdf/2609.04355v1)：同状态动作配对、人工/自主分开、系统与算法拆分；未引入其完整算法或HIL ranking loss。
7. [eRLT](https://arxiv.org/pdf/2610.00913)：token/效率与评价指标对照；未引入token routing，学习曲线AUC与Critic ROC-AUC不混同。
8. [推理期RTC](https://arxiv.org/pdf/2506.07339)：已用于参考动作的前缀引导和异步重规划；不表示最终MLP Actor做了训练期RTC。
9. [Training-time RTC](https://arxiv.org/pdf/2512.05964)：10月12日研究备选，未实施。
10. [Xiaomi Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)及[XR-0](https://arxiv.org/pdf/2602.12684)：旧现场异步/EMA作为执行参考；训练前缀/注意力方案仍仅研究，没有迁移网络。
11. [Real-Time EXPO-FT](https://arxiv.org/pdf/2609.18207)：10月12日研究延迟/前缀与backup一致性，未实施，区别于旧EXPO-FT。
12. [Ruckig](https://docs.ruckig.com/)：速度/加速度/jerk候选调研，未引入依赖或部署。
13. 用户提供的plug_v3_online_rl_review和截图：提出credit/近期数据/探索/约束问题的线索，不是已验证事实或独立算法依据。
14. 本项目直接实验依据：9月24–26日Warmup选择，9月30日credit实验，outputs/supported-online-20261006/REPORT.md，docs/online/tasks/2026-10-10-online-credit-repair/TASK.md。MC30、teacher50、LR1e-5、HIL8等具体取值是本地研究选择；无可靠原始出处的不补造论文归因。

## 表述纠正及证据强度

原5k最准确称“沿用Yyshadow目标/网络、改变本地数据和训练预算的基线”。Supported/HIL8应称“RLT改进候选”，不再称严格忠实复现。原仓库完全一致也不自动证明在我们机器人/数据上有效；当前修改版亦缺在同一数据、预算、执行方式下，相对忠实基线的完整独立真机优势证据。已验证代码/参数差异；HIL局部拟合改善；自主持续提升证据不足。

历史9月26日记录称offline脚本stratified，但本次核对其_sample_batch实际为rng.integers均匀采样，运行Replay另外支持stratified；不能继续混写。历史外部holdout已反复选型，统一称DEV，不称新独立TEST。Stage1原始14条并非未见测试。未因此改写旧实验数据或资产。

用户要求先结束本次忠实度核对再讨论抖动；因此不继续实现前缀Actor/RTC-SFT/新滤波或新探索，也不自动回滚当前现场。下一决策需先选定严格Yyshadow基线与改进候选的对照边界，不能继续叠改后声称同一算法。

## 证据/复核

产物根：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/reproduction-audit-20261012。
upstream-core-check.json含8文件作者/当前字节比较与SHA；upstream-runtime-diff.txt为作者→部署固定上游diff；candidate-profile-summary.json取真实候选profile；session.json为本次只读现场状态。原始profile及配置不改。主要命令：git diff --stat c1e40ac 8cef77e；git show c1e40ac:<path>逐字节与当前文件比较；读取online.yaml/profile和sampler/trainer；经A6000 SSH对Cobot GET /api/rlt/session。本次只核对证据，不启动训练、不声称新模型验证通过。
