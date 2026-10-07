# 2026-10-07 Online现有数据与失败末端采样对照

截至18:31现场只读快照，机械臂电源由用户管理。本任务没有启停服务、恢复运行、加载Stage1、发布动作或使用现场GPU。A6000代码基线4be36f3848925e4825d045018bf2e6322c352439；新分析工具仅独立experiment/terminal-exposure-20261007工作区。没有修改现场训练参数、默认配置、生产Replay或权重。以下图和能力诊断均为TRAIN；没有独立TEST，既有反复选型DEV本批未复用。

## 直接结论

- **已验证**：7条新增完整合格Episode（3自主成功、1辅助成功、3自主失败）贡献281窗，全部曾被实际batch抽到；Replay2496→2777，Critic7000→7281共281更新，Actor3500→3640共140更新，新增更新预算0，实际梯度UTD=1。放弃、未标注、退出轮另列，不能计作自主失败。早期成功录制未提交Replay，不能算入新增7条。
- **已验证**：实际执行的Actor仍3500，served文件SHA不变；3640属于保存的pending候选，尚未发布。因此今天这些结果没有比较新Actor的真机能力，不能判定在线学习已经改善/没有改善。
- **失败**：执行可靠性尚未通过。两次新时钟故障分别累计50.5ms（logical_step6）与50.1ms（step49）触发C10守卫。最后一次前Learner已caught up，不能直接归因训练抢GPU。
- **已验证／证据不足**：新自主成功TRAIN的active6 BC误差7000=.00427584→7281=.00388835，降低约9.1%；新失败行为回报误差.11184→.07106；这些是训练内代理改善，不能替代自主任务成功或位置泛化。旧HIL BC误差.070603→.070757略差、Q1对行为回报误差.010897→.015409（约+41%），保留为退化信号，不据此证明实际旧场景能力遗忘。
- **失败**：三条失败末端目标均0，但最新Q1仍.435939/.363756/.242084。10004与10005有所下降，10010没有下降。终止目标0有直接数据/代码依据；非终止失败轨迹含bootstrap，不机械要求所有Q为0。
- **证据不足**：当前3640未满足全面保持/独立泛化证据，未晋升；尚不能承诺Online越来越好。

## 数据与实际采样

|类别|Buffer Episode|Buffer窗|Critic实际draw|Actor实际draw|
|---|---:|---:|---:|---:|
|专家|120|1186|12927（35.94%）|6463（36.07%）|
|自主成功|22|497|14890（41.40%）|7494（41.82%）|
|辅助成功|24|428|3715（10.33%）|1863（10.40%）|
|失败|42|666|4436（12.33%）|2100（11.72%）|

Episode成功/失败约79.8/20.2，窗约76.0/24.0，Critic实际draw约87.7/12.3，Actor约88.3/11.7。因此“8:2”只能近似描述Episode构成，不是强制batch比例。专家作为独立类别；专家本身也命中HIL池，HIL与source/结果/年龄标签重叠，不能相加。近期/旧Online/Warmup实际draw分别14565/235/21168。实际281个Critic批次35968draw，140个Actor批次17920draw；with-replacement重复抽取，2749个独特窗曾被采样，新281全部覆盖。图中的轨迹时间分段不等于语义阶段。

![](../../outputs/terminal-exposure-20261007/current-data-and-sampling.png)

## 用户新HIL

Episode10009／UUID f72960e6-cab7-41d8-9762-159745713d7d；episode_000008.labels.json成功且1次右臂manual接管，单次2.680016秒、53个实际human逻辑步、62个RL步。18个Replay窗中7个含HIL（含2个混合边界），11个纯策略窗；不能把重叠窗口内人工步数相加为独立样本。Native metrics的intervention_count=6是触及chunk数，不等于6次人工接管。

其后首次18更新7225..7242：2304 Critic draws中65来自该Episode、25来自HIL窗；9次Actor更新1152draw中38来自该Episode、13来自HIL窗。全部18窗/7HIL窗覆盖。用户明确澄清10005为自主失败，无HIL。

**已验证**：在同一批7个HIL窗、Critic7281下，真实纠正动作相比离线重算Actor：Q1均差+0.047238、min-Q+0.011520；相比Reference：Q1 -0.015922、min-Q +0.011330。Q1和min-Q不能混用。这里Actor是相同保存输入的离线重算，接管时原始未执行提议未保存，不能冒充历史反事实。纠正动作来自介入后状态，也不能假定它在所有状态最优。仅1 Episode，不给置信区间。

新HIL窗BC7000=.117291→7281=.117242几乎持平；以接管前7204=.119455为参照约改善1.85%，也没有单独对照隔离其他两条失败数据/后续更新的影响。整条辅助轨迹18窗BC下降约2.35%包含非人工部分，不能说成纯HIL显著学会。

![](../../outputs/terminal-exposure-20261007/current-learning-signals.png)

## 单因素离线采样对照

**已验证**：复用A6000完整原7000状态（参数、target、Adam、RNG逐数值叶与现场7000规范化SHA完全一致）。使用首4新Episode204窗，重播7001..7204的实际204个batch身份；后续10005/新HIL/10010不进此对照。仅失败末端到达后替换每batch0/1/4个slot，batch128、更新预算204保持。可用pool根据当时已观察batch身份开放，未使用未来Episode；替换seed42，只有1条新失败Episode，无重复seed或独立TEST。

实际固定：MC=.3、BC=5、Q=.1、delta=10、active6 teacher=50、ActorLR1e-5/CriticLR1e-4、period2、tau=.005、C10、gamma=.99、完整原生状态恢复。基线终点7200失败末端Q1=.654529与现场CPU数值解释.655077相近，但CPU/GPU数值并非逐位复现；不宣称完全相同最终权重。

|7204指标|原实际batch|替换1 slot|替换4 slots|
|---|---:|---:|---:|
|新失败末端实际抽到次数|5|34|121|
|末端Q1（目标0）|.591686|.234939|.044279|
|末端Q2（目标0）|.774348|.628319|.437279|
|旧HIL行为回报MSE|.018859|.019773|.039666|
|旧HIL mixed-target残差MSE|.004295|.006441|.011848|
|旧HIL active6 BC MSE|.070725|.070747|.070762|

**已验证**：增加失败末端暴露能纠正该TRAIN末端Q，支持采样曝光不足是原因之一。**失败**：quota4同时明显损害旧HIL回报代理及mixed目标拟合，Q2仍偏高；不采用、不发布。不能把单个失败末端Q1接近0当成通过。仍待证：小quota、多seed、旧数据保持/重复DEV一起评估；任何候选仍不自动晋升。

![](../../outputs/terminal-exposure-20261007/terminal-sampling-tradeoff.png)

## 超时与收尾等待

**已验证**：现场source async_execution SHA7a46080d2e7ce23ecf48e09bb698947843842d77dbce6430e95cdc6c4bc6a4d9，实际逻辑20Hz／发布50Hz／RTC replan5、delay4／EMA tau.08，动作速率限制不变。最新退出日志model-20261007T181855.log：step49的logical-boundary wait调用_shift_clock，累计C10顺延50.1ms失败；前Learner7281预算0、连续caught-up。未见本次超RTC200ms的直接触发。

最新pending trace1791368627513718769：49已保存逻辑行、122物理发布；间隔min20.319ms、median21.078ms、p95 22.185ms、max31.842ms；发送调用p95.133ms、max.539ms。等待返回到发命令median.970ms，其中已计控制采样/准备后未细分部分median.764ms；持续向后顺延积累，真实发布并非精准50Hz。最新已存前缀跨全程总顺延159.85ms，**不是本次C10的50.1ms**。失败partial step49没有完整回执；缺sleep/authority探针/记录/调度细分，不能补造最后耗时或认定网络/CAN、GPU、断电根因。断电发生在故障报告后，不将其解释为此前故障。

![](../../outputs/terminal-exposure-20261007/recurrent-clock-timing.png)

**已验证**：标签到Replay提交10005=5.630秒、HIL10009=4.627秒、10010=10.993秒。输入审计第一到末回执构建跨度分别5.272/4.231/10.632秒，最后回执到Replay确认均约毫秒量级。等待定位在Replay窗口构建/特征与输入审计阶段，尚未细分各成本；页面HTTP总时长没有精确trace，不能称其等于上述标签到提交时间。Learner在提交后才获得新预算。不能为了快跳过严格输入身份审计，亦未本批重新调用Stage1前向。

## 身份、复现和保存

**已验证**：新增281窗的pre-submit strict receipt全部通过，全部序列化数组的dtype/shape/finite/SHA逐窗与实际journal完全一致，0不匹配。只证明结构身份进入Replay；不证明图像语义、真实插入成功或自主能力。

现场journal：/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_online_20261006_runtime_v4/replay/replay_journal.pkl，36,697,696 bytes，SHA7ec6da3ac6ffd66a1074b87444400cff46860c58018578997f7d1bad1889d0a5。served Actor SHAa4eb7e7e4fc9517fa87bc416ee66046aa56de0b9b00b5002067d450c79299f95；normSHA4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c；profileSHAa36aa64cf055e4cafa477e461b0784bb342e22c3306677022066c3cf42e0935e。数据事实来源为用户结果标签、source trace、Replay和batch身份；没有新增语义质量验收。

**已验证**：保存间隔原配置50 Learner步；现场完整checkpoints包含7000/7050/7100/7150/7200/7204退出/7250/7281退出，pending_actor/history包含相应3500/3525/3550/3575/3600/3602/3625/3640；逐文件SHA索引latest-sampling-and-checkpoints.json。恢复学习需要配套完整optimizer/target/RNG，不能只回退Actor后沿用未来Learner状态。本批没有修改保存间隔或发布。

所有产物在A6000 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/terminal-exposure-20261007；仅取回204个新派生窗1,981,452 bytes（SHA bdd00ed7ceafb8deaabb6f659801c007063fd6cf515133f78017fd88ad4febc3），未复制整份现场数据或权重到笔记本。训练launch.json给出完整命令/CPU环境；执行scripts/offline_terminal_exposure.py --assets <现有A6000初7000目录> --input <本输出目录> --output <独立目录>。三分支所有metrics有限，sampling-preflight.json验证empty pool/zero quota/RNG保持、quota4恰替换4slot和输入不变。NumPy解释先在A6000对原生CPU验证Actor/Q最大差约1.6e-6；现场只用CPU20/21共约6秒读取解释，未启用GPU或训练；生成图使用系统Python plot_current_analysis.py。单个失败Episode、单个新HIL不计算不确定性；多Episode区间由完整Episode重采样，4,000 bootstrap。工具环境路径和argv保留launch及审计脚本，不将测试通过写成模型能力证明。

## 当前交付边界和后续问题

**证据不足**：本批只完成数据核对、三臂离线对照、最新候选数值分析与图；没有修复本次累计超时、没有新Actor发布、没有机械臂恢复/运动、没有独立任务能力验收。早期故障快照录制仍open/incomplete；2026-10-07T18:39:28.669309+08:00再次只读确认已由现场收尾：stopped/committed、episode_000010.hdf5、3000帧（final-field-readonly-state.json）。本任务没有收尾/赋标签，故障轮不计本次新增7条。

优先问题：执行等待后准备与记录的时钟累积，需独立CPU模拟和细分计时，保留单次严重迟到/禁止补发/RTC边界；Replay构建成本需拆特征获取、身份审计、快照写入；算法侧仅进一步小quota与旧HIL保持对照，当前quota4拒绝。先消除执行链路中断及候选保持风险，再评估发布后的真实效果；不能用本批训练拟合替代这个证据。

分析工具最终输出守卫拒绝资产根及其子目录作为实验输出，避免覆盖不可变初始资产；输入/生产训练实现不变。新增7项工具合同回归覆盖边界、精确配额、原batch不变及RNG保持。
