# 2026-10-07 Online 参数、训练与下一轮 rollout 决策

状态：审计和六组单因素 CPU 对照完成；新采样未采用、无权重发布。现场控制发布时钟仍失败，不能宣称已经具备可靠连续 rollout 条件。

## 结论与决策

|结论|状态|直接证据与边界|
|---|---|---|
|新增经验进入训练|已验证|281新增窗口 strict 回执逐数组SHA匹配Replay，281全部曾被采样；结构身份不证明任务语义/时间配准全部正确|
|Learner确实更新|已验证|7000→7281，Critic281/Actor140次，预算归零；checkpoint与batch日志可追溯|
|今日执行了新Actor|失败|实际served仍3500，待发布3640。不能用今日rollout说明3640收益或无收益|
|当前HIL学习弱是Q项方向错误造成|证据不足|固定7281、六个实际batch身份重算梯度，Q项和保持项对唯一新HIL探针局部均有帮助；混合BC四批方向冲突。不是历史梯度重放或全局证明|
|只改近期Episode采样可全面改善|失败|3种子HIL拟合均改善，但旧DEV失败Q回报误差均值上升；不作为现场替换依据|
|Online自主能力越来越好|证据不足|仅7新增TRAIN Episode；无新Actor真机结果，无独立TEST，位置标签不完整|
|连续执行可靠性|失败|最新累计控制发布延迟50.1ms触发退出；Learner当时已追平，不能直接归因训练GPU争用|

建议保留当前MC30低学习率基线，不关闭Q、不提高UTD、不去除HIL标签、不重新训练Stage1。先修控制发布时钟，再采少量有明确位置和接管边界的新TRAIN轨迹；按批审核候选，执行Actor在批内固定。

## 当前实际数据与状态

现场资产根：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_online_20261006_runtime_v4`。
Replay 2777窗口/208 Episode，原2496窗口+281新窗口。新增7条：10000/10001/10002自主成功175窗；10004/10005/10010自主失败88窗；10009辅助成功18窗（7含human）。10005按用户纠正为自主失败。7条不是独立TEST，不用窗口数计算成功率置信区间。结束异常的episode_000010.hdf5后来已收尾3000帧，但不混入上述7条合格训练Episode。

|互斥Episode类别|Episode|窗口|本轮累计Critic实际抽样比例|
|---|---:|---:|---:|
|专家|120|1186|35.94%|
|自主成功|22|497|41.40%|
|辅助成功|24|428|10.33%|
|失败|42|666|12.33%|

专家单列；HIL属于重叠标签，不能再作为互斥第五类。Episode成功/失败约79.8/20.2，窗口76/24，累计batch87.7/12.3。按实际到达阶段，失败draw比例8.73%→14.52%→17.85%→16.67%→22.06%；当前末段已约78/22，不能用全程平均认定接下来固定12%失败。

最近一条HIL：Episode10009、UUID f72960e6-cab7-41d8-9762-159745713d7d。人工接管一次2.680016秒/53独特逻辑步，延续到成功，无接管后自主续接。原生intervention_count=6是受影响chunk数，不能称6次接管。首次18更新，Critic2304draw中25个HIL，Actor1152draw中13个HIL；配置HIL20%绝不是新HIL占20%。

实际Learner7281/待发布Actor3640；执行3500。每50Learner步保存完整状态，7000、7050、7100、7150、7200、7204退出、7250、7281退出可追溯。回退训练需Adam/targets/RNG完整状态，不只Actor文件。正常新增数据触发学习，但候选staged，不自动替换执行Actor。

## 参数判断

|参数|实际值|本轮建议|
|---|---|---|
|Actor/Critic LR|1e-5 / 1e-4|保留；历史较大ActorLR已有保持损害|
|BC / Q / delta / active6保持|5 / .1 / 10 / 50|保留；不能按scalar loss大小猜梯度；当前新HIL局部诊断不支持关Q或移除保持|
|MC混合|.3|保留；不把所有失败中间动作Q强压0|
|gamma / chunk|.99 / C10|保留；成功terminal reward在末步时目标gamma^9=.913517，失败terminal目标0|
|batch / UTD / Actor周期|128 / 1 / 2|保留；UTD指每新增窗口一次Critic更新，不是每Episode一次|
|target tau|.005|仅Actor更新时更新targets；140次后初始target系数约.4957，不等于“半数经验没学”|
|reference dropout / fixed std|.5 / .002|本次固定；尚无单因素证据要求改变|
|sampler slots|recent51 / warmup38 / HIL26 / uniform13|各池重叠；本次只对照recent池内抽法|
|执行|逻辑20Hz / 发布50Hz / RTC / EMA .08|保持用户已体验的设置，与算法实验分开|

Recent按最大EpisodeID−19划窗，不是最近20条有效完整Episode；中止ID也消耗跨度。当前仍覆盖全部7新Episode。HIL池1438窗口=1186专家+245旧非专家HIL+7新HIL，旧数据占绝大多数。Actor BC在人类source2/3上拟合实际动作，其他步拟合Reference；不能说失败policy窗口的BC在模仿失败Actor。Critic使用实际动作；Actor优化Q1，TD用min(Q1,Q2)。左臂/夹爪固定，判断动作拟合与退化以右臂active6为准。

## 已有实际模型对比及本轮梯度证据

原7000→实际7281，3条新自主成功TRAIN的BC6约下降9.1%；唯一新HIL相对7000几乎不变，相对进入HIL前7204下降约1.85%。旧HIL部分Q—行为回报代理误差恶化。证据见上一审计`outputs/terminal-exposure-20261007/REPORT.md`。

在同一个实际7281 Critic上，新HIL实际纠正动作相对重算Actor的Q1高.047238，相对Reference低.015922；min-Q对应+.011520和+.011330。重算Actor不是历史接管时记录的未执行提议；一个TRAIN Episode不能证明全局排序正确。故“现在HIL Q低”必须说明比较对象，不能泛指低于Actor。

本轮固定7281，取最后六次实际Actor batch身份7270/7272/7274/7276/7278/7280，受控重算dropout与梯度。唯一探针是10009的human步active6，BC=.130993；不是六个独立Episode。BC梯度范数3.43–13.76，Q .526–.625，保持5.90–12.92。Q和保持与探针梯度cosine六批均正，BC四批负。包含保存Adam状态的六个局部一阶预测均为改善，Q0略弱。用零更新capture optimizer验证原生loss差0、梯度最大差1.4782e-5、参数不变。只支持局部方向诊断，不证明Q是正确价值函数。

## 六组单因素离线训练

固定真实7000完整参数/Adam/targets/RNG、实际7条Episode到达順序和281更新预算。仅recent51slots从窗口均匀改成先Episode均匀再窗口均匀，其他77slots及最终排列在成对实验中一致。3个采样种子41/42/43，不是三个网络初始化种子。6组共1686 Critic/840 Actor更新，CPU172–175，不占GPU，约44.3秒。全部281步指标有限、每组140次Actor更新，无未来Episode被训练采样。

|采样种子|新HIL draw 窗口→Episode|新HIL TRAIN BC6 MSE 窗口→Episode|旧DEV失败Q—回报MSE 窗口→Episode|
|---|---:|---:|---:|
|41|83→188|.117227→.114059|.030338→.053328|
|42|76→167|.118140→.114970|.042252→.035686|
|43|87→176|.117684→.116499|.036626→.037191|

新HIL拟合降低1.0%–2.7%，旧HIL拟合也略改善。但DEV失败Q误差跨种子均值增加.005663，先在同一Episode内平均种子差、再按9条完整Episode bootstrap，95%区间[.003195,.008144]；此区间不包含种子总体不确定性。因保持代理存在代价，不采纳为生产默认，不只挑seed42。

DEV20条370窗已反复用于选型（5自主成功、6辅助成功、9失败），不是TEST。新旧BC目标相同：human用实际动作，其他用Reference。每Episode计算active6物理RMSE再求组均值；不能与早期仅Reference-fit物理误差直接比较。Episode采样相对窗口采样，DEV自主BC RMSE差−.004957mrad，Episode95%区间[−.016764,.005025]跨0；辅助差−.003745mrad，幅度很小。新HIL只有1条，无Episode CI。所有TRAIN组先Episode内平均再跨Episode平均，重叠窗口不作独立样本。

6个候选虽然都标step7281/Actor3640，但SHA不同，也不同于现场实际3640；必须用variant+seed+SHA定位。标记production_release=False，未导出/发布为Online合同。

## 下一轮执行方案（建议，尚未现场执行）

1. **P0：先修时钟与故障可观测性。** 50Hz发布间隔中位21.078ms、p9522.185ms，发布调用本身最大.539ms；等待后的准备存在累计漂移。把准备/排程/发布分段记录，修绝对deadline对齐与准备位置；保留超限停止，不能只增大容忍值或通过突发补发追时钟。失败partial step也必须落trace。先用假时钟/延迟注入CPU验证，再由现场负责人受控验收；本轮未实施此修复。
2. **P1：下一批8条左右、有明确位置的TRAIN。** 保留served3500作为固定采集策略，中心2、左3、右3，位置以既有安全范围和可复现标记为准；记录偏移量或照片/位置ID。出现可恢复偏差时纠正，有条件时纠正后交回自主执行，记录接管前/中/后。凡有接管一律辅助，不为8:2配比刻意制造失败。此分配是采集预算建议，不是统计验收样本量。
3. **P1：批内学习可继续，批内Actor不发布。** 保持UTD1及50步完整checkpoint。每6–8条检查严格回执、来源/位置/年龄实际draw、Q1/min-Q同状态配对、失败terminal残差、旧TRAIN与重复DEV active6误差尾部；出问题保留数据并回退完整训练状态。不要以ActorQ上升或低TDloss单独批准。
4. **P2：优先研究采样对Actor和Critic的不同作用。** 当前共用Episode均衡的Actor拟合收益与Critic保持代价不一致。下一次单因素离线对照应仅改变Actor使用的近期Episode权重/采样，Critic批次保持历史配对不变；固定loss、LR、预算、初始化与数据。若旧DEV失败Q继续恶化或动作尾部退化，弃用该候选。尚未实施或声称有效。
5. **P2：审核真正新Actor后，才安排小规模冻结成对比较。** 固定位置/RTC/EMA/频率对比初始3500与审核后的新候选；这批用于选择就是DEV。最终另留独立TEST。冻结验收过后再分批Online，不反复要求用户评测未变的旧Actor以证明学习。

当前不推荐立刻大批量rollout，也不要求现在加载这六个实验权重。下一次真机前首要待办是发布时钟修复；现有算法证据不足以保证持续自主收益，但也不支持把当前问题统一归咎于Q方向错误。

## 来源、命令与复现

A6000主路径：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`，输出`outputs/online-rollout-strategy-20261007/`。基线代码d92d54d57b2762895e6b440d13d620feaade7e8d；隔离分支experiment/episode-balanced-recent-20261007。固定上游只读。未改生产Replay/配置/权重、未启停服务/加载Stage1/运动/占GPU；现场CPU梯度诊断仅2核、9.34秒，参数无写入。

关键身份：
- 现场Replay SHA256 `7ec6da3ac6ffd66a1074b87444400cff46860c58018578997f7d1bad1889d0a5`。
- 实际7281 `da3dda177f204bd1fced04ca566c715c9e209e36bc0747f9fa34f591f548f45e`。
- served3500 `a4eb7e7e4fc9517fa87bc416ee66046aa56de0b9b00b5002067d450c79299f95`。
- A6000完整7000 `c58c15ba4854a6a2bfbc05fcbc2c01cbab3cd55deb5a127082381053b21de11c`，与现场初始数值状态一致证据见上一报告。
- profile `a36aa64cf055e4cafa477e461b0784bb342e22c3306677022066c3cf42e0935e`；norm `4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c`；teacher `3669525ccea7a22a914fbfb9ea5963d6728f7f904edbaab57bbe1ea305949ae6`。
- 重复DEV journal `edf3f3bc3d6315fbcb55d58ce1ad21a71c18e730fed62407f92d677d3f1fca21`。本轮验证phase/Episode不重叠，未重新完成原始视频/UUID近重复全审计。

完整运行命令/环境：`study-launch.json`；预注册范围：`EPISODE_SAMPLING_PLAN.json`；实际新旧数据到达与所有抽样身份：`episode-sampling-study/{arrival_schedule.json,sample_indices.npz,metrics.jsonl}`；各步评估与六候选：同目录`comparison.json`及variant子目录。只额外取回77派生窗口747901bytes，没有复制整份Replay或现场权重。

图：`01-actual-sampling.png`、`02-gradient-direction.png`、`03-matched-sampling-study.png`。其他：`period-metrics.json`、`gradient-sensitivity.json`、`DEV-paired-episode-intervals.json`、`verification.json`、`artifact-manifest.json`。绘图命令：`python3 scripts/summarize_episode_sampling.py /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/online-rollout-strategy-20261007`。测试：原生CPU环境pytest新增4项与既有terminal工具7项共11通过，日志`sampling-tool-tests.log`。所有结论限此资产快照，不能替代后续现场状态。
