# 2026-10-07：Online 离线准入收尾（未晋升 Actor）

本次结论：离线软件/恢复与资产一致性检查通过，可交回现场做修复后的首轮受控 Online；真实连续发布仍须首轮确认。未完成独立自主能力验收，不能宣称 Online 会持续改善。当前待发布3640在开发集的指标混合，暂不晋升；Actor-only近期Episode采样有小幅拟合收益，但仍是隔离研究，不进入生产。

## 证据表

|层次/检查|状态|直接证据与边界|
|---|---|---|
|本批资产身份|已验证|当前Replay2777窗/208Episode、Learner7281、served3500/pending3640及profile/norm/teacher SHA在CPU核验前后及20:51最终只读均相同。未写生产资产|
|新增数据进入训练|已验证|继承已核实的7Episode/281窗、281个strict回执与journal身份，281Critic/140Actor更新；本批用实际到达顺序复演。自主3成功/3失败、辅助1成功，不合并为自主4成功|
|恢复/导出/实际推理|已验证|实际7281连续8更新 vs 4+内存序列化+4，参数/Adam/target/RNG所有叶逐位一致；pending Actor等于完整state；原生ActorService的64状态CPU推理最大差1.1920928955078125e-7；Replay服务外部采样RNG未宣称精确恢复|
|恢复更新预算|已验证|实际checkpoint warmup_ready_adds_total=2496，2777条在7281步时预算0；合成新增1/18/39条对应预算1/18/39。未执行生产更新|
|运行中降版本回退|失败|原生服务在3640后忽略3500文件；新建服务对象可载入3500。直接覆盖旧权重不是可靠live回退，本批未覆盖。现有暂存发布策略避免未经审查切换|
|发布准备开销修复|已验证|上一批57b378f修复后四频率合成120逻辑步通过，A6000相关125/现场48 CPU回归通过。现场模块SHA b04a782b已同步；不等于真机持续50Hz|
|实际7281开发集全面优于7000|失败|自主5Episode Q1—行为回报MSE .007758→.013177；辅助6Episode动作p95 4.82579→4.85417mrad。提升和退化并存，不用训练loss或Actor Q放行|
|当前动作有显著整体恶化|证据不足|DEV主动六关节RMSE变化很小，多数区间跨0；上述辅助p95差+.02838mrad虽区间不跨0，不能直接换算TCP误差或任务退化|
|Actor-only采样可带来任务收益|证据不足|三seed新单条HIL TRAIN BC6 MSE改善1.00–2.63%；没有独立TEST或真实新Actor结果|
|真实运行连续性与位置泛化|证据不足|本批不启动服务、不加载Stage1、不占GPU、不运动。20条旧holdout已反复选型，仅DEV；独立TEST为空|

状态“失败”对应表中明确命题，不将某个代理变坏解释成整个方法失效。Q与行为实际回报的误差不是反事实动作真值，尤其HIL后的成功不能证明前面的动作最优。

## 身份、配置与命令

代码基线5e5fdcd1cd12e472e8febe5826ff08ff64598ecd；隔离分支experiment/online-admission-closure-20261007。自有实验工具不改固定上游/native全局函数，不修改生产配置/Replay/权重；最终提交见输出PROGRESS.json。

实际现场资产根：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_online_20261006_runtime_v4`。初始根同级`supported_online_20261006_v4`。

|资产|SHA256|
|---|---|
|现场Replay|7ec6da3ac6ffd66a1074b87444400cff46860c58018578997f7d1bad1889d0a5|
|实际7281完整checkpoint|da3dda177f204bd1fced04ca566c715c9e209e36bc0747f9fa34f591f548f45e|
|served3500|a4eb7e7e4fc9517fa87bc416ee66046aa56de0b9b00b5002067d450c79299f95|
|pending3640|62a16e1986517f002141fcff270ae433f99f33af7851f30bd3a160999cd7f83f|
|profile|a36aa64cf055e4cafa477e461b0784bb342e22c3306677022066c3cf42e0935e|
|归一化|4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c|
|teacher|3669525ccea7a22a914fbfb9ea5963d6728f7f904edbaab57bbe1ea305949ae6|
|初始7000完整checkpoint|c58c15ba4854a6a2bfbc05fcbc2c01cbab3cd55deb5a127082381053b21de11c|
|重复DEV journal|edf3f3bc3d6315fbcb55d58ce1ad21a71c18e730fed62407f92d677d3f1fca21|

实际训练沿用MC=.3、Actor/Critic LR=1e-5/1e-4、BC/Q/delta=5/.1/10、active6保持50、reference dropout=.5、std=.002、batch128、Actor每2次更新、UTD1、gamma=.99、C10、target tau=.005于Actor更新时应用。执行logical20/publish50、RTC、EMA=.08、关节限速.6/夹爪.08，夹爪与左臂固定。每50Learner步保存checkpoint，退出也保存；已存7000/7050/7100/7150/7200/7204/7250/7281。此处参数身份不是能力证据。

A6000 nativeCPU环境`scratch/rl-platform/runtime-restore-verification/envs/online/bin/python`；CPU172–175最多4线程。Cobot现有`envs/online/bin/python`，CPU20/21最多2线程；CUDA_VISIBLE_DEVICES空/JAX_PLATFORMS=cpu。命令完整argv和环境在study-launch.json、field-readonly-probe-command.json、final-tests-corrected-command.json、cached-dev-recheck/receipt.json。

主要入口：
- `scripts/offline_episode_sampling.py --actor-only`，其assets/input/output/dev绝对参数见study-launch.json。
- `scripts/verify_staged_runtime_readonly.py --code-root <现场root> --runtime <上述runtime>`；`--budget-only`核对原生加载预算。
- `scripts/evaluate_staged_development.py --dev <A6000重复DEV journal> --output <独立新目录>`，工作机通过SSH stdin临时发起CPU请求；任务专用worker绑定上述现场资产。

## 一次只改 Actor 近期采样

前次同时改Actor/Critic近期采样，失败DEV的Q1回报MSE平均增加.00566314（完整Episode 95%区间[.00319504,.00814431]）。本次把Critic的281个批次逐个固定，仅Actor recent51/128 slots改为Episode均衡，其他77 slots配对相同；三种子41/42/43，每臂281Critic/140Actor，六臂总1686/840。所有1686行指标有限，三基线所有训练状态叶与前次逐位一致。

新HIL实际Actor draw分别32→85、44→94、44→85（每臂17920 draw）；对应Critic83/76/87保持不变（每臂35968 draw）。Actor每两步更新，不能把每个Critic步提议的Actor batch都计为实际采样。本批工具已明确区分proposed与applied；历史输出保留原样，以actor-only-verification.json复算实际计数。专家/HIL/成功/近期池标签重叠，不能把池配额解释成互斥结果比例，也没有强制8:2成功失败比例。

新单条HIL TRAIN BC6归一化MSE改善2.6268%、2.6314%、.9966%；Actor-only失败DEV Q1代理增量仅.0000093143（95%[.0000058449,.0000144969]）。这支持“大部分前次Critic代价来自Critic采样改变”的归因，不能说Critic轨迹完全相同：目标Actor反馈仍会改变后续Critic。重复DEV动作差异极小，单条HIL不足以证明任务泛化。不选择最好seed上线，六个checkpoint均production_release=False；新采样尚未接入现场生产合同。

![Actor-only采样对照](../../outputs/online-admission-closure-20261007/01-actor-only-sampling.png)

## 当前真实待发布3640：全量DEV复核

20完整Episode/370缓存状态，5自主成功/6辅助成功/9失败。拟合目标为HIL步的人类动作、其余步Reference；六个活动关节，夹爪和左臂不参与退化门。按完整Episode求指标、成对bootstrap10000次，不把重叠窗口或三个seed当独立Episode。

|重复DEV|RMSE mrad 3500→3640|p95 mrad|Q1回报MSE 7000→7281|minQ回报MSE|
|---|---|---|---|---|
|自主5|1.40007→1.39135|2.76684→2.73296|.007758→.013177|.009236→.008966|
|辅助6|2.70078→2.70506|4.82579→4.85417|.080657→.065979|.089947→.074179|
|失败9|1.05921→1.05326|2.26518→2.23440|.038799→.046737|.025861→.036116|

自主Q1误差增量95%[.002139,.008699]，辅助p95差95%[.004293,.055931]mrad。完整逐Episode、逐关节、区间见actual-DEV-metrics.json。Q1用于Actor，TD使用min-Q，不混为一个“Q改善”。所有DEV Episode动作变化p99最大.46638mrad；训练内全量208Episode也已做保持检查，见field-readonly-probe.json和actual-pending-fit-summary.json。绝不将拟合Reference或小动作差当作真实插入增益。

原尝试在Cobot既有journal匹配DEV缓存身份没有匹配成功，失败记录保留，不用近似数据替代。最终只把4字段(z_rl/proprio/ref_chunk/action_chunk)作为临时CPU请求经stdin传到Cobot，1,435,474压缩字节；不复制整份Replay/权重/图片、不落现场数据文件。CPU推理约2.12秒，返回小预测报告到A6000；两次独立请求所有预测数组逐位一致，5项资产SHA保持。请求SHA7351e30993f7c7c2d5553c1d9c59742fdabedcab246abd27f7777b869dd5fedb。

![真实待发布开发集](../../outputs/online-admission-closure-20261007/02-actual-pending-development.png)
![真实待发布分关节误差](../../outputs/online-admission-closure-20261007/03-actual-pending-joints.png)

## 本批修改、验证与交付边界

新增私有Actor-batch包装器、原生完整状态只读检查及跨机缓存输入CPU诊断工具，扩展已有采样脚本的Actor-only模式/分维度误差/实际采样计数。18项相关回归通过（34.09s），6个修改Python文件AST通过，git diff --check通过。初次脚本语法错误及一次测试路径写错已修正，原失败日志保留；不宣称全库/Stage1/GPU测试通过。3张图已渲染检查。生产模块本批不变；前次发布器修复沿用。

20:51现场只读：deployment error/运行进程退出、recorder stopped/complete、Stage1 PID2073785/start8759073保持；故障仍显示旧轮execution_clock_late历史状态，未启动新运行验证。安装源b04a782b，7项算法资产未变。

可交付：已修复并现场安装的发布时序、受控Online的数据/训练恢复链路、明确的3500执行/7281学习/3640暂存版本、保存和回退限制。尚不能交付：“新版已改善自主能力”或“持续自动换权重必然越来越好”。当前选择是保持已执行Actor3500，Learner从7281接收新合格数据、候选暂存；原8:2不是固定采样合同。

优先级：P0现场首轮检查修复后的发布/收尾/strict入库；P1带目标位置和介入起止标记的小批纠正TRAIN，补足目前只有1条新HIL的覆盖，每批审查真实batch与目标/Q/动作保持；P2通过后才做同执行配置的新旧Actor冻结配对比较并明确晋升。Actor-only采样只作为下一批离线单因素候选，固定Critic采样/初始state/到达顺序/预算；失败即保留原配方。真实闭环增益不能由更多同分布离线重训单独补证。位置泛化需要未参与选型的完整Episode，若被用于选型即改列DEV。

操作单是本次交付的operator_prompt.txt；沿用已有网页流程，不新增自动发布入口。本报告为事实审计，不修改正式RUNBOOK流程。发布控制、真正短轮结果与后续独立TEST仍需现场。

## 全部产物

A6000绝对根：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/online-admission-closure-20261007/`。
包含REPORT.md、ADMISSION.json、PROGRESS.json、operator_prompt.txt、3图、六臂checkpoint/指标/采样身份、实际7281只读核验/预算/全量TRAIN和DEV逐Episode报告、请求收据、命令、失败记录、测试日志与SHA清单。数据/大产物被Git忽略，图在A6000存在，GitHub报告中的相对图链接不会包含这些未上传产物。

本批MD完整路径：
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/audits/2026-10-07-online-admission-closure.md
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/MIGRATION.md
- /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/online-admission-closure-20261007/REPORT.md
- /data/LFT-W02_data/jiaan/jiaan/agent-guide/projects/rl-platform/README.md（仅事实摘要，不提交guide Git）
- /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/coordination/online-admission-closure-20261007.md（本任务完成登记）
