# Teacher5：下次 Online 前的离线对照与接续包（2026-10-14）

**已验证：8234 / Actor4117 通过本轮预登记的开发集保持、原生恢复与导出检查，可作为下一次受控 Online 的接续候选。证据不足：独立自主成功率提高、动作级 Critic 全面正确。** 用户当前使用 EXPO，本轮没有现场安装/晋升、服务启停、机器人动作或 GPU 工作。现场 Teacher5 的 served 仍4063，不能把A6000准备好的4117当作已上机。

## 范围、版本和输入

- 主项目 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`，本轮输出 `outputs/teacher5-preonline-20261014/`（下称 O）。
- 隔离分支 `audit/teacher5-preonline-20261014`，基线 `a5cbfeb`；工作树 `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/teacher5-preonline-20261014/rl-platform`。
- 运行来源：Cobot `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013`。只读取回最新RL checkpoint、served/pending、配套小资产与107条增量约1.04MB；复用A6000已有3623条约44.95MB。整个transfer约46.99MB，不复制整份现场Replay、Stage1或视频。
- 本轮 A6000 CPU172–175、CUDA禁用；EXPO GPU、进程、Replay、模型入口全部保持原职责。本轮只在内存重演少量更新验证恢复，未新增优化配方或发布生产权重。
- TRAIN 3730窗/235Episode，其中今日4集107窗（2自主成功标签、2失败、无HIL）。用户说明插头/夹爪相对位置略变，不能作严格同条件效果对照。
- 旧重复DEV20集/370窗，新重复DEV6集/222窗；旧DEV含人工动作6集，新DEV含人工动作仅1集。身份无TRAIN交集，独立TEST缺失。窗口重叠不按独立样本计算置信区间。
- 配方不变：teacher5、MC.3、BC5/Q.1/delta10、Actor lr1e-5/Critic lr1e-4、batch128、C10、gamma.99、Actor每2步更新、HIL8、recent20窗口混合采样、CPU Learner。执行沿用50Hz RTC/单级EMA，逻辑20Hz；本轮不证明全50Hz物理轨迹与Replay完全等价。

输入、训练源码逐文件SHA：O/source-manifest.json；原始运行前后资产SHA：O/field-assets-unchanged.json。最新checkpoint `3d85d142f231ae1251be9f5f4755b220e19f50962602376845e7a006173104d6`；Replay `d2a74cfce3f3e77e4ff23704dc09e0c9f541988acc014a53b030af0310e109c6`。

## 对照与预登记门槛

比较原上机初始8127/Actor4063与真实Online最新8234/Actor4117，在完全相同缓存输入上CPU前向。每Episode先汇总再等权；固定夹爪不纳入6活动关节误差。人工时刻对比实际人工动作，非人工时刻对比Reference，另保留recorded-action误差、p95/p99/max、逐关节和成对Q。

O/PLAN.json在本轮比较前写入：沿既有review门槛，旧DEV人工/Reference均值最多恶化2%、人工p95最多恶化5%；将相同2%均值保持约束预先用于新DEV。没有根据结果调整门槛；这是工程保持检查，不是成功率门槛。

| 比较（mrad） | 初始8127 | Online8234 | 变化 | 状态 |
|---|---:|---:|---:|---|
| 旧DEV人工MAE | 2.03182 | 2.01409 | -0.87% | 已验证：通过 |
| 旧DEV非人工Reference MAE | 0.74979 | 0.73914 | -1.42% | 已验证：通过 |
| 旧DEV人工p95误差 | 8.94635 | 8.98408 | +0.42% | 已验证：在+5%容差内 |
| 新DEV人工MAE（仅1集） | 2.12262 | 2.08032 | -1.99% | 已验证：通过；无泛化CI |
| 新DEV非人工Reference MAE | 0.62387 | 0.59263 | -5.01% | 已验证：通过 |
| 新DEV人工p99误差（诊断项） | 12.34675 | 12.65773 | +2.52% | 已验证：该尾部指标变差，未隐去 |

新4集TRAIN recorded-action平均误差下降4.52%，Reference拟合下降12.17%；属于训练内拟合。旧TRAIN均值约+0.25%，并非所有指标单向改善。完整分维度/尾部在O/comparison/comparison.json；均值和p95均是逐Episode指标的均值，不冒充把所有窗口混起来的总体分位数。Episode配对bootstrap10,000次/seed42见O/gates.json；单集HIL不制造区间。

![离线保持](../../../../outputs/teacher5-preonline-20261014/dev-retention.png)

## Critic 的限制仍保留

**已验证**：新4集TRAIN Q1对观测行为回报MSE 0.20281→0.06596，旧DEV 0.04717→0.04165，新DEV 0.02988→0.02935。

**已验证**：新DEV min-Q对行为回报MSE反而0.02574→0.02905（约+12.9%）；新DEV失败集评分下调，同时部分成功集低估加重。因此不能将Q1均值改善写成双Critic全面改善。Actor使用Q1、TD使用min-Q，分别报告。

**证据不足**：新DEV唯一人工Episode上，同状态HIL与Actor成对Q的偏好仍不稳。为排除Actor提议变化的混淆，固定同一套初始Actor动作，仅替换Critic，HIL减固定初始Actor的Q1从+0.01860变成-0.00043；固定候选Actor时对应+0.01798→-0.00169。Q2仍为正。只有1集、8个重叠纯HIL窗口，不能由这个近零符号变化证明人类动作劣于Actor，也不能声称动作级价值已可信。O/fixed-action-q.json保存完整Q1/Q2/minQ和控制对象。

这些限制不支持现在放大Q权重、探索、UTD或自动晋升。继续使用已通过保持检查的保守配方，小批真实反馈才可验证自主收益。本轮不借3轮数据反复调参，不把“逼近Reference”当作插孔精度收益。

## 接续、导出和回退检查

**已验证**：泛化原生验证工具，去掉只允许3623条/8127/4063的硬编码；区分served与pending，比较pending参数与完整checkpoint，拒绝重复Replay身份。

- 原始8127包回归通过；最新8234 staged状态及最终8234独立候选各通过64真实输入的原生Actor方法对比，最大误差1.19e-7。
- 内存8次连续更新与4次+序列化恢复+4次：149个状态叶逐位一致；双采样RNG及全部8批身份完全一致。没有把这些探针更新保存成新模型。
- 原生Learner载入8234后旧数据预算0；模拟新增0/1/18/39条分别领取0/1/18/39更新。不重置优化器、target或采样状态。
- 新包served/pending/完整checkpoint均Actor4117；现场原served4063不变。原生运行实例忽略较低版本，回退要按既有流程用新的Actor实例读取旧快照，不能仅覆盖文件期待热降级。
- 14项相关回归通过（teacher5目录/登记、HIL8采样、分离Actor批次）；源码diff检查通过。实际命令见O/COMMANDS.json；回执final-package-64.json、staged-runtime-64.json、initial-regression.json、regression-tests.log。

## 已准备的交付物与下一次起点

A6000独立候选：
`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/teacher5-preonline-20261014/candidates/teacher5-8234/`

完整checkpoint、teacher、norm、Replay、采样状态、pending和served Actor、重定位online.yaml均齐全。包内served设为已评估的4117；原4063单独保留在rollback/actor4063.pkl。这是新离线包的设置，不是修改当前现场served；checkpoint/预算完全继承，旧数据不再领预算。只把包的served从4063替换为4117，没有新优化步骤。

下次接续清单在O/NEXT_ONLINE.json，引用现有方法 `supported-mc30-hil8-teacher5` / ID `plug-v3-teacher5-online`。不是已经新增“8234”网页按钮，也没有排队自动切换EXPO。现场结束EXPO、交还控制权后，只剩既有切换/暂不运动的身份核验和真实Online rollout；不需再重做本轮离线研究。若现场Teacher5资产已变化，先核对新身份，不能用旧包覆盖新经验。

接续后：Learner8234 / Actor4117作为已评估候选起点；新增transition每条1次Critic、Actor每2次更新、每50步保存及生成pending，保留staged发布。新一批权重仍须检查后发布，不承诺每轮自动提升。若不晋升4117，现有现场入口也可用4063继续采集、Learner从8234继续，必须在记录中区分两者。

**剩余证据不足**：新4117物理闭环、独立位置/抓持泛化、自主Online持续收益；正式GPU崩溃底层原因尚未查明，当前继续CPU Learner规避。未来现场连接/服务状态无法提前用今天的离线测试替代，切换当时必须只读核对并做暂停身份检查。本轮没有要求额外现场采集或自动开始机器人运动。

## 实际改动与复现

- scripts/verify_teacher5_candidate.py：支持真实续训版本/Replay长度、pending检查和64输入导出/回退语义检查。
- scripts/compare_teacher5_online.py：固定TRAIN/两套重复DEV，统计六活动关节、HIL/Reference、误差尾部和Q/TD/混合目标；所有输入只读，输出独立目录。
- 本报告及项目/Online/guide事实摘要；未改生产算法、默认参数、训练资产或现场代码。

命令统一加`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 172-175`，Python使用P/.venv/bin/python；绘图为/usr/bin/python3。精确参数、源码SHA、输入SHA和结果路径见O/COMMANDS.json、O/source-manifest.json、O/candidate-package.json。首次transfer在A6000系统Python3.8因Path.is_relative_to不可用而止于解包前，改用resolve().relative_to校验后重试成功；未影响现场。重复传输的失败尝试未当成功收据。

## 2026-10-18 用户授权发布回执

**已验证**：北京时间17:20:58，现用 `plug-v3-teacher5-online` 的 served Actor4063 原子晋升到已评估的 Actor4117；直接读取 Actor服务 `http://127.0.0.1:9141/version` 依次为4063、4063、4117。本次仅发布既有候选，没有新训练或参数修改。上文“未发布”是10月14日历史状态，以本节为最新结果。

- A6000基线 `863d2d9`；发布脚本、完整前后状态和SHA收据：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/teacher5-publish-4117-20261018/{publish_on_cobot.py,receipt.json}`。命令：通过 A6000 SSH 到 agilex@10.12.166.118，用现场 `envs/online/bin/python -` 执行该脚本；仅CPU文件校验和HTTP状态读取，没有调用infer或机器人接口。
- 现场 runtime：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013`。发布前已核对pending/完整checkpoint/Replay与10月14日离线审查SHA完全一致，持有Web两层model-operation锁，在Session waiting_scene、policy_paused、recorder idle条件下备份旧Actor并原子替换served。
- 新served SHA `08dd512f78bbedd7fa1ca7a76db9d9e1ad6e8857236f3ab3f2177f1ecde81ae6`；Learner8234/Actor4117、Replay3730、预算0、training_frozen=false、ready_for_online=true。checkpoint、Replay、pending、profile、teacher、norm、online.yaml前后SHA均不变。
- 模型PID1568092、Session6936f34e-a4a3-4718-ba46-ea8ce6d837c6/Episode2/generation16保持，执行50Hz RTC、逻辑20Hz、单级EMA tau0.08保持；没有启停服务、开始Episode、录制或机器人动作。Session的actor_version仍4063是上一推理的记录，不代表Actor服务未更新；下一Episode动作版本尚待真实推理记录验证。
- 现场旧4063备份与收据：`/home/agilex/jiaan/project/rl-platform/runtime/verification/teacher5-publish-4117-20261018/{actor4063.pkl,receipt.json}`。原生Actor拒绝热降级，回退仍需要新Actor实例读取4063，不能仅覆盖旧文件宣称回退成功。
- 接续策略未改：新transition每条1次Critic、Actor每2次更新、每50步checkpoint/pending；后续候选仍staged，不自动替换执行Actor。用户负责触发Online rollout，本轮不创建后台自动晋升任务。

**证据不足**：4117的自主成功提升、独立泛化、动作级Critic全面正确仍待真实闭环验证。发布就绪不是持续学习必然改善的证明。
