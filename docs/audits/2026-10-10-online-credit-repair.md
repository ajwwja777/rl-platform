# 2026-10-10 Online credit / HIL8 交付审计

## 结论与边界

已验证：候选 `supported_hil8_clip_20261010`，Learner 7281 / Actor 3640，来自完整 Supported7000 状态的匹配重放训练；与此前现场 pending3640 数字相同但参数身份不同，必须同时核对模型 ID、候选目录与 SHA。选择 sampling seed42 是预先约定，不是事后选最好种子。它是可选、可回退、分批 Online 候选；不代表 Critic 全部正确或自主成功率已提升。

已验证：三个采样种子下，新 HIL TRAIN 动作六关节 MSE 相对匹配 baseline 改善 5.32–6.48%；20 条重复 DEV 的三类动作尾部与自主/辅助 Q 行为回报代理均在预设容差内。容差是本次工程筛选条件，不是机器人插入精度标准。旧失败数据动作拟合略变差，不掩盖此代价；详见 whole-Episode bootstrap 和分关节图。夹爪、左臂固定，不把夹爪预测误差称为物理退化。

失败：接管 credit 截断、Critic 失败末端 quota2 以及修正后的 quota1 均有种子违反旧 DEV 保持条件，未交付这些改动。只有最终 TD/MC 目标限幅几乎不改变结果，作为有数据前提的边界保护保留，不能宣称其已治好高 Q。

证据不足：三个新失败 Episode 的终点 Q1/Q2 仍明显大于目标0；没有独立 TEST、目标位置 OOD、自主持续增益或闭环动作优劣认证。不能用训练集拟合、Actor Q、TD loss、末段 AUC 放行自动晋升。历史 HIL 是执行过的行为动作，不是每个状态的最优反事实动作。

本次更明确的定位：新HIL人工窗在匹配baseline末状态，Q1(HIL)-Q1(Actor)=+0.1127，minQ差=+0.1046，当前并非统一的“人工动作Q低于Actor”。但新HIL拟合较7000反而略差（MSE .11729→.11814），加Actor名额后降至.11185。支持“Actor混合批次中纠正曝光不足/其他梯度竞争”的局部诊断；不能把一条HIL推广为所有状态。同一新失败终点的训练目标确为0而预测仍正，说明该处残差未拟合消除，不能一概归因其目标被标成成功。见图5和comparison.json；同状态Actor是各checkpoint重算提议，不冒充当时介入前已执行动作。

## 身份、训练与实际比例

- 基线代码 `dc548958a53f19ac3941cb21692f3470c4b61b09`；固定 upstream 不改。新代码与 Git 发布版本见同目录 delivery.json / source-release.json。
- 数据：2496 窗原 TRAIN 加实际新7 Episode 的281窗，共2777窗/208Episode。顺序10000、10001、10002、10004、10005、10009、10010。唯一新人工介入 Episode10009，共18窗、7窗含human、53个去重人工控制步。独立 TEST 空缺。
- 外部20Episode/370缓存窗是反复用于选型的 DEV，其中5自主成功、6辅助成功、9失败。Stage1历史134全部训练的14条不能重命名成测试。20DEV也不是新的盲测。
- 起点、参数/Adam/target/RNG固定7000；seed41/42/43；各281Critic更新、140Actor更新。ActorLR1e-5、CriticLR1e-4、MC0.3、BC5、Q0.1、teacher50、delta10、batch128、Actor周期2、gamma.99、C10、std.002。Reference dropout .5。没有重训 Stage1/Warmup。
- 基础批次仍近期51/warmup38/human26/uniform13，窗口有放回；专家/human/成功标签可重叠。近期按max Episode ID减19，不是最近20个成功轮次。
- 新方案只替换 Actor 随机8槽为近期真实 Online human/mixed窗；Critic不加此配额。无合格HIL时不替换，不伪造或删标签。HIL8是8/128=6.25%的替换槽，最终人工占比还包括基础采样，也不是8条独立示范。
- seed42实际离线重放：Episode10009入库后首18更新的Actor人工窗占比由2.08%到8.33%；随后10010阶段由约0.83%到约7%。详见analysis.json精确计数；这是实际离线抽样，不冒充旧现场历史批次。历史现场同段为1.13%，抽样随机状态不同。
- Replay存量、每批抽样和Episode比例是三件事；不固定成功:失败=8:2。采用独立journal、两个采样RNG、全量身份回执，恢复采样轨迹可复验。

## 对截图与此前问题逐项处理

| 项目 | 结论 | 本次处理/证据 |
|---|---|---|
| TD自举外推动作 | 证据不足 | 原target Actor仍在；MC30和限幅不保证排序正确，未换成无依据最优动作 |
| Q量级 | 已验证 | 当前完整Episode只有一次0/1终奖，gamma=.99，观测回报在[0,1]；凸混合目标再限幅。不是因为任何任务Q>1都错误 |
| 接管credit | 失败 | target-only censor单因素违背DEV保持门槛，不把接管改写成环境done，不采用 |
| 新纠正曝光不足 | 已验证 | Actor HIL8真实索引与离线281批全部相同，3种子HIL拟合改善；不是因果自主收益 |
| 失败终点曝光 | 失败 | quota2/1副作用，未采用；Q偏高保留为开放问题 |
| 关节2偏差 | 证据不足 | DEV辅助轨迹关节2误差最大；分维度图保留，没有常量补偿，均值相关不证明根因 |
| 探索不足 | 证据不足 | std.002和Reference种子保持；无离线证据支持在接触区增加噪声 |
| HIL标签 | 已验证 | source/HIL/outcome原样，自主和辅助分开；删除标签不会消除人工救回路径 |
| Replay时间对应 | 已验证/有边界 | 沿用完整RTC输入身份、实际动作与FP32审计；不恢复无历史trace的反事实 |
| Stage1泛化 | 证据不足 | 沿用历史完整链审计；本次不制造未见数据或把重建loss当Critic表征证明 |
| 执行抖动与超时 | 证据不足 | 保留50HzRTCEMA及10月8日修复；本批不启动模型，不新增真机长期稳定结论 |
| 恢复与导出 | 已验证 | 原生CPU8步 vs 独立进程4+4，149状态叶、双RNG、8批身份逐位相同；32缓存输入原生Actor最大差1.19e-7；预算用完不额外更新 |

## 方法、结果与可复现产物

独立 worktree：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/online-credit-repair-20261010/rl-platform`。输出根：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/online-credit-repair-20261010`。

预注册见 EXPERIMENT_PLAN.json / FOLLOWUP_PLAN.json；原始15臂、组合9臂及修正6臂分别在 single-factor、followup、terminal1-corrected。followup中标为critic_terminal1的3臂曾误启用HIL8，已保留失败记录并明确排除单因素结论，重跑真实quota1；见followup-correction.json。三seed baseline与先前研究149状态叶逐位复现，见baseline-reproduction.json。

A6000 CPU172–175、CUDA_VISIBLE_DEVICES为空、JAX_PLATFORMS=cpu。原生命令/启动参数见launch*.json与原始日志；脚本`scripts/offline_credit_repair.py`、`prepare_hil8_candidate.py`、`validate_hil8_worker.py`、`report_credit_repair.py`入Git。运行环境在scratch/rl-platform/runtime-restore-verification/envs/online。5图重生成命令：`python3 scripts/report_credit_repair.py outputs/online-credit-repair-20261010`（系统matplotlib3.1.2；训练环境不安装新包）。

图1为各单因素和组合，图2为20DEV分关节及Episode bootstrap，图3明确显示未解决的失败终点Q，图4显示真实离线Actor/Critic抽样构成，图5区分Actor的Q1与TD的minQ配对偏好。bootstrap以完整Episode为单位；仅1新HILEpisode不报置信区间，三个seed范围不是泛化置信区间。图中早期checkpoint在未来才入库的Episode上仅作固定诊断探针，不参与当时训练或作为独立测试。

## 可选交付与操作边界

候选A6000路径：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010`。

拟现场两入口：`plug-v3-hil8-7281`（冻结）与 `plug-v3-hil8-online`（分批Online）。现场仅在安装回执验证后可称可选项已安装；文件存在不代表模型已加载。原7k/原5k模型与默认选择保持。现有生产Replay和pendingActor不覆盖。新runtime从相同完整候选独立fork，旧数据不会二次领取预算。

现场负责人操作单：
1. 设备由现场人员确认正常；采集页面选择“插孔 · HIL8（分批 Online）”。核对模型ID、初始Learner7281/Actor3640和候选目录，固定50Hz＋RTC＋滤波。加载/开始Session由现场人员执行；本次交付不自动加载、运动。
2. 第一轮完成标注、录制flush及完整Replay准入，人工介入保持HIL，成功分别记自主/辅助。中止/录制不完整不当任务失败或成功。
3. 审查新增N窗对应N Critic更新，Actor每2次一次；看actor_critic_batches.jsonl的实际双批身份与新增HIL占比，不能只看Replay存量。每50次更新保存step checkpoint，正常收尾另保存latest，停机与恢复预算不重复。
4. 继续少量、有明确目标位置标记的训练轮次，训练输出写pending_actor。禁止根据上升Q或降低loss自动晋升；先审查有限值/双头终点残差/旧DEV保持/新HIL方向，然后小批比较候选与固定旧Actor。现有DEV只作回归开发集，另保留新位置与独立完整Episode用于最终自主收益判断。
5. 录制/trace不完整、身份或预算不一致、非有限值、持续物理异常、旧DEV超过上述容差，停止该批学习/拒绝发布；保留候选及原因，使用旧模型入口回退。单次失败不等于软件故障，低TD loss也不等于通过。

正常持续Online自主收益目前仍“证据不足”；本交付达到的是离线软件一致性和受控分批学习候选条件。独立真机结果只能由真实闭环轮次补证。

## 最终安装回执

已验证：源码 `6e828bb96a866cefb108f8e33f0fc58c4fab1d05` 已先A6000提交push，再在Cobot模型操作锁内同步9个运行文件。Cobot真实Python3.10 CPU22专项通过；独立副本8更新、32输入原生Actor一致性1.19e-7、预算停止通过。A6000专项共38独立用例通过；原生8 vs 4+4逐位恢复通过。第一次现场测试因pytest向上发现旧项目root而导入旧代码，显式指定隔离root后通过；首次安装因未等网页30秒目录刷新自动回退源码，复验原文件与候选SHA后完成安装。失败尝试及回退均保留，不隐去。

网页只读API已显示两个新增入口available=true，Online项training_enabled=true、冻结项false。Cobot静态候选目录 `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_hil8_clip_20261010`；独立Online目录 `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/supported_hil8_clip_runtime_20261010`。初始Replay2777、Learner7281、Actor3640，新预算0，旧数据不重复训练。模型入口已安装，不等于现场已加载或已获得新成功率。

7项旧Replay/权重/配置/profile SHA保持，已有服务PID/start_ticks保持；未启停服务、未加载现场Stage1、无GPU或机器人动作。CPU测试只在隔离副本构造本地Learner/Actor对象，不开RPC。完整回执在输出根 `field-install.json`、`field-checks.json`、`native-validation/verification.json`。不可变profile中的native_resume=pending是打包时快照，以上后续实际验证回执为准，不为改标签重写checkpoint契约。

可进入第一轮受控分批Online；自主持续增益仍证据不足。新方案与旧pending3640的数字版本相同，因此归档、比较、恢复必须带候选ID和SHA。按轮晋升需外部检查，默认不会把pending_actor覆盖到served Actor。保存策略是每50更新加正常flush；突然断电未落盘的更新允许回退到最后完整checkpoint，其日志不能算仍保留的学习。
