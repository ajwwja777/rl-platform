# plug_v2 r5 canary闭环发散与 Runtime r6安全封锁

日期：2026-09-19

## 结论

`rtc-corrective-r1@2000` 的r5单轮canary现场出现明显乱走并向上运动，操作员放弃。该轮共执行93个inference chunk，actor/hash/Stage1/seed均未变化，没有online update、RTC deadline或RPC reconnect，因此不是参数热更新或deadline恢复造成。该固定候选已标记为 `rejected_after_onsite_instability`，后续Session启动会在加载候选时拒绝，禁止继续真机评测、探索和在线更新。

问题不是首段命令立即向上。r5首帧快照相对成功47的起始state最大差0.000942rad，三相机像素MAE分别约2.01/3.49/5.25（0–255）；确定性重推的首10步右臂reference最大差0.03956rad，但经过0.10rad/s、0.9rad/s²滤波后，首16步末端相对成功轨迹最大差约4.9mm，首16步Z仍略向下。发散发生在后续闭环观测—重规划中。

历史48–51的compact trace进一步证明：即使去掉corrective actor、只使用同一Stage1 reference，经过同一conditioning后四轮仍都会越过起点上方12mm；corrective会把越界提前约一个RTC chunk，但不是唯一原因。当前主要问题是Stage1闭环reference在分布外状态中逐步漂移，actor没有可靠地把它拉回成功流形。

## r5证据缺口

r5虽保存首帧，但 `--no-record` 在finalize时清空了逐步frames/plans，因此本次93-chunk canary只能还原首段，不能定位精确发散chunk。该逻辑已在r6修复：不录Task5/不进replay时仍保留小型state/reference/context/command trace，不保存视频或训练数据。

## Runtime r6

Cobot源码：

- `/media/agilex/Getea1/jiaan/projects/rlt/methods/openpi_rlt/plug_v2/rtc_queue.py`
- `/media/agilex/Getea1/jiaan/projects/rlt/methods/openpi_rlt/plug_v2/runtime.py`

新增仅对fixed corrective候选启用的fail-closed保护：

- 用部署Piper URDF参数实现无Pinocchio依赖的NumPy link6 FK。
- 每个RTC计划在入队前检查未来16条命令；30Hz执行时也检查实测右臂。
- 相对每轮起点的包络：X `[-15,+125]mm`、Y `[-15,+80]mm`、Z `[-110,+12]mm`，半径不超过160mm。
- 当前候选最多执行360条policy command，即30Hz下12秒；成功28/47分别约320/287个policy frame。
- deadline仍采用r5的首次超时即暂停；no-record compact trace继续保留。

离线回放结果：

- 成功28：325个rollout state全部通过。
- 成功47：313个rollout state全部通过。
- 异常48/49/50/51分别在约1.77/4.07/1.77/1.23秒被Z上界拦截。
- 本次r5首段的16条受限命令通过r6包络。
- `py_compile`通过；RTC/conditioning/RPC focused tests共13项通过。
- 全测试目录需按既有多环境运行；控制环境无Torch，旧aloha环境Torch1.10又不支持代码使用的二阶`torch.diff`且缺PyArrow。这些环境失败与r6改动无关，未计作r6通过证据。

证据：

- `runs/plug_v2/learning/candidates/rtc-corrective-r1/runtime-validation-r6.json`
- `runs/plug_v2/diagnostics/post-success-scene-20260919/r5-canary/r5-canary-analysis.json`
- `runs/plug_v2/diagnostics/post-success-scene-20260919/r5-canary/first-plan-fk.json`
- `runs/plug_v2/diagnostics/post-success-scene-20260919/r5-canary/workspace-guard-replay.json`
- `runs/plug_v2/diagnostics/post-success-scene-20260919/r5-canary/reference-vs-corrective-decomposition-v2.json`

## 当前状态与下一步

r6仅完成离线验证，没有由agent发布任何机器人动作。现有runtime进程由r5启动，但candidate manifest已封锁，下一次开始Session会拒绝加载该候选。r6需未来新候选通过离线门槛后随runtime重启生效。

下一步停止扩大真机样本，先处理模型本身：

1. 以成功28/47和异常48–51的闭环plan为硬留出，要求reference/corrective经过conditioning后不产生向上越界。
2. 给Stage1加入更强的闭环状态/视觉扰动验证，并补充明确成功终止信号，避免插入后继续重规划。
3. 新actor必须在异常状态上降低reference的向上分量，而不是只改善离线动作MSE/Q排序。
4. 新候选通过离线方向性、成功包络和紧凑trace验收后，才做一条受保护canary；不恢复explore/online learning。

## 2026-09-19 补充：训练/部署滤波不一致与因果结论修正

已只读核验当前Session stopped/policy_paused、93 chunks、0 deadline/0 reconnect。固定候选继续封锁。

实际训练入口 train_inverse.py 导入 learning.conditioned：0.20rad/s，无加速度限制或Piper限位；后续部署已改0.10rad/s、0.9rad/s²及Piper限位，未同步重训。成功28/47及异常48–51的固定上下文对照，动作最大差0.0266–0.0417rad，约5.1%–20.5%的右臂关节命令差超过0.01rad。这是已证实的合同不一致，但尚不能认定为发散唯一原因。

新增离线候选模块 methods/openpi_rlt/plug_v2/conditioning_v2.py，保持当前production learning.py和权重不变。1566个真实/扰动窗口（d0=104、d6=1462）对齐现有CommandFilter，最大误差5.59e-9rad、梯度finite。证据位于Cobot projects/rlt/runs/plug_v2/diagnostics/conditioning-contract-20260919/{report,parity}.json。

修正此前审计的过强表述：在corrective采集的状态/已承诺prefix上重算reference，是单步反事实比较，不能证明reference独立闭环也会失败，更不能由此排除actor的累积影响。当前能确认后续reference输出存在向上分量、actor未稳定纠正；最初发散的因果归属仍未闭合。首段FK仅为URDF link6预测，不能替代本轮实际运动录像/逐步trace。

r6安全包络只在两条成功和四条异常历史轨迹上验证，不能保证新场景安全，也不是模型效果修复。360条命令预算是新增候选限制，当前旧r5进程未加载；重新启用候选前须明确展示。被反复用于选方案的28/47/48–51应称回归集，不能再称独立留出。

下一步：在统一conditioning下重评原候选与简单减速/保持基线，核查真实RTC前缀与训练支持；如需重训，使用独立输出且禁止自动发布。新鲜现场评估仍待上述离线门槛通过。
