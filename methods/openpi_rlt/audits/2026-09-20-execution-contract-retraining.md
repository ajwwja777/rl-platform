# 2026-09-20 执行动作定义与重训对照

## 范围

用户要求修正到可以开始在线RL。当前仍未放行；所有新模块为隔离实验，不导入生产runtime、不切换current.json、不执行真机动作。机器人Session检查为stopped。未访问HPC，Cobot已有模型保留。

## 已完成的对照

1. 重新计算486个有限速的因果BC前缀窗口（82个专家文件）：只用当前状态及过去命令构造hold/past_velocity前缀，调用同一Stage1重算z/ref；未来动作只作BC标签。明确标记非事实TD，未写入RL replay。
2. 原仓库BC+delta、Q权重0，保持原网络/LR/dropout，冻结critic：1000步后执行误差未优于起点，拒绝发布。路径 learning/causal-prefix-bc-r2-20260920；无r2的第一次尝试在恢复状态时失败，没有训练完成。
3. 独立执行误差BC实验（非原论文损失）：10×归一化执行MSE+1×原始BC，确定性actor、critic冻结，1000步。100步时因果前缀留出执行MSE .00113377 -> .00108376，改善约4.4%；1000步 .00109543。变换与部署最大误差3.73e-9rad。不能据此声明闭环纠偏，未发布。
4. 发现自主replay动作是滤波前proposal，human动作是实际命令。404个完整自主窗口，raw与实际下发命令平均绝对差.02716rad，窗口均值p95 .04240rad。存在动作定义不同的混杂，不能单凭该差异证明是失败唯一根因。
5. 新 executed_command_data/core/train 三模块：469条policy行统一为可核验conditioned commands；human及z、proprio、reward、done、duration、next_*保持逐值相等。actor的Q分支、target actor的Q分支通过与部署相同的投影和滤波，再进入critic；BC、delta、网络、优化器、target更新周期仍用原仓库。628窗口数值对齐最大误差1.19e-7rad，梯度有限；已删除11个UUID未恢复。
6. 新定义下从随机初始化重训5000步完成；另20k同配置对照也已完成。5k同滤波目标d6 MSE2.94484e-5；实际human MSE仍约.00208539。Q(受限human)>Q(受限reference)为55.79%/561复用窗口，早期Q仍负，未达到可靠动作价值证据。

## 奖励图检查

按已存next_frame/next_proprio/next_z逐值寻找TD后继，不跨暂停或缺失数据虚构连接：训练44条HIL成功episode中，245个policy行只有108个能沿图到达正奖励，137个因censored edge断开；21条episode有可达路径。留出11条成功中65个policy行有43个可达、22个断开。并非完全没有正奖励传播，但不能把这些HIL成功Q当自主成功概率。

## 产物

Cobot /media/agilex/Getea1/jiaan/projects/rlt：
- runs/plug_v2/diagnostics/rtc-clean-20260920/causal-bc、build_causal_bc.py、causal_bc_train.py、execution_bc_train.py、bc-failure-counterfactual.json、critic-action-contract.json。
- methods/openpi_rlt/plug_v2/executed_command_{data,core,train}.py：独立实验，生产未接入。
- runs/plug_v2/diagnostics/executed-command-20260920/：数据、test_contract.py、contract-test.json、q_probe.py、q-command-probe.json、candidate-comparison.json、reward_graph.py、reward-graph.json。
- runs/plug_v2/learning/executed-command-20260920/ 与 executed-command-20k-20260920/：独立训练产物。

实验不同于原论文完整复现，不把适配说成论文原配置。禁止仅根据loss、同滤波目标MSE或复用HIL留出自动发布。20k及后续结果见下。

## 最终离线对照（均已完成，未发布）

| 对照 | global steps | Q(受限human)>Q(受限ref)，561复用窗口 | 实际human动作误差d6，未作投影的actor经滤波 | human batch比例 |
|---|---:|---:|---:|---:|
| 执行命令联合 | 5000 | 55.79% | .00208729 | 92.05% |
| 执行命令联合 | 20000 | 40.11% | .00208230 | 92.32% |
| BC2000 / critic-only6000 / joint2000 | 10000 | 48.84% | .00209532 | 91.89%（actor更新阶段） |
| 互斥human26 + policy102 / batch128 | 5000 | 60.25% | .00211115 | 20.3125% |

表中动作误差字段为training evaluator的executed_human_mse（未投影actor经滤波），不得与candidate audit的projected actual_human_mse混称。部署matched MSE与加速度字段包含线性投影。需始终查明指标口径。原仓库分层池重叠，20%专用human采样并不保证最终只有20%human。quota仅作为明确标记的非原仓库采样对照，无证据认为硬20%更适合生产。

延长训练、分阶段、固定human20%均没有建立足够可靠的动作价值依据；20k早期Q还出现失败高于HIL成功，不发布。release-decision.json记录not_ready_for_online_rl；这是离线结论文件，不是已安装的生产硬拦截。current.json保持不变，新实验不可能被正常入口自动选中。completion-receipt.json保存源码/权重哈希，核验四组训练completed、Session stopped、生产指针未变。

## Stage1视觉核验

固定最近失败首帧的state、seed42、d0，对8个专家留出首帧逐一替换顶部/左/右图像及全部图像，只有RPC推理，无机器人publisher。替换顶部、左、右图像造成首10条右关节reference平均绝对变化分别约.000180/.002315/.002060rad，不能说模型不看左右相机；混合不同场景图像只是敏感性测试，不是成功证据。

最近失败首帧与最近专家起点右关节最大差约.000384rad、夹爪值一致；腕部画面和顶部构图有可见变化。已通过异步问题询问用户：录制专家后是否调整中臂camera位姿、相机安装或插头夹持深度/朝向。尚无答复，不能据此认定物理设置变化是根因，更不能擅自移动机械臂/相机。

新增本地图08-execution-contract-training.png、09-camera-observation-review.png，位于既有20260920-training-review目录，已检查。中间新增数值变换曾在测试发现delta_chunk枚举不匹配，已于训练前修复并通过628窗口对齐；生产从未使用该实验代码。

下一步需先确认/排除场景和夹持几何差异，再决定Stage1重训或取得受控新闭环证据。当前未达到用户要求的可启动在线RL交付状态；不得声称已完成目标，或让用户直接开启旧在线学习入口。
