# RTC纠正监督与连续HIL成功回报：合同修复实验

## 结论与边界

2026-09-18用户要求继续验证/修复warmup没有明显真机改进的问题。本轮补充了独立因果RTC BC数据构建和保守的真实接管回报连接，但候选未通过纠正能力门槛。不能把它称为已修好的生产warmup/在线RL流程；不继续以更新次数替代策略改善证据。当前生产20000 actor、默认learning/replay/runtime/robot代码、原始数据及v2缓存均未替换。未执行机器人动作、HPC访问、commit/push。

Cobot执行根 `/media/agilex/Getea1/jiaan/projects/rlt`；主证据目录 `runs/plug_v2/learning/rtc-contract-validation-20260918-v3`。completion-receipt.json确认四个实验进程均已退出，Session waiting_scene/policy_paused、actor20000、fault=null，生产SHA `c6395e9d44139326625cdc2c894fe56b5d8c0196fcb4d722267a0eb24715a9b3`未变；模型预加载保留，后续动态状态仍须核验。

## 因果监督合同

新增执行模块 `methods/openpi_rlt/plug_v2/rtc_supervision.py`，使用真实RTCQueue和CommandFilter：先cold prime，然后同一tick以d6重规划，再每10tick消费队列/请求。prefix来自此前reference推理的已承诺命令，未来teacher动作仅用于post-prefix标签，绝不进入RPC观测或prefix；序列按valid/generation和真实传感器时间戳切段。离线仍使用录制的teacher图像/state，而非这些预测命令的真实物理后果，因此明确是**BC-only因果增广，不是实际执行过的RTC专家transition**，不能赋予factual RL奖励。

严格切段生成1393行（expert898、HIL495）。原始预测prefix相对录制实测state最大右关节差<=0.04rad+数值容差的支持检查留下196行，train172、heldout24；eligible原UUID train专家74/HIL44，heldout专家8/HIL11，零交集。约85.9%行不通过该检查反映预测prefix与录制教师轨迹不自洽，不能当作实际机器人tracking failure统计，也不能靠填入未来教师prefix伪造一致性。该支持检查保守，未声称穷尽可用增广。

50槽队列prefix只有前6槽已承诺；早期诊断上下文误打包715维，导致无eligible样本、训练前失败。修复为99维，并新增shape校验/测试。RPC原先已正确只发送6槽，因此v3复用正确ref/z，机械修复序列化，不重复GPU提取。v1/v2失败/中断证据保留，不覆写。

## 成功回报连接

新增 `handover_credit.py`：仅认可相邻且有效的policy->right HIL；generation必须+1、控制tick一致、时间间隔0<dt<=50ms、teacher观测新鲜且同一新generation。普通bootstrap边及整个窗口也逐帧检查非递增/超50ms间隔，暂停、缺帧、复位及未知模式转换不桥接。

现有v2非专家正常bootstrap有85条边跨越时间戳间隔（teacher76/policy9）；加上窗口连续性检查，严格派生连接拒绝125条normal边/窗口，另1处teacher观测间隔不合格。该检查不修改原始事实或v2数组。32处真实直接接管边（train26/heldout6）通过；可到成功奖励的policy行为145（train118/heldout27），原合同为0。此处是控制窗口行数，**不是episode数/成功率**。奖励保留原位；接管discount跨度按实际时间计算，derived targets独立保存。

连接只能定义**已记录辅助控制器下的observed return**，不是自主actor Q。依赖HIL的成功不能直接给自主动作赋高分。因此未把该回报用于生产critic或actor Q优化；自主改善仍需同条件的无HIL终局证据。

## BC诊断与基线

独立5000 actor BC updates，50%专家/50%HIL组内row抽样，LR1e-4/batch128/seed42；无Q、reference anchor或额外平滑loss，执行仍是原0.2rad/s/30Hz滤波，raw residual仍0.05rad。500/2000/5000保存diagnostic_only，生产load_actor拒绝。训练20.39秒。减速beta仅用train从0/.125/.25/.5/.75/1选择，最优.75；heldout不用来选择beta。

| heldout组 | reference MSE | 5000 BC MSE | train选定减速MSE | 保持state MSE |
| --- | --- | --- | --- | --- |
| 专家（12行） | .00148601 | .00145266 | .00141821 | .00331523 |
| HIL（12行） | .00191016 | .00130781 | .00132333 | .00076832 |

专家仅改善2.24%、仍弱于减速；HIL改善31.53%、仍弱于保持。teacher确实运动（目标相对state平均绝对差>=.01rad）的子集，专家BC仅比减速好约1.74%，HIL仍弱于保持；不能宣称学会插孔纠正。被动臂/夹爪输出偏差0、最大执行步长<=.2/30+容差。所有候选均未通过预设分组BC基线门槛，未发布。

## 执行可达性与梯度

heldout因果监督在0.05残差+固定prefix+原限速下，专家41.39%/HIL39.94%监督坐标超出严格逐坐标可达包络。MSE下界分别.00121777/.00096306。保持限速，仅把raw residual半径增到.2：下界变.00114569/.00014801。专家主要受当前prefix/速度/时间窗约束；HIL残差幅度影响更大。这是上述反事实BC样本的条件可达性，不是实际真机episode的可达性结论。

reference处hard velocity clamp导致raw-to-executed梯度为0的有效坐标：专家33.75%、HIL24.71%、真实policy heldout99行29.51%。不是全部动作被锁死；不能单独据此证明无学习能力。

额外三项各2000更新（约8秒/项），仅改变raw residual半径与训练backward估计，前向仍保持原滤波/限速；straight-through是有偏梯度估计，不当作精确梯度或论文原配置：

| 半径 / clamp backward | 专家MSE | HIL MSE |
| --- | --- | --- |
| .05 / surrogate | .00148537 | .00122289 |
| .2 / exact | .00145654 | .00118177 |
| .2 / surrogate | .00158709 | .00088701 |

扩大半径+surrogate虽降低HIL误差，仍不如保持，专家还变差；不能直接放宽部署参数解决。前向一致性测试误差<1e-6，生产Actor类仍固定.05，所有消融artifact均diagnostic_only，.2模型显式标记不可按生产Actor加载。

## 原始动作与执行条件核查

另用完整cold teacher heldout（专家439/HIL546行，不混用上述24行）审计：当前训练把滤波后的预测与未滤波教师绝对关节目标直接比较。专家raw reference对raw teacher MSE .00018099，filtered reference对raw teacher .00029822；同时滤波两者后.00005433。HIL对应.00027104/.00022780/.00010559。此处数字描述条件不同，二者同为rad绝对SDK目标，并非单位错误；filtered预测拟合实际教师目标本身有意义，但须审查可达性/梯度。不能把变小的变换后标签误差称真实效果改善。

教师原始目标连续变化速度超过.2rad/s的坐标比例仅专家8.20%/HIL4.63%，不能把85.9%context排除归因于多数教师运动过快。已确认字段合同为14D absolute front joint targets(rad)、grippers(m)；下一轮须分别定义原始命令BC监督、controller-compatible派生监督和实际发送命令的critic transition，禁止把派生滤波/投影label当作实际记录的奖励轨迹。

## 证据、检查与下一步

- v1：额外clock审计后核验PID身份停止；operation interrupted，原产物保留。
- v2：13:24:06 UTC/PID1793473，完整特征提取后序列化校验失败；未训练/发布。
- v3：13:29:14 UTC/PID1868310，reuse_and_train.py，13:29:43完成；13:32:46 PID1872318启动ablate_bounds.py，已核验completed/进程退出。
- 10项合同测试：prefix因果性、cold/d6真实队列、99维、短段、连续HIL奖励、无缺帧/invalid/left-only/未知generation/暂停/teacher窗口gap连接；contract-tests.log全通过。
- context-repair.json、credit-audit.json、bc-validation.json、postcheck.json、execution-bounds.json、bounds-ablation.json、action-space-audit.json、completion-receipt.json、source-hashes.json保存完整可复查结果。大数组/权重/日志仅在Cobot；框架仅摘要与hash。
- 下一步先对齐BC的原始命令/执行命令监督，并补足部署d6的独立、物理自洽纠正检验；门槛必须同时考虑raw目标、可执行目标、保持/减速及有运动子集。再用无HIL留出rollout检验目标识别与真实纠偏，过关后才重新warmup/启用Q优化。当前不存在已验证可替换生产的改进actor。

收尾：框架git diff --check、索引重建通过；workspace doctor正常退出，0 error、7条既有legacy attachment warning。本轮只新增/更新RLT项目审计、STATUS、CONTEXT、DECISIONS与execution_sources；其他项目未提交修改保留，无commit/push。
