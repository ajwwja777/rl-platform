# 2026-10-12：动作一致性、训练期 RTC 与探索核对

本次用户要求重新从输出/训练动作一致性设计平顺执行，不以既有EMA为前提。代码基线84a0c463f0f49853bd051e53102131dccf459ce0；固定上游8cef77eb7c5211b45382bf9199c9cdf0aaf60a19。仅A6000 CPU静态审计/历史数值统计/原文代码调研，无新训练、现场连接/修改、GPU、模型加载、服务或机器人操作。之前候选安装与运行状态属于10月11日证据，本次不重新认证当前运行。

## 已验证：当前动作链并非严格逐发布一致

async_execution.py:525插值，526滤波，随后安全处理并发布；626/641每逻辑步action取最后发布目标；原生inference.py:1327将这些逻辑步堆成C10；trainer.py:198-210直接将Actor输出送Critic。guarded_retained_actor只增加保持损失，没有将执行滤波引入Actor Q路径；held_gripper另有固定维投影，不能解决六关节时序问题。

对已有29完整Episode数字trace（重复选型DEV）重新统计：5327个有planned_action和发布记录的逻辑步、13291次物理发布；5327/5327的action等于本步最后发布命令；7964条中间命令保留在trace中，但未逐条成为训练action序列。六关节planned_action与最后命令的每Episode RMS中位1.700511mrad；planned_action在探索生效时已包含探索，非纯Actor提议。只统计策略发布行，不据此评判HIL反馈动作。

结论：之前strict一致性验证只证明逻辑步action/训练数组一致，不能推广为完整50Hz轨迹一致。20Hz动作抽象不自动等于算法错误，也不能据此确定Q异常根因；但它未满足用户要求的最终输出轨迹和学习动作同一语义/时基。不能简单将原提议替换进Replay而让已发生的状态转移仍对应不同命令，也不能直接将C10从20Hz解释为50Hz（0.5秒会变成0.2秒）。

物理反馈位置与发布目标需分开，action一致性指命令/策略接口一致，不要求电机无跟踪误差。Actor不必输出与离线样本一模一样的动作；要求同一动作定义、归一化和时间跨度，以及训练Q/TD候选遵守同一执行规则。

## 已验证：原始 RLT 与社区复现不能混同

RLT v2方法IV/V、实验VI、附录B/C已读。原论文基于pi0.6，VLA H50对应1秒，RL C10在50Hz下对应0.2秒；Actor直接预测动作chunk，BC向Reference靠近，Replay记录执行动作；人接管时Replay的Reference改成人类动作。论文没有给出足以复现其底层EMA/jerk/RTC调度的细节，不能声称原RLT用了某种特定滤波或绝对没用滤波。

固定openpi-RLT是Yyshadow社区pi0.5复现。其pika_sync_ros.py:1119-1125把动作限步长后send_action并记录同一bounded值；1401-1409是步长裁剪，不是两级EMA。trainer.py:245-247的delta_penalty拟合相邻动作增量与BC目标增量之差，不是零速度/jerk硬约束，也不提供跨chunk前缀条件。不能把我们的20Hz逻辑+50Hz插值当成原论文50Hz C10原样复现。

来源：https://arxiv.org/html/2604.23073v2 ，固定上游源码见上述路径。

## 已验证：Training-time RTC / Xiaomi / Real-Time EXPO-FT

Training-Time RTC（2512.05964）方法IV和实验V：训练时随机模拟延迟d，前d步动作作为干净前缀，后续动作加噪，只在后缀计算flow loss；推理时承接已承诺前缀。目的是让模型生成连续后缀、减少推理期引导成本，不是对输出再做平均。文中低延迟并非所有设置都更优，也没有接触抖动绝对保证。

Xiaomi Robotics-1（2607.15330）真机实验明确沿用XR-0异步训练方案。读XR-0方法2.2.2及实验消融：前缀条件、位置编码区分、限制前缀注意力、误差加权，强调避免模型只复制前缀而不响应视觉纠错。XR-1当前公开源码0dd7aef8dc87296246aae812a1f59ccb708e5546进一步核对：XR1.py:342有50%概率抽1–6步前缀；287-295随机遮掉较早前缀注意力并保留最后两步，382-383后缀RoPE偏移，414-422后缀loss与误差权重。不能把XR-0论文Lambda mask和XR-1当前随机前缀遮罩称作完全同一代码。XR-1是流模型，当前RLT Actor是MLP，不能只打开配置就获得同等训练行为。旧Cobot XR-1还有本地EMA；现场定性顺滑不能归因全部于训练RTC。

EXPO-FT v2（2605.25477）与后续Real-Time EXPO-FT（2609.18207）分别读取，不把后续能力归给旧论文。后者方法IV/实验V/附录和官方代码21fe3d3b7d913c80836496817965932f49e0aedf核对：先RTC-SFT，昂贵VLA提前生成，轻量edit使用执行时最新观测修正/选择；训练backup同样取延迟观测、已执行前缀及对应后缀窗口。realtime_expo_ft.py:793-856明确delay切片和前缀；rtc.py训练入口转train_step_p1_prefix。其filter指候选/噪声筛选，不是EMA低通。它更新VLA/使用额外视觉Critic等，不能原样接到冻结Stage1的RLT上，也不能假设已有RLT Q足以选动作。

来源：
- https://arxiv.org/html/2512.05964v1
- https://arxiv.org/html/2607.15330v1
- https://arxiv.org/html/2602.12684v1
- https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/mibot/models/VLA/XR1.py
- https://arxiv.org/html/2605.25477v2
- https://arxiv.org/html/2609.18207v1
- https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/alg/realtime_expo_ft.py

## 探索截图：已验证机制，因果归因证据不足

当前exploration.py每次apply抽六关节偏移，rho=.5；async_execution.py:203每次规划调用一次。RTC配置每5个20Hz逻辑步重规划，即名义0.25秒。高频改偏移确实是可消除的外加不连续来源。截图35–45%/J2约12%仅为另一分析的相关性主张，本次未获取其完整拟合脚本/统计收据，不升级为因果解释度。

每Episode固定偏移+平滑引入值得作为独立候选；它只能消除本轮反复重采样偏移，不能保证总动作高频分量为零。暂停/HIL重新接入、限幅饱和、RTC前缀与新偏移重复相加均须处理。仅J2探索是定向假设，不代表其他关节不重要；一轮成功也不能证明该偏移因果有效。先在无探索基线与只改变时间结构的固定偏移间做单因素比较，再另比六关节/J2；不同时修改频率、幅度和维度。无新增采集或探索部署。

## 下一候选设计取向（待实现/验证，不是已发布方案）

优先级转为：动作定义/时基一致 → 低频有界探索 → 前缀条件生成/训练与延迟对齐；两级EMA/Ruckig不作为本轮主线，旧EMA只保留回退基线。安全保护不直接取消。

最终策略输出须定义为可执行命令序列；发送、Replay及Critic训练使用同一序列和时间索引，Actor-Q与TD计算也须遵守同一前缀/延迟规则。若保留确定性执行解码器，它必须成为显式策略组成并在训练中复现，含其起始状态；不能继续训练原提议却只用最后滤波目标解释转移。若将RL动作时基统一到50Hz，需要重审chunk持续时间、reward/discount、归一化、Actor/Critic维度及checkpoint兼容性；不静默把C10切成50Hz或把旧Replay重标频率。

先在不依赖GPU的schema/时间索引/前缀固定/暂停与HIL/归一化来回/历史轨迹重建上验证。前缀机制需覆盖最终RL Actor，不只覆盖Stage1 Reference；可先做轻量prefix-aware Actor的受控离线可行性实验，再独立评估Stage1 RTC-SFT，不能假设改Stage1就必然修复Actor衔接。不改变现有训练资产或默认入口。模型训练方案与资源另列，不以本次调研认证可持续Online。

## 产物与验证边界

A6000 outputs/action-contract-rtc-20261012：audit_action_contract.py、action-contract.json、sources/download-manifest.json及固定SHA源文件和六篇原文HTML。命令：python3 outputs/action-contract-rtc-20261012/audit_action_contract.py。输入commands.json SHA f43140f322b2140161e40c5f1d4c81d3df79295f5e473daba4ba6b27e28bc039；逐Episode原trace路径/SHA在JSON。主仓库无运行代码修改；本次统计CPU执行成功，不是训练/机器人验证。末端大幅抽插、Critic问题与执行差异之间的因果关系仍证据不足。
