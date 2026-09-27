# 2026-09-19 部署滤波一致性、奖励可达性与 Q 动作方向审计

## 结论
当前没有达到稳定真机部署条件的新候选。未替换生产模型、未启动机器人动作。Cobot Session 606994c3-0a19-4125-9cb4-94891a1f133c 已核验 stopped；驻留进程仍为 r5。r6 修改在磁盘，未真机执行。旧 rtc-corrective-r1 保持 rejected_after_onsite_instability。

## 训练/部署一致性实验
Cobot 根 /media/agilex/Getea1/jiaan/projects/rlt。
- 新增离线 conditioning_v2.py 数值对齐固定候选部署滤波：速度 .10 rad/s、加速度 .9 rad/s²、Piper 关节裁剪。前次 1566 窗口最大差 5.59e-9 rad。
- runs/plug_v2/learning/conditioner-matched-20260919：CPU PID 3019290 已完成 2000 步，耗时12.91秒。独立 BC 诊断，不是在线 RL；没有发布。
- 直接以滤波后命令拟合动作，旧 inverse 仅 .01 权重辅助；它仍是旧滤波启发式，不是真正 oracle，不能据此主张可达最优。
- 相同部署滤波下，成功示教验证 MSE：reference .00139731、旧 warmup .00139568、旧 corrective .00115881、新500 .00113962、新2000 .00113756。新2000比 reference低18.6%，比旧 corrective低1.8%。
- 专家验证 MSE仅小幅改善，顺滑度无一致改善。复用的验证集不是新的盲测集。
- 历史成功 episode8 无空间越界；异常 episode9–12 上新模型仍有越界命令。仅是历史 corrective 生成的状态上的单步反事实输出，不能当成任一模型的独立闭环轨迹。
- deployment-gate.json 明确 rejected_for_onsite_release。

## 奖励传播
证据 runs/plug_v2/diagnostics/credit-propagation-20260919/{report,curves}.json、q-propagation.png、q-propagation.pdf。
- 0.994^30约83.5%；实际代码是每控制tick .99，10tick/chunk，30chunk对应 .99^300约4.9%。压缩TD链长度并不消除同一物理时长的折扣。
- 旧warmup成功标签下自主片段 train259/val66行，终局正奖励可达数均0。辅助成功的人控段并非如此。这解释了为什么加训练步数不能修复断链。
- 审核过的 contract 数据恢复部分真实紧邻接管边：train118/256、val27/65自主行可达正奖励。不能跨暂停缺口臆造连接。
- 图横轴为同来源的标准化行位置，不是真实时间；可能跨片段，不能据图声称连续时间传播。

## Q不是只看成败分离
证据 runs/plug_v2/learning/q-direction-matched-20260919/report.json。
- 在固定滤波下重复诊断：旧warmup的15episode早期策略片段（11 HIL成功、4失败）AUC .409；contract2000 1.0；contract5000 .773。验证集被反复使用，且没有独立自主成功类别。
- contract2000 对实际成功示教状态的动作扰动，偏好更接近老师动作的比例仅58.3%（544行）；专家55.1%（437行）。这些是反事实方向探针，不是实测动作成功率。
- contract2000 在失败策略状态上96.97%的行给 brake 比 reference更高Q；hold也93.94%。Q成败分离不等于它能指导正确插入，不能用AUC 1.0宣布actor可在线持续改进。
- training_contract 的接管边 next_action 仍由target actor产生；这不是对人控后续动作的纯事实SARSA评价。辅助回报与自主Q解释必须区分，尚需明确后再接入生产。

## 生产链审计
- cobot-platform/scripts/rlt_demo.sh 只调用 --corrective，固定候选、无在线更新/探索。
- training_flow.work 仍 prepare replay/v2，再调用 learning；learning.conditioned仍旧 .2rad/s且无加速度裁剪。此次离线修正尚未接入生产在线学习。
- cycle 对 rejected 批次也更新 consumed_uuids，导致失败批次自动消费、不重试。需要独立记录失败批次并定义显式重试，不能无限自动重训。
- 此时不应让用户继续积累rollout并期望上述离线修复自动生效。

## 后续验收条件
1. 同一带版本的执行滤波贯穿候选训练、离线评价、推理；旧模型保留历史配置。
2. 明确奖励图、HIL20%采样以及human/reference BC独立指标，审核正确的控制模式下bootstrap。
3. 候选通过异常回归、停机/HIL/队列测试；Q动作方向及独立episode验证，不以loss或单一AUC替代。
4. 发布前只可声明通过离线检查、待现场验收。稳定成功率提升和在线趋势仍需要固定初态分布、冻结对照、分批现场评测。
