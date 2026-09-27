# 2026-09-14 连续拒绝诊断与受控动作训练边界

## 已确认原因
正式线上37次候选均被拒绝，actor保持5000。隔离消融复现了第一轮EP24的原报告数值：score 0.02003392、velocity 0.04679529、acceleration 0.08174508。
单独加入EP13、14、23均通过，加入含HIL的EP16即退化。归一化统计左夹爪dim6的q01=q99约0.0004，实际HIL读数约0.0016；scale=q99-q01+1e-6，使该被动维度训练目标约2399.53。完整历史数据BC损失约16656，human BC约40839。固定左臂的少数维度也有小尺度漂移。右臂本身归一化动作在合理量级。
因此根因是当前右臂/夹爪固定执行契约与训练动作输入不一致，不是操作者HIL操作错误，也不是应取消验收门槛。

## 修复
online_cycle.project_training_action只构造learner私有动作副本：右臂关节7:13保留实际动作；左臂和两夹爪使用reference。source_chunk、奖励、terminal、proprio、特征、原始replay与原始HIL记录保持不变。原始实际动作仍是追溯证据；训练critic动作输入与BC目标不再受不可控制维度偏移主导。
worker将此契约显式记录为controlled_arm_reference_passive_v1，active_arm默认right，hold_grippers=true。学习率、BC/Q权重、32步周期、采样30/40/30以及验证门槛均未调整；未重新计算归一化统计，以保持既有模型和特征接口兼容。
修复适用于当前right/held-grippers实验；若以后开放左臂/夹爪，必须重新核验归一化和控制契约。

## 消融及连续验证
同一warmup seed、32步、全数据：
- 原逻辑score 0.03294982、velocity 0.07247838、acceleration 0.12725956，拒绝。
- 仅处理左夹爪：score 0.00972924，BC0.01394，通过。
- 全部被动维度处理：score 0.00957751，BC0.01243，human BC0.01908，通过。
按原EP顺序24..61（缺30）重放37个episode，每轮32步：26接受、11拒绝；最终actor5416/global_step10832。接受步累计832，尝试总步1184。更新验证中位1.3559秒，不含固化/归位。早期若干轮约3秒。
最终验证score0.00956593、velocity0.00828389、acceleration0.01009284、first0.01196026、boundary0.04263747。这些是冻结验证集指标，不是真机成功率；数据消融和候选选择复用了验证集，不宣称独立泛化评估。

## 产物
Cobot隔离实验：/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/scratch/online-rejection-20260914。
固定候选：runs/plug-action-contract-review-20260914/artifacts/actor.pkl 与checkpoint.pkl，review_manifest.json记录来源和SHA256。
actor SHA256 d1a84cb93629c1f481fc00a067ffb60d9ecd00fa71ad27875aedfe340dcebfbd。
checkpoint SHA256 d5eaa4f4ef542487a9c27a13a0ccdf7ff8fe480f72f43b076c5e8314ef18d6fb。
正式runs/plug-online-warmup-r1/current.json仍actor5000/global_step10000/transactions37，原始warmup与正式actor权重未替换。

## 现场验收
同一部署目录：
- 对照：./interface_task2_teach_rlt_smooth_warmup.sh （原始5000）
- 候选：./interface_task2_teach_rlt_contract_review.sh （固定5416）
两者都无参数、无探索、无在线更新、EMA0.35/command step0.01；前一Session结束且终端退出后切换。候选入口启动时核验版本和两个文件哈希，禁止传参。候选脚本从online launcher生成并固定run和freeze，后续通用launcher变更应同步审计。
继续在线更新时需明确从哪个accepted checkpoint续接；目前普通online入口仍从正式5000恢复，不能误以为已使用5416。
50项相关回归通过、Cobot3项契约测试通过、候选5416冻结加载验证成功(training_enabled=false)，bash语法检查通过；未启动机器人env/publisher或现场动作。
部署备份runs/deploy-backup-training-contract-20260914；runtime manifest已更新。未提交push。
