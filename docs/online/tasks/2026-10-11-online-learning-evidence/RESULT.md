# 原配方与 HIL8：采样修复及匹配学习结果（2026-10-12）

## 交付判断

- **已验证**：独立采集 journal 的小号 Episode 直接合并会漏出近期池和 HIL8 专用池；已在派生实验入口做命名空间映射、顺序与唯一性校验。生产 Replay 未改，不能据此断言历史生产 Online 必然发生过相同错误。
- **已验证**：同一起点、同一逐集数据到达、同预算的两配方 × 三种子训练完成；HIL8 对新纠正数据的关注更多、局部拟合改善更一致、旧 DEV 保持更好。
- **证据不足**：改善幅度很小，新增 HIL 仅 TRAIN/DEV 各一集；没有独立 TEST、位置泛化或持续自主 Online 增益证据。新训练权重不发布，仍保持现场 Actor3640。
- 抖动作为执行问题单独交付[低频探索候选](../../../execution/tasks/2026-10-11-chunk-continuity/LOW_FREQUENCY.md)，不能用训练代理指标代替真机平顺性。

## 身份、输入和边界

主项目根 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`；全部产物在其 `outputs/matched-learning-20261012/`（以下简称 O）。实现提交 `fab2cc0`（导入/训练）、`0b9dd4a`（诊断/低频探索）。固定上游未改；隔离分支 `experiment/matched-learning-20261012`，工作区 `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/matched-learning-20261012/rl-platform`。

| 资产 | 身份/用途 |
|---|---|
| 原5k完整state | `models/rlt/plug_v3_yyshadow/warmup-5000/checkpoints/step_5000.pkl`；SHA be50b6cf0586864d89173a6456128bffa8b37621a6fda500f5be287223050f33；Critic5000/Actor2500 |
| 共同旧训练池 | `models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010/replay/replay_journal.pkl`；2777条/208集；SHA029d09050303a6d87b0f40b52ad69ced36c9caf20c9d189e103f52ea53ec2ceb |
| 新探索快照 | O/exploration-29ep-1068.pkl；29集1068条；SHA919ff1adb31b6fcae0f3b3380ca80dc2e881b7f2501566d7042237674d0dcad8；11自主成功/2辅助成功/16失败 |
| 新TRAIN | 固定23集846条：9自主成功/1辅助成功/13失败；顺序按原采集先后逐集加入 |
| 新DEV | 固定6集222条：2自主成功/1辅助成功/3失败；native ID2/5/15/17/23/25；历史已分析，非独立TEST |
| 旧DEV | `outputs/rlt/plug_v3_yyshadow/history/warmup_20260924_v1/holdout/replay/replay_journal.pkl`；20集，6集含HIL；反复用于选型 |

完整路径/SHA在 O/asset-identities.json、run-v1/study.json；拆分在 comparison-manifest.json。按 phase/Episode 身份检查旧DEV不在原5k训练journal及本次旧训练池；新DEV实际采样次数为0。并非仅检查manifest声明。原5k/HIL8归一化数值一致。两组初始化参数、target、优化器状态相同、种子配对、step0动作相同；实际学习率按配方不同。用户中途移动插孔但轮次边界未知，不把分组差别解释成位置泛化。

## 采样错误与修复

新源native ID2–39，旧池Online ID到10010；近期采样按max ID和10000门槛识别。直接拼接的复现实验：新数据近期候选0、HIL额外池0。`journal_import.remap_collection`按manifest映射到10011起的唯一递增实验ID，保留原namespace/native ID、数组内容与采集顺序，检查每源及合并后 phase/Episode/step 唯一、总数和TRAIN/DEV完整性。映射后新近期候选565条、HIL额外候选10条（不是抽样次数）。见 O/preflight-v2/namespace-regression.json。

这修复的是独立 journal 导入风险。原生在线延续已有ID floor保护；生产采样器与journal未修改。后续导入独立采集必须走该入口，不能裸拼pickle。该映射不是用编号伪造数据年龄，映射顺序来自固定源时间顺序。

下表来自真实 sampler 记录的batch索引，3seed平均，统计实际发生的423次Actor更新；与最终buffer静态比例不同。专家/人工窗口标签有重叠，因此分层展示。

| Actor batch互斥来源 | 原配方 | HIL8 |
|---|---:|---:|
| 旧专家 | 37.249% | 33.527% |
| 旧rollout | 49.485% | 36.855% |
| 新自主成功Episode | 7.560% | 16.700% |
| 新辅助成功Episode | 0.092% | 1.322% |
| 新失败Episode | 5.614% | 11.597% |
| DEV | 0 | 0 |
| 新数据合计 | 13.265% | 29.619% |

Critic新数据13.264% / 30.014%。其中真正新人工动作窗口占Actor batch约0.048% / 1.196%，与整条辅助成功Episode占比不同；唯一TRAIN HIL较晚到达，不能用最终池比例替代全程平均。最终buffer新数据846/3623=23.35%，原配方平均只有13.3%也包含逐集到达效应。来源 O/run-v1/sampling.json、sampling-composition.json、*_batches.npz。

![实际采样比例](../../../../outputs/matched-learning-20261012/run-v1/sampling.png)

## 实际训练与结果

CPU172–175、JAX CPU、无GPU。两组均原5k完整state起点，seed41/42/43；每到达一个新TRAIN transition分配一次Critic更新，共846/Actor423；batch128/C10/BC5/Q0.1/delta10。原配方调用原生TD与uniform ReplayBuffer、Actor LR1e-4；HIL8调用真实SupportedSampler（51recent/38warmup/26human/13all，有放回，Actor另8真实HIL）、MC0.3/目标clip[0,1]/teacher50/LR1e-5/固定夹爪Critic输入。

这是**整套配方比较**，不是单独采样消融；原历史UTD5没有在固定预算中复现。不得把结果归因到单一采样、MC或LR。

| 六活动关节MAE，mrad，Episode等权/3seed平均 | step0 | 原配方846更新 | HIL8 846更新 |
|---|---:|---:|---:|
| 新TRAIN人工动作（1集） | 1.69431 | 1.68616（−0.48%） | 1.65551（−2.29%） |
| 新DEV人工动作（1集） | 2.17184 | 2.15315（−0.86%） | 2.14067（−1.44%） |
| 旧DEV人工动作（6集） | 2.02734 | 2.07701（+2.45%） | 2.02708（基本不变） |
| 旧DEV对Reference拟合（20集） | 0.76074 | 0.79676（+4.74%） | 0.75242（−1.09%） |

HIL8新TRAIN/DEV三个种子都改善，原配方有种子变差。旧DEV人工MAE变化按完整Episode bootstrap：原配方+0.04968mrad，95%区间[0.02614,0.06916]；HIL8−0.00026，区间[−0.00699,0.00671]。新HIL只有各一集，不给虚假的跨Episode置信区间。夹爪/左臂固定，不用预测误差声称物理退化。

![更新曲线](../../../../outputs/matched-learning-20261012/run-v1/learning-curves.png)
![分关节纠正误差](../../../../outputs/matched-learning-20261012/run-v1/hil-joint-errors.png)

### Critic究竟学到什么

同状态、纯人工chunk，所有动作候选固定同一物理夹爪后，比较Q1(HIL)−Q1(Actor)。TRAIN集均值初始−0.02627，原配方三个seed仍负（−0.01472/−0.02966/−0.02540）；HIL8转正（+0.01511/+0.00853/+0.01254）。DEV初始−0.06417，原配方仍负（−0.03878/−0.04278/−0.04149）；HIL8转正（+0.01127/+0.00237/+0.01079）。min-Q也检查，但不与Actor使用的Q1混淆；正值是Episode均值，非每个窗口。

![同状态动作偏好](../../../../outputs/matched-learning-20261012/run-v1/paired-q.png)

**已验证**：此配方在这两集的动作相对偏好和局部模仿方向改善。**证据不足**：重构Actor动作来自待评checkpoint，并非当时接管被取消的原提议；HIL不是所有状态的最优动作；高HIL Q不直接证明自主价值准确。Q对观察回报的MSE在新DEV HIL8约−69.4%、原配方+5.9%，但MC混合直接以该回报训练，辅助成功不等于自主成功，因此不作为放行标准。

## 验证、命令、失败记录

- 86项相关CPU回归通过：导入、实际采样、探索、异步执行、存储及supported runtime。O/regression-tests.log。
- 两配方seed42连续8步 vs4步+序列化恢复+4步，149个state叶完全相同；32输入原生Actor导出最大差0。O/run-v1/resume-export-check.json。额外验证更新不计入主试验。
- 逐212/423/634/846更新保存完整state和评估数组；`run-v1/{original,hil8}_seed{41,42,43}_u846.pkl`等是研究权重，不是发布快照。
- 初次preflight缺openpi_client依赖失败；隔离worktree引用固定上游后重跑。第一次相关测试因worktree旁缺vla-platform依赖失败7项；补只读依赖后30项通过，再86项回归通过。首次画图环境缺matplotlib，保留temporal-exploration失败产物，系统python重跑到temporal-exploration-v2；不删除失败记录。

复现入口（先读manifest，使用新输出目录，勿覆盖既有结果；代码提交0b9dd4a）：

```sh
P=/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform
W=/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/matched-learning-20261012/rl-platform
O=$P/outputs/matched-learning-20261012
# 实际训练argv与资源环境在launch-v1.json，实际运行JAX CPU/CUDA禁用
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' taskset -c 172-175 "$P/.venv/bin/python" -u "$W/scripts/run_matched_learning.py" --assets "$P" --output "$O/run-v1" --prepared "$O/preflight-v2"
python3 "$W/scripts/summarize_matched_learning.py" "$O/run-v1"
python3 "$W/scripts/compare_exploration_temporal.py" --assets "$P" --output "$O/temporal-exploration-v2"
```

## 下一步与回退

本轮明确了“正确进入训练后，HIL8确实改变了局部偏好，但学习幅度小”。优先保留现有采样修复和完整回执；新权重不自动上机。下一算法实验只改teacher50→5，固定本轮HIL8采样/MC/LR/数据/初始化/846预算与3seed，用同一动作偏好、物理修正、旧DEV保持判定，失败退回teacher50；本报告只提出，未运行。不要同时提高探索强度、LR和Q权重。

现场下一步仅固定Actor的旧探索/低频探索受控对照，先验收执行质量；它不继续学习、不承担验证新训练权重的任务。出现更大往返、跟踪误差、接触恶化或发布超时即停止并回原探索入口。仍需独立位置分层Episode验证自主收益，不能把本次重复DEV转称TEST。
