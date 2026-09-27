# 2026-09-20 干净数据重训与在线学习判断

结论：暂不放行在线 RL；新候选未发布。下述离线结果不代表真机成功率。当前5000仍包含已删除旧11条数据的训练影响，删除文件不会消除权重影响。

## 已完成

- 按用户要求删除旧 actor2000 的11条数据及对应缓存；删除审计在 Cobot runs/plug_v2/audits/20260920-retire-actor2000-online.json。
- 保留158个有效episode特征文件，训练可用TD行3687；重叠human RTC窗口不能计为独立rollout。
- 随机初始化两组actor/critic、优化器与RNG，排除已删除UUID，不续训旧权重。冻结Stage1未改变。
- 干净联合训练5000 global steps / 2500 actor updates；分阶段对照5000 global steps / 2000 actor updates。分阶段前500步actor仅BC，500–1500冻结actor只训critic，之后联合训练。更新预算不同，不能称严格等actor步数实验。
- 完成loss、Q沿episode位置、同状态动作Q排序、部署滤波后动作、输入分支敏感性和最近失败状态反事实对照。

## 结果与限制

连续RTC(d6)同滤波目标MSE：reference 9.0346e-5、现部署5000 4.0602e-5、干净联合2.8043e-5、分阶段3.1461e-5。但对实际human命令MSE仅从reference .00217465变为干净联合 .0020856，不能用同滤波目标指标宣称大幅真机提升。

同状态Q(human)>Q(reference)：现部署65.45%，干净联合60.68%，分阶段67.5%。共同留出仅11条后续HIL成功与4条失败，无独立自主成功；留出已重复使用用于选方案。分阶段Q偏负且动作拟合未优于联合，未证明训练顺序改动有效。

原仓库已具备分支编码：token→256+LayerNorm，proprio→64+LayerNorm+tanh，chunk→256+LayerNorm+tanh。打乱action chunk会明显改变Q，不能说critic完全不看动作；proprio敏感性较低，仍需控制变量。打乱测试可能形成不自然组合，不能等同因果贡献。

最近两轮失败均触发workspace保护；原始actor修正经过部署限制后明显缩小。干净重训在相同失败状态上略减向前/增加向下分量，未充分扭转方向。该测试不是新策略闭环，link6 FK也不是实际插孔误差。起始右臂关节接近专家起点（最近最大差约0.000331rad），不能简单归咎于错误关节初态。

## 产物

Cobot根 /media/agilex/Getea1/jiaan/projects/rlt：
- runs/plug_v2/diagnostics/rtc-clean-20260920/：manifest、candidate-comparison.json、staged-comparison.json、modality-audit.json、staged-modality.json、start-support.json、failed-state-counterfactual.json、figures/。
- runs/plug_v2/learning/rtc-clean-original-20260920/：干净联合checkpoint。
- runs/plug_v2/learning/rtc-clean-staged-r2-20260920/：已完成分阶段checkpoint；无r2目录的第一次尝试因日志字段缺失失败，不作完整结果。
- methods/openpi_rlt/plug_v2/modality_audit.py、staged_warmup_experiment.py；均为独立离线工具。

用户明确要求的六张PNG已复制到笔记本 D:/Code/jiaan_workspace/cobot_rlt/figures/plug_v2/20260920-training-review，并逐张检查可读性。本地不复制数据、权重或源码。

## 下一步

先定位reference视觉支持、动作时序/速度与部署滤波造成的方向误差，核查训练目标与实际执行的差异，再决定是否重训Stage1。当前证据不支持只增加warmup步数或直接启动在线RL。需要独立自主评估和有效动作价值证据，不能以loss稳定代替。新候选保持隔离；不自动恢复动作、不申请HPC、不放宽保护。

## 后续：部署前缀支持与训练目标可执行性

新增 execution-feasibility.json、supported-prefix-audit.json 及对应脚本，位于同一 Cobot rtc-clean-20260920 诊断目录；仅离线读取。第7张 execution-prefix-support 图已同步笔记本并检查。

代码核查：human d6前缀使用未来六条人工命令；实际自主部署前缀来自过去已提交的限速命令。它是teacher条件增强，不等同自主策略前缀分布。conditioned 与 CommandFilter 都先速度clip，再按前缀末速度限制加速度；若人工前缀本身过快，后者可使结果暂时超出速度上限。这不证明合法自主前缀会同样超限；未修改生产滤波器或提高限速。

按文件等权，val专家d6窗口58.88%、HIL d6窗口48.51%在前缀末步至少一关节速度超过0.1rad/s。进一步要求前缀内部全部速度<=0.10001rad/s且全部加速度<=0.9001rad/s²，仅17/356个human d6窗口符合，分布于14个文件；339个不符合。该条件仍不代表全部闭环支持，样本少且重复使用留出，不能放行。

17个符合窗口按窗口加权：reference原始MSE .00257166，干净actor .000443312（降82.76%）；部署后对实际human动作MSE .00363162 -> .00335281（仅降7.68%）。人工目标自身经过同滤波MSE .00321412：这是目标不可按相同时序重现的证据，不是最优控制器误差下界。先前同滤波目标比较不能独立用作发布判据；还要分开检查prefix支持和对实际human动作的误差。

下一步优先建立真实部署前缀下的BC对照与执行可行目标审计；若用合成/重新计时的动作，不能捏造对应图像、next_state或当成事实TD transition。不得直接裁剪前缀后沿用旧z/ref特征，因为Stage1以prefix为条件。暂不盲目重训Stage1，不提高机器人限速，不启动在线RL。
