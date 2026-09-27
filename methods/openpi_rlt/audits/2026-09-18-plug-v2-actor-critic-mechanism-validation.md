# Actor可学习性与Critic纠偏方向：离线机制验证

## 结论

用户授权验证两个前提；当前原流程不具备继续堆叠在线更新即可改善的证据。最强事实是自主部署d6子集没有正向BC、成功奖励或到HIL成功段的bootstrap路径。已有actor能产生非零可执行改变量，但未验证出优于简单减速基线的纠正能力；critic能拟合部分回报，却未验证可靠纠正方向。此次不改生产训练/执行代码、不发布诊断模型、不操作机器人，不访问HPC/commit/push。

执行目录 `/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/mechanism-validation-20260918T130128Z`。13:02:03–13:02:23 UTC，PID1744039/start_ticks8755974；原始validation.log、validate_mechanism.py、dataset-audit.json、results.json、controls.py/controls.json、context-ablation.json。两项2000 actor更新分别8.54/8.16秒，均是BC-only诊断、无Q/reference/smooth附加目标，非部署warmup。

## 真实数据支持

| 划分/上下文 | 行数 | 正向BC行 |
| --- | --- | --- |
| train d0 | 6180 | 6180 |
| train d6 | 388 | 0 |
| heldout d0 | 985 | 985 |
| heldout d6 | 99 | 0 |

d0正向为专家/HIL；实际policy487行均d6。含成功奖励的窗口684个全部d0（teacher overlapping windows，不是684episode）；d6成功奖励窗口为0、终局11均零奖励。bootstrap边只有teacher->teacher6466和policy->policy411，没有policy->HIL成功边。故自主子集Q=0即可满足其零奖励/独立截断的Bellman方程；低TD不能作为学到插入策略的证据。此为当前记录支持/适配契约缺陷，不可直接把暂停/缺帧/复位后数据拼接去补奖励。

## 两个监督探针

d0 BC-only用6180真实teacher行训练，985留出行检验：train MSE降低30.48%，heldout仅降低0.55%（.000259393->.000257954），泛化验证不充分。该小实验使用统一按row抽样与2000更新，不代表穷尽该网络可学习性。

d6配对使用真实自主请求的z/state/已承诺prefix/ref，与随后采集HIL标签按控制时钟对齐。未来teacher动作仅作为label，禁止进入prefix或观测；42原UUID训练/10原UUID验证、零交集。仅针对接管附近10个post-prefix动作中的有效HIL位置，训练sample需至少3个有效不同帧。HIL接管实际上丢弃原prefix，因此这些是**反事实监督诊断，不是实际执行过的RTC专家轨迹**，无奖励/bootstrapping，不写正式replay。hypothetical critic动作窗口包含全部10动作，未知标签位置使用reference。

| 配对heldout（10 UUID） | 相对reference的MSE降低 |
| --- | --- |
| 原500 | 3.43% |
| 原20000 | -2.75%（更差） |
| d6配对BC-only | 63.00% |
| 有界减速基线（train选beta=.75） | 68.34% |

减速基线只将raw ref向实测request state收缩，仍限于每关节±.05rad残差，再用同一30Hz平滑/限速，beta仅按train选，不用val调参。它优于BC-only，说明配对改进主要可能来自接管时制动，不能把63%称为学会插孔纠正。d6探针条件化改变量均值约.01023rad，证明不是网络输出绝对被锁死。配对teacher标签与request state差均值约.0132rad，短时间窗只覆盖接管阶段。当前表示+限速下12.63%有监督坐标超出可达区间（严格逐坐标时间可达包络），MSE下界.00004050；剩余可达标签仍未给出优于简单基线的泛化证据。d0不可达坐标8.77%。

打乱heldout视觉token后d6 probe MSE .00012046->.00015367，存在敏感性，但只有10样本且弱于制动基线，不证明视觉定位能力。

## 冻结Critic与上下文消融

10反事实heldout配对中，对同一状态的物理可执行HIL目标（greedy projected feasible action，并非全局最优oracle）评分：500的min twin Q仅40%高于ref，20000为50%；Q1局部梯度沿该纠正方向为60%/50%。样本很少且没有reference反事实终局，不能称分类准确率或判定critic必然错误；缺乏可靠纠偏支持。

仅将actor输入改成cold d0、执行平滑仍保留真实d6 prefix的输入消融：20k在99真实d6留出状态的平均执行改变量 .0002326->.0024843rad（约10.7倍）；配对MSE .00033455->.00030695。支持上下文对纠偏抑制的依赖性，但不能单独证明因果根源，且这种输入不符合部署状态契约，**不得作为真机绕过方法**。

## 制品隔离与下一步

`d0-bc-only.pt` / `d6-boundary-bc-only.pt`明确status=diagnostic_only，生产load_actor拒绝；只为诊断留存。factual-d0-train/val.npz、counterfactual-train/val.npz是独立测试输入，不混入正式replay。现有生产20000 SHA `c6395e9d44139326625cdc2c894fe56b5d8c0196fcb4d722267a0eb24715a9b3` 保持不变，原数据未删除或修改。

优先修数据/目标合同：真实或因果重演的部署RTC上下文应有可执行纠正监督；成功回报只能沿同一物理episode实际有效状态/动作传递，handover/generation转换须显式建模，不伪造跨暂停/缺帧/复位bootstrap。reference/smooth-boundary与BC监督应分开，先验证actor优于保持/减速基线且在独立场景有纠正能力，再启用Q优化。单纯降anchor并增大Q存在无可靠方向的风险，当前未启动新生产训练。
