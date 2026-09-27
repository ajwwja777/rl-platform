# 插孔 RLT 离线 warmup 交付（2026-09-11）

状态：**offline-trained-not-deployed**。用户授权本轮训练，Cobot 单 RTX4090 完成20000 learner updates，actor更新10000次，训练及分段评估用时119.45秒（包括首次训练编译，不含模型冷加载和专家特征提取）。没有发布到正式rollout，也没有启动机器人动作。

## 模型与训练结构

```
冻结 Stage1 step_4000（VLA + RL token encoder，不更新）
  ├─ 真实观测 → z_rl[2048]、reference action chunk
  └─ 已缓存特征 + proprio[14] + ref_chunk[10,14]
       ↓
  actor：投影层 + 2×256 MLP，输出10×14动作
  twin critic：两套投影层 + 2×256 MLP
       ↓
  critic每步更新；actor每两步更新；target tau=0.005
       ↓
  独立checkpoint/actor快照 → 离线验证 → 人工决定现场验收
```

输入特征预计算一次；训练时不重复运行VLA，不通过ROS、HTTP replay或机器人服务取训练数据。保留双臂delta/absolute转换与既有对称平滑损失。actor是独立chunk预测网络，不能把输出简单描述为有硬上限的小残差。部署侧右臂控制、左臂/双夹爪hold、速度/加速度限制属于额外执行约束。

## 数据

| 分组 | 训练episodes / transitions | 验证episodes / transitions | batch抽样 |
|---|---:|---:|---:|
| 旧专家成功 | 20 / 223 | 4 / 44 | 约30% |
| 当前成功 | 17 / 385 | 5 / 117 | 约40% |
| 当前失败 | 7 / 101 | 2 / 33 | 约30% |
| 合计 | 44 / 709 | 11 / 194 | 128条/步 |

- 专家从原78条release的train中按右臂初态/终态/运动范围做farthest-point选20条；原val选4条，test不参与。筛查有限数值、时间单调和动作跳变，原文件不修改。此为自动数据筛查，不是逐条人工视频质量评级。
- 专家原30Hz按最近帧对齐20Hz并保留终帧；图像不插值。完整chunk长度10，常规stride10，末尾向前对齐完整chunk（可能重叠），避免零动作padding；只最后一个transition终局成功。
- 24条专家总267 transitions。同一个Stage1 step4000实际提取z/ref，特征服务就绪后耗时32.44秒；全量模型冷加载是额外时间。
- 当前在线636 transitions按整episode拆分；自主成功/HIL成功分别分层留出，所以验证成功含1条自主+4条HIL，训练成功含1条自主+16条HIL。失败保留，6条放弃及历史incomplete不进入。
- 按三组 → 均匀episode → 均匀chunk抽样，避免长HIL轨迹主导。batch128取38/51/39，分别约29.69%/39.84%/30.47%，是30/40/30的整数近似。
- 训练日志sample_success_ratio统计的是稀疏transition终局标记，不能拿它当成功episode抽样比例；精确分组见training_manifest.json。
- 当前replay已有的暂停/终局时间对齐继承原数据。本轮未重新采集或重新构造在线raw trace；专家重采样不消除既有在线时序抖动。
- 专家val沿用release隔离；不要据此声称这些数据一定从未被任何历史Stage1流程看过。此处明确保证的是本轮actor/critic训练和验证episode不重叠。

## 参数与效率

batch128，actor/critic学习率均1e-4；BC权重10，Q权重0.1，平滑权重10；reference dropout0.5；fixed_std0.001；gamma0.99；actor_update_period2。整轮Q权重0.1，没有另行运行Q=0阶段。

保留原双臂速度误差、state→首动作约束、加速度误差及夹爪平滑项；权重分别遵循现有learner_patch。HUMAN/MIXED动作参与BC，policy包括失败样本的BC指向reference，不盲目模仿失败动作。失败仍参与critic。

使用已验证status_io节流避免每步fsync。数据驻CPU数组批量抽样，固定batch复用JAX编译；只在100/500/2000/5000/10000/20000保存checkpoint和actor。没有改学习率或batch来制造吞吐提升，也未声称GPU驻留采样已实现。复现环境与源码快照在run内。

## 分段结果与限制

评分：专家与当前成功验证集右臂目标RMSE的等权平均，单位rad；HIL以执行动作为目标，policy以reference为目标。这是离线初始化/模仿指标，不是真机成功率或RL收益估计。

| learner step | 验证评分 | 首动作/速度筛选 |
|---:|---:|---|
| 100 | 0.013537 | 不通过 |
| 500 | 0.010520 | 不通过 |
| 2000 | 0.009982 | 不通过 |
| 5000 | 0.009843 | 通过 |
| 10000 | **0.009824** | 通过 |
| 20000 | 0.010027 | 通过 |

筛选阈值在训练前代码固定：各组raw动作velocity p95≤max(0.01,reference×2)，首动作p95≤max(0.04,reference×1.5)。这个粗筛不包含全部动态风险，不能称为真机安全验收。相邻chunk仅比较同episode且step_id相差10的窗口；不覆盖所有HIL断点。

**候选为step10000，但不直接发布：**
- 旧专家验证的人类动作RMSE：reference0.012632，候选0.013941。
- 当前成功HIL样本RMSE：reference0.006391，候选0.006660。
- 当前成功组加速度p95：reference0.001953，候选0.009231 rad/sample²。
- 当前成功组chunk边界跳变p95：reference0.012695，候选0.015883 rad。
- 因此候选尚未证明优于reference，raw加速度仍更大。平滑补丁明显改善随机初始化，但没有消除全部抖动风险；不能宣称已经更顺滑或插孔成功率提升。step20000相较10000亦没有验证优势。

下一步应先针对候选做更严格的离线动作对照和确定性小范围现场验收，现场动作需新的明确授权。不要直接打开explore，也不要假定后续在线更新自动改善。每k条episode集中更新的调度尚未实现/启用，不由本次离线warmup自动开启。

## 产物位置

Cobot运行根：
`/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/plug-warmup-20260911-r1`

```
code/                           本轮提取、训练与数据工具
runtime-overlay/                固定的既有RLT补丁快照
expert_selection.json            专家选择与来源
expert_features/                 逐episode的真实特征缓存
expert_replay.pkl                专家transitions
training_dataset.pkl             固定train/validation完整记录
training_manifest.json           分组、episode ID、哈希和预算
training_config.yaml             训练使用的配置（离线入口消费；不用于直接启动all角色）
training/checkpoints/step_*.pkl   完整actor/critic/target/optimizer/RNG
training/actor/history/          各检查点actor快照
training/metrics/learner.jsonl    20000条训练指标
evaluations.json                 分段离线评估
reference_validation.json        冻结reference对照
validation_predictions_step_*.npz
training_result.json             运行终态
verification.json                恢复与正式资产哈希验证
delivery_manifest.json           候选路径、版本与发布边界
code_hashes.json                 源码快照哈希
expert_extraction.log
training-attempt2.log
```

候选完整checkpoint：`training/checkpoints/step_10000.pkl`，SHA256 `ea4770d1609c380a3029bcceebe37689d019f10a7f3831e2ef3bcfbf9dc58ccd`。
候选actor：`training/actor/history/actor_v005000.pkl`，SHA256 `9eb0e455e3d22a270288a7504a2566759ace125dba399902a7ba57844dfecc2c`。
`latest.pkl`是最终step20000，不是选优候选。Stage1仍是step4000，learner step10000对应actor version5000。

## 验证与收尾

- 20000连续global_step、10000 actor更新，所有日志数值finite。
- 六个checkpoint读回有限，全部在相同batch128下重现保存动作，最大差0。
- batch1部署形状与batch128结果存在约4e-5至6.1e-5量级差异；首次以1e-5比较未通过，因此另外验证相同batch精确恢复，跨batch实测小于1e-4。未把跨batch结果称为位级一致。
- 上游完整state loader恢复step20000、actor version10000，通过；本轮不额外继续优化已完成checkpoint。
- 新数据层及既有smoothness/conditioning/actor pinning共18项测试通过。最初组合测试缺upstream PYTHONPATH，补正确环境并固定CPU后通过。
- 首次训练启动在梯度更新前因numpy零维episode ID无法hash失败，显式int转换修复，保留training.log；正式运行日志training-attempt2.log退出0。
- 正式replay/checkpoint/actor/status哈希未变；collection-only保持，不替换旧正式模型。
- 本轮Machine A、训练和验证进程均已退出，GPU计算进程为空。不commit/push。测试原始产物保留用于审计。

轻量机器可读结果见同目录 `warmup-20260911/`。
