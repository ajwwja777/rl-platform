# plug_v2 Human采样、BC、Q与干预时机诊断（2026-09-18）

结论：隔离5000 critic/2500 actor更新完成；主要判据是留出episode Q排序，非Loss。修复后Q排序改善，但HIL动作拟合仍弱于保持基线、没有自主成功类别，故不发布、不追加20k。生产actor20000及原始录制/v2缓存未改；未操作机器人、访问HPC、commit/push。

## 当前证据与上游工具

已读取固定上游rlt_online_rl/trainer.py、replay.py及scripts/offline/eval_episode_q.py。作者分别记录human_mask_ratio、bc_human_penalty、bc_ref_penalty；这些用于定位batch人类监督占比与人类动作拟合，不能替代episode Q评估。上游stratified默认recent .4/demo .3/human .2/uniform .1，池可重叠，human20%专用池不等于实际human mask20%。

当前制品为Torch PT及NPZ，无用户示例的replay_journal.pkl/checkpoints目录。作者工具需要JAX合同且episode-ids为必需参数，不冒用历史制品。新增同语义当前合同评估器methods/openpi_rlt/plug_v2/learning_diagnostics.py，确定性actor_mean、min twin Q、原始UUID聚合，逐行CSV及完整/自主/最初3窗口分开比较。维度/归一化不同于上游，BC数值不可跨实现直接比较。

## 采样与监督验证

同一经过严格clock检查的数据、固定生产actor，100个batch审计：原outcome_balanced实际HIL样本35.30%、demo30.49%，human_mask66.77%；新stratified HIL20.3125%、demo29.6875%，human_mask51.80%。128 batch明确38 demo/26 HIL/32辅助成功前policy/32失败policy；不是发现旧采样uniform或旧HIL不足。独立配额用于隔离来源，不能把20%当上游保证的最佳值。

新实验使用raw actor对实际SDK human命令BC，额外172 train支持因果d6行仅作BC，不作reward/TD；raw/conditioned/派生filtered目标分别报告。真实连续接管边经generation、实际timestamp、stream显式证明后用于target_actor Bellman bootstrap，不把辅助observed MC return当自主Q标签。增加命令加速度/边界速度连续性项，减轻reference anchor；保留±.05rad残差、原CommandFilter/30Hz及固定轮内actor。多项同时改变，非单因素因果消融。

9项实际数据合同检查通过（typed边、奖励未改、配额、UUID隔离、padding、raw BC饱和梯度、AUROC方向、恒速边界等）。源事实、原reward/v2 replay未改，独立V3 snapshot仅实验使用。

## 主要判据：相同留出episode Q

11条HIL辅助成功/4条自主失败，排除专家，固定原UUID留出。

| 模型 | 完整记录Q_data AUC | policy-only AUC | 最初3自主窗口Q_data AUC | 最初3自主窗口Q_actor_mean AUC |
|---|---:|---:|---:|---:|
| 旧20000 | 1.000 | .295 | .477 | .432 |
| 新100 | .841 | .273 | .182 | .182 |
| 新500 | 1.000 | .364 | .341 | .455 |
| 新2000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 新5000 | 1.000 | .886 | .818 | .818 |

限定delay=6的最初3自主窗口复核结果相同。旧完整AUC1与自主接近随机并存，不能称critic已学好；修复后自主排序改善。2000优于5000，因此不以最小Loss或最后一步选优，也无依据自动延长20k。Q_actor_mean、Q_data相同排序仍不能证明Q对纠正方向有效。

类别是后续被HIL救成功与自主失败，非自主成功/失败；HIL选择时机、场景、轨迹本身仍可能混杂。样本只有15条且多checkpoint复用同一留出，不是独立最终验收。episode bootstrap仅描述当前样本，完全排序产生[1,1]区间不代表总体无不确定性。当前缺自主成功类别，不能检验真正自主成功率或证明Q插入方向可靠。

## Actor与顺滑检查

留出支持因果d6专家12行/HIL12行。2000预测命令MSE（rad²）：专家 .00142039 vs reference .00148601、brake .00141821；HIL .00119908 vs reference .00191016、brake .00132333、hold .00076832。HIL较reference改善约37.2%，仍不如保持；专家仍略不如减速。真实factual d0 HIL bc_human_penalty从旧 .000185880降至2000 .000164096，raw-human从 .000249103降至 .000171359。原作者指标确实能显示拟合变化，但不能替代d6/自主控制检查。

98个留出实际policy窗口：旧20000平均条件化纠偏 .000230892rad，新2000 .00457191、新5000 .00215934；新2000不再几乎等同reference。最大单tick关节步长 .00666666rad，passive偏差0；命令加速度归一化指标旧 .00279584、新2000 .000964513。以上离线指标，不能称真机顺滑或成功率提升。

## 干预时机

55条HIL成功首次接管中位2.106秒（.732–2.874）；有效HIL持续中位3.000秒（1.833–5.538）。时间本身不能证明救得太早或太晚，也没有base未被接管时的反事实结果。救晚的轨迹若最终标签失败，才按失败reward处理；若人最终救成功，不能仅因介入晚就重标失败，也不能把成功归于自主部分。

下一批应记录介入前偏差/接触是否不可逆、介入原因，明确可恢复纠偏段。固定位置分布与评估窗口，收集独立自主成功/失败以及可恢复的HIL纠正；重新验Q排序与动作方向，再决定是否发布。不能把持静止/选择HIL来源学成高Q误称插入学习。

## 执行与复现

执行根 `/media/agilex/Getea1/jiaan/projects/rlt`；实验 `runs/plug_v2/learning/human-contract-r1-20260918`。13:55:23 UTC启动PID1896213/start9076013，13:56:35完成，优化55.68秒；进程已退出/operation.lock可用，production_publish=false/production_actor_unchanged=true。

文件：operation.json、sampling-audit.json、q-comparison.json、q-d6-first-three.json、q-uncertainty.json、actor-heldout-controls.json、command-postcheck.json、completion-receipt.json、source-hashes.json；每候选eval目录有report及episode_q_rows.csv。代码与测试仍在Cobot，不在框架复制权重/日志。

CPU评估当前合同（独立新输出目录，不能复用已有目录）：

```bash
cd /media/agilex/Getea1/jiaan/projects/rlt
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/agilex/junfeng/workspace/pi05_cobot/.venv-server/bin/python \
-m methods.openpi_rlt.plug_v2.learning_diagnostics \
--replay-path runs/plug_v2/learning/human-contract-r1-20260918/replay \
--model-path runs/plug_v2/learning/human-contract-r1-20260918/checkpoints/step_2000.pt \
--actor-mode mean --allow-experimental \
--output-dir runs/plug_v2/learning/human-contract-r1-20260918/eval-independent-check
```

实验checkpoint为diagnostic_only，标准部署load_actor拒绝，未注册生产选择器。最终没有提出现场模型切换。
