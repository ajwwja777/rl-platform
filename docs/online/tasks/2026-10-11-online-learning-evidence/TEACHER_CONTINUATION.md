# 从 HIL8 7281 续训：teacher50 对 teacher5（2026-10-13）

## 结论

**已验证**：只把teacher损失系数从50降至5，三个配对种子的新HIL开发集误差均进一步下降，平均相对teacher50改善1.61%；旧DEV平均保持及人工动作误差尾部在预先登记的工程筛选范围内。支持“teacher50限制了部分纠正学习”，不支持“它是唯一瓶颈”。绝对改善约0.03445mrad，仍然很小。

**证据不足**：新HIL TRAIN/DEV各仅1集，旧DEV反复选型，未建立独立TEST、位置泛化或持续Online收益。teacher5可进入后续候选准备；当前现场Actor3640及生产Replay保持，不自动发布8127研究权重。

**失败／证据缺口**：最新低频入口运行并没有进入采集Replay，不能将用户口述约12/15直接纳入这轮训练。实查为部署evaluation路由；该路由按设计关闭探索和Replay提交。之前把“选中低频入口、抖动可接受”直接当作低频探索真实生效的证据不充分，现明确纠正。

## 固定身份、配置及实际输入

主项目根 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`。产物根 `outputs/teacher-continuation-20261013/`，下称O；隔离工作区 `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/teacher-continuation-20261013/rl-platform`，分支experiment/teacher-continuation-20261013，基线d138611。训练只用A6000 CPU172–175，CUDA关闭；没有现场GPU、服务启停、模型加载或机器人动作。

| 项目 | 实際身份 |
|---|---|
| 共同初始完整state | `models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010/checkpoints/latest.pkl`；SHA fc27c4ce86d808f4f738f7788b8e5be20598247a342d92235d1ea857638903ff；global7281/Actor3640 |
| 初始Actor | 同包actor_snapshot；SHA020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5；训练state参数逐叶等于该快照 |
| teacher | 同包teacher.pkl；SHA3669525ccea7a22a914fbfb9ea5963d6728f7f904edbaab57bbe1ea305949ae6；保持原teacher，不重新锚定到7281 |
| 旧训练池 | 同包2777条/208集；SHA029d09050303a6d87b0f40b52ad69ced36c9caf20c9d189e103f52ea53ec2ceb；旧数据不重新领取更新预算 |
| 新数据快照 | 前一实验exploration-29ep-1068.pkl；SHA919ff1adb31b6fcae0f3b3380ca80dc2e881b7f2501566d7042237674d0dcad8，与现场原探索journal当前SHA仍相同 |
| 新TRAIN/DEV | 保持原comparison-manifest：23集846条TRAIN（9自主成功、1辅助成功、13失败）；6集222条DEV（2自主成功、1辅助成功、3失败），不重新划分 |
| 旧DEV | 外部20集370条，旧5k及本次训练池无身份交集；重复开发数据，不称独立TEST |

完整SHA/路径见O/run-v1/input-check.json；12个候选训练源码SHA与原profile全部相同，见source-contract.json。两组网络、target、Adam状态和计数一致；研究RNG按41/42/43配对，采样随机流也配对重置，**不冒充复原历史实时随机流**。

唯一训练差别：teacher系数50 vs5。ActorLR1e-5、CriticLR1e-4、MC0.3、目标clip[0,1]、BC5/Q0.1/delta10、reference dropout0.5、gamma0.99、C10、batch128、固定夹爪Critic契约全部相同。保持source/HIL/结果标签，不改原始reward。

逐集按原采集顺序加入新TRAIN，每条新增transition分配1个Critic更新，累计846次；Actor每2次更新一次，423次。因起点7281为奇数，**第一步就更新Actor，实际Actor采样统计取schedule[0::2]**，不能沿用原5k实验[1::2]。最终均global8127/Actor4063；212/423/634/846更新留完整研究state及评估数组。

## 两组实际采样完全相同

每基础batch从recent/warmup/human/all分别抽51/38/26/13个名额，有放回且池重叠。Actor专用8个真实HIL名额是**替换基础batch中的8个位置，总数仍128**，不是另加成136。

预先生成真实SupportedSampler的Critic和Actor身份序列，两组逐数组完全相同；实际Actor更新时点与原生did_actor_update断言一致。新数据Actor平均占29.7608%、Critic30.0138%；新人工动作窗口Actor约1.2257%，不能与整条辅助成功Episode或专家编码来源混淆。DEV实际抽样0。证据：O/preflight/sampling.json与两组*_batches.npz，实际训练亦保存相同序列。

![实际采样](../../../../outputs/teacher-continuation-20261013/run-v1/sampling.png)

## 学到了什么

以下六活动关节MAE按完整Episode等权，再平均三个种子；夹爪/左臂固定，不作退化能力指标。

| 指标，mrad | 起点7281 | teacher50续训 | teacher5续训 |
|---|---:|---:|---:|
| 新TRAIN人工动作，1集 | 1.68859 | 1.65907（−1.75%） | 1.62431（−3.81%） |
| 新DEV人工动作，1集 | 2.17309 | 2.14217（−1.42%） | 2.10772（−3.01%） |
| 旧DEV人工动作，6集 | 2.02419 | 2.02965（+0.27%） | 2.02887（+0.23%） |
| 旧DEV对Reference误差，20集 | 0.74955 | 0.74814（−0.19%） | 0.74480（−0.63%） |
| 新DEV相对起点的动作变化 | 0 | 0.15759 | 0.29446 |

新DEV teacher5相对teacher50改善1.608%，三个配对种子都改善。旧DEV人工p95平均相对7281+0.278%；按Episode的三个种子平均，最差旧人工MAE+1.36%、最差旧Reference-fit+3.38%，不能把整体平均保持说成每集都改善。

旧DEV人工MAE相对初始的完整Episode bootstrap95%区间：teacher50变化+0.00547mrad，区间[−0.00027,+0.01138]；teacher5变化+0.00468mrad，区间[−0.01208,+0.01988]。新HIL各仅一集，不制造跨Episode泛化置信区间。学习曲线阴影是种子范围，不是独立样本置信区间。

![更新曲线](../../../../outputs/teacher-continuation-20261013/run-v1/learning-curves.png)
![分关节误差](../../../../outputs/teacher-continuation-20261013/run-v1/hil-joint-errors.png)
![Actor损失组成](../../../../outputs/teacher-continuation-20261013/run-v1/actor-loss-components.png)

同状态纯HIL chunk，固定物理夹爪比较Q1(HIL)-Q1(Actor)，两组在新TRAIN/DEV三个种子均为正；teacher5并未产生更强的Q优势，不能把Actor误差改善解释成Critic已更准。新DEV记录动作Q对观察回报的MSE两组约−32%，旧DEV反而约+3.0%/+3.7%；奖励含未来人工帮助，不能当自主价值认证。Q1与min-Q分别保存。Actor提议由当前checkpoint重构，不是接管时历史原提议。

![同状态Q偏好](../../../../outputs/teacher-continuation-20261013/run-v1/paired-q.png)

## 筛选与验证

训练前O/PLAN.json登记工程筛选：teacher5新HIL DEV平均比teacher50改善至少0.5%、三个种子不更差；旧DEV人工与Reference MAE平均恶化不超过2%、人工p95不超过5%；非有限值拒绝。本轮全部通过，见O/diagnostics.json；这是候选筛选，不是机器人安全/任务成功认证。

16项导入/采样CPU回归通过。两组seed42连续8步与4步+保存恢复+4步的149个state叶完全一致；32输入Actor导出最大差0，见run-v1/resume-export-check.json。验证更新与846主实验分开。所有受保护源资产SHA在训练结束复核不变。

初次汇总脚本遗漏改旧文件名original_seed41，绘图后组成统计报错；已修正为teacher50并修正7281奇数起点的Actor统计切片，完整汇总重跑成功，保留summary.stderr/summary-v2.stderr。训练本身六组均完成且未重跑。初次只读现场日志探针的转义错误也保留，无资产修改。

实际命令保存O/launch.json（含脚本SHA、PID与CPU限制）。复现入口如下，运行时必须另设新输出目录，不覆盖既有结果：

```sh
P=/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform
W=/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/teacher-continuation-20261013/rl-platform
O=$P/outputs/teacher-continuation-20261013
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 taskset -c 172-175 "$P/.venv/bin/python" "$W/scripts/run_teacher_continuation.py" --assets "$P" --output "$O/run-v1" --prepared "$O/preflight"
python3 "$W/scripts/summarize_teacher_continuation.py" "$O/run-v1"
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu taskset -c 172-175 "$P/.venv/bin/python" "$W/scripts/verify_teacher_continuation.py" "$P" "$O/run-v1"
```

## 新近现场记录：为何不能进入本轮训练

只读核对：低频入口online/metrics/rollout_metrics.jsonl有38次底层运行记录，全部transitions_written0；两条success，其余底层success0不等于36个标注任务失败。对应private replay与trace目录为空；session-use.json为evaluation。实际exploration.py只在lifecycle.collecting为true施加偏移，共享环境也只有collection路由允许Replay提交。源码与配置名不能替代逐Episode实际启用证据。

找到2条完整网页success、intervened=false记录，位于旧网页选择的目录：
`/media/agilex/Getea1/jiaan/data/evaluations/in_the_pot/vla-platform/pi05/dagger_round_001/eval-20261013T154252-36dea0b9/result.json`
及同父目录`eval-20261013T154530-1dcb5a65/result.json`。记录内部model为RLT constant，不能按目录名误认为pi05。两条登记50Hz/RTC/滤波开启；最后一次加载尝试显示滤波关闭的设置，不能混为这两条实际配置。用户口述12/15及位置变化保留为用户反馈，目前没有逐轮记录闭合，不当独立TEST/训练数据，不改标签或回填Replay。

证据O/field-inventory.json、new-evaluation-inventory.json、deployment-records.json、field-code-contract.json及field-shared-model.diff。现场shared_model_env仅目录选择函数保留历史差异，collection/evaluation路由代码相同；本次未覆盖它或改web。没有发现足以证明“采集提交代码坏了”的证据，当前缺口是操作路由/可追溯记录。

## 下一步

保留当前7281现场基线。teacher5作为从当前模型继续学习的优先候选，但尚未打包注册/同步现场，不因工程筛选通过自动开启Online。下一次现场应明确区分冻结评测与固定权重采集：若需要训练数据，使用数据采集并完整保存结果，先核对1集录制/trace/Replay入库和探索是否启用；不需要反复测相同权重凑成功率。新候选部署还须完成独立包及原生加载契约，固定执行方式交替比较后再分批学习。任何失败可回当前Actor3640；生产权重未变，不需要回滚本次离线实验。
