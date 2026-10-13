# Online-review 交接核验、单因素实验与候选准备（2026-10-13）

## 当前结论

**已验证：优先候选仍是 teacher5，不叠加 AWBC、Actor min-Q 或梯度裁剪作为默认。** 从 HIL8 7281/Actor3640 完整状态出发，沿用原 5k teacher，仅将保持损失系数50降至5。候选8127/Actor4063已在A6000独立打包并通过原生CPU加载、更新预算、双采样随机流/完整训练状态恢复和Actor接口验证；**尚未注册或同步现场，不是网页已有的新Online入口**。

**失败：**旧review脚本的回报/校准口径不正确，已修复；AWBC W2未满足旧开发集保持要求；clip10和min-Q未达到预登记的增益门槛。这里的实验未通过不等于这些方法普遍无效。

**证据不足：**没有独立TEST、位置泛化或自主Online成功率改善证据。新HIL TRAIN和DEV各只有1集。准备好了软件候选，不把它写成持续Online已经有效。

## 身份与可恢复位置

- 主项目：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`，下称P。
- 本轮产物：`P/outputs/review-validation-20261013/`，下称O；正式实验`O/run-v2/`。`run-v1`是初版实现验证，不能把两次重复算成6个独立种子；最后在隔离模块实现上重跑3配方×3seed，teacher5三个完整state与此前teacher对照逐位一致。
- 工作区：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/review-validation-20261013/rl-platform`，分支`experiment/review-validation-20261013`。
- 本轮源码提交：58eab6b3148206d73d541696b4ae0209f9747fd0；逐文件身份`O/final-code-sha256.json`、启动时身份`O/launch-v2.json`。固定上游、现有运行训练模块、生产配置均未修改；保留其他对话EXPO提交。
- 交接原件：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/online-review-handoff-20261013/rl-platform/docs/online/tasks/2026-10-11-online-review/HANDOFF.md`。旧研究worktree和历史产物保持，选择性导入4个工具/模块，来源SHA见`O/import-provenance.json`。

| 项目 | 实际身份与边界 |
|---|---|
| 共同初始化 | `P/models/rlt/plug_v3_yyshadow/history/candidates/supported_hil8_clip_20261010/checkpoints/latest.pkl`；SHA `fc27c4ce86d808f4f738f7788b8e5be20598247a342d92235d1ea857638903ff`；global7281/Actor3640 |
| 固定teacher | 同目录`teacher.pkl`；SHA `3669525ccea7a22a914fbfb9ea5963d6728f7f904edbaab57bbe1ea305949ae6`；原5k teacher，不重新锚定到7281 |
| 旧训练池 | 2777条/208Episode；SHA `029d09050303a6d87b0f40b52ad69ced36c9caf20c9d189e103f52ea53ec2ceb` |
| 新采集快照 | 29Episode/1068条，SHA `919ff1adb31b6fcae0f3b3380ca80dc2e881b7f2501566d7042237674d0dcad8`；本轮只读现场复核仍相同 |
| 新TRAIN | 23Episode/846条；9自主成功、1辅助成功、13失败；沿用固定划分，不因结果改分组 |
| 新DEV | 6Episode/222条；2自主成功、1辅助成功、3失败；已分析过，非独立TEST |
| 旧DEV | 20Episode/370条；重复选型DEV，与训练身份隔离 |
| 最终候选训练池 | 3623条/231Episode；命名空间映射后按实际到达顺序纳入；无DEV；本轮旧数据不重复领取更新预算 |

数据完整路径、命名空间、所有输入SHA见`O/run-v2/input-check.json`，实际batch索引见`*_seed*_batches.npz`。不能从源内Episode号码推断不同journal之间的新旧关系。

## 修复的分析问题

1. **重叠窗口折扣错误。**旧`review_online_training.py`按`gamma**(C10×剩余窗口数)`推回报。正确时间距离来自原始`step_id`和终局控制步；例如起点0、5两个重叠窗口，终点14，回报是`gamma**14`与`gamma**9`，不是`gamma**10`与1。新增合成回归验证。
2. **将接管前回报置零却称回报校准。**旧脚本混入一个人为自主目标，未保留原始行为回报语义。新工具保留真实奖励，分别标记接管前、含人工动作窗口、接管后；辅助成功回报不能解释为自主价值。
3. **统计与动作比较口径。**Episode键加入phase，拒绝重复身份；统计按Episode等权；HIL误差仅取实际HUMAN/MIXED时间点、六活动关节；对所有Q比较固定实际夹爪，分别保存Q1/Q2/min-Q；末3行等仅为时间位置，不称语义阶段。
4. **历史实验checkpoint配置记录错误。**旧锚定/AWBC工具保存原cfg，未记录各臂实际Actor LR和delta。现在保存实际臂参数，并明确research_only；不重写历史checkpoint，不凭历史元数据恢复训练。

在旧训练池2777行上，旧/新回报定义的逐行绝对差均值0.04384、最大0.52560；旧DEV370行对应0.04745/0.55268。这是定义错误的数值影响，不是泛化置信区间。接管前平均旧/新回报差，按Episode等权为TRAIN0.23923、DEV0.29515。修正后HIL8旧DEV接管前Q1相对**含未来人工帮助的观察回报**偏差约−0.200，而旧定义会把正Q直接解释成正偏差。此对比说明旧校准结论不可靠，**不证明Critic本身已正确，也不推翻所有历史独立实验**。

本次数据均通过单次终局0/1奖励和时间连续性检查，因此观察回报`G_t=gamma^(T-t)×success`在[0,1]。这个范围的依据是实际奖励结构；不把它推广到多次奖励/其他任务，也不只凭Q>1判定算法失败。9/29旧报告及其五折TD组合实验尚未在本轮逐项复现，不能据此认定四项改动都必需。

修正后5k/7k/HIL8/W2/teacher5统一回放：`O/train_reused-corrected-review.json`、`O/dev_reused-corrected-review.json`。5k/7k/HIL8的状态排序不能代替同状态动作价值验证；绝对关节目标偏差不能简单乘chunk数解释为物理误差累积。

## 同起点单因素对照

预登记`O/PLAN.json`。三臂：teacher5基线、仅Critic梯度裁剪10、仅Actor目标改min(Q1,Q2)。各3seed（41/42/43）、846次Critic/423次Actor；相同初始网络/target/Adam和配对RNG、完全相同batch。A6000 CPU172–175，CUDA关闭。

其余固定：ActorLR1e-5、CriticLR1e-4、MC0.3、target clip[0,1]、BC5/Q0.1/delta10、teacher5、reference dropout0.5、gamma0.99、C10、batch128、Actor每2步更新。此前teacher50/5对照不是重做选择，见[前轮](TEACHER_CONTINUATION.md)。

| 假设/候选 | 证据 | 状态/处理 |
|---|---|---|
| teacher50限制部分纠正学习 | 前轮3seed新HIL DEV较teacher50改善1.61%，绝对0.03445mrad；本轮teacher5完整149叶×3seed精确复现 | 已验证局部拟合改善；采用teacher5候选，非自主收益证明 |
| 梯度爆炸是当前主要瓶颈 | 本轮最大梯度范数2.180，clip10在2538次更新中0次触发；动作差仅浮点量级 | 证据不足以支持该根因；增益筛选失败，不启用 |
| Actor改min-Q明显改善纠正 | 新HIL DEV相对teacher5约恶化0.0008%；旧DEV人工MAE约改善0.094%、ref约改善0.653%，未达到新纠正收益门槛 | 增益筛选失败；保留研究开关，默认仍Q1 |
| AWBC W2可以直接上机 | 旧实验相同seed/预算对A0：旧自主DEV p95误差在281次时+6.17%至12.18%，1281次时+15.08%至25.64% | 保持检查失败；不采用。该误差是拟合代理，不把它直接叫成功率退化 |
| 无纠正数据时离线学习存在1.4mrad硬上限 | 仅少数已试目标/超参数达到该量级，无普遍上界证明 | 证据不足，不写成定律 |

teacher5新HIL TRAIN/DEV误差分别1.62431/2.10772mrad；旧DEV人工MAE2.02887（相对7281+0.23%），旧ref MAE0.74480（−0.63%）。旧DEV人工误差变化的6Episode bootstrap95%区间[−0.01208,+0.02043]mrad；新HIL各1集不制造泛化置信区间。图中阴影仅为3个种子的范围。配对Q当前重构动作不等于历史未执行动作的真实收益；teacher5没有证明Critic更准。

![学习曲线](../../../../outputs/review-validation-20261013/run-v2/learning-curves.png)
![Critic梯度](../../../../outputs/review-validation-20261013/run-v2/gradient-norm.png)
![同状态动作Q](../../../../outputs/review-validation-20261013/run-v2/paired-q.png)
![AWBC保持代价](../../../../outputs/review-validation-20261013/run-v2/awbc-retention.png)

## 实际采样

每批51近期、38warmup、26human、13全池；池可重叠，是抽样名额，不是互斥数据比例。近期池内部沿用窗口均匀采样，**本轮没有改成Episode均匀**。Actor8个近期HIL槽替换基础batch中的8个位置，总数128。global7281为奇数，Actor使用实际更新序列[0::2]。

| 互斥类别 | Critic实际抽样 | Actor实际抽样 |
|---|---:|---:|
| 旧专家 | 35.27% | 33.50% |
| 旧rollout | 34.71% | 36.74% |
| 新自主成功 | 17.46% | 16.69% |
| 新辅助成功 | 0.28% | 1.35% |
| 新失败 | 12.27% | 11.73% |

新数据总体Critic30.01%、Actor29.76%；Actor实际新人工动作窗口约1.23%，不等于新辅助成功整Episode的1.35%，专家编码的human也不能算作新HIL。DEV抽样0。比例是本次逐集到达/更新的实测，不保证未来每批相同，更不是固定success:failure=8:2。

![真实采样](../../../../outputs/review-validation-20261013/run-v2/actual-sampling.png)
![Q与目标分布](../../../../outputs/review-validation-20261013/run-v2/q-target-distributions.png)

## 候选及软件验证

唯一完成候选：`P/outputs/review-validation-20261013/candidates/teacher5-8127-v2/`。含profile、原teacher、norm、3623条独立Replay、完整checkpoint、原生Actor4063、pending_actor和online.yaml。不覆盖原生产资产。

- 预先固定seed42，不挑最高单seed；原teacher和归一化SHA保持。
- 导出时重建全部846次采样回执，与训练Actor/Critic索引逐项一致；保存实际双RNG状态与global8127。
- 19项针对性CPU回归通过；三个实验臂8vs4+4完整149状态叶精确恢复、研究Actor导出一致。
- 最终包使用实际supported_hil_runtime和原生Learner加载方法验证：历史池无重复更新预算，新增加0/1/18/39条对应预算0/1/18/39。
- 最终包训练状态、两组采样RNG和8批真实身份8vs4+4一致；实际原生ActorService方法在32输入上最大误差1.19e-7；无服务实例启动、端口、Stage1前向、GPU或运动。
- 每50更新保存checkpoint/生成pending，发布策略staged，继续保持Actor直到显式晋升。此处是包配置与原生CPU验证，尚无现场运行证据。

完整回执`O/candidate-verification.json`，各资产SHA在其中；`O/teacher5-reproduction.json`、`O/run-v2/resume-export-check.json`、`O/final-tests.json`。首版包因NumPy标量计数错误未写完provenance，保留`candidates/teacher5-8127/FAILED.json`，**禁止使用该不带v2的目录**。已修复导出工具后在新v2目录重新完整验证。训练venv无matplotlib，绘图使用现有系统python3；初次review缺openpi_client路径已修复，错误和重试日志保留，未改共享环境。

## 现场边界与下一步

本轮只读Cobot确认旧29集journal SHA不变；constant目录38条底层运行记录，transitions_written总和0、Replay为空。不能把success0的底层记录当36个标注失败，也不能把用户口述12/15回填为训练标签。证据`O/field-collection-inventory.json`，此前evaluation路由核对见TEACHER_CONTINUATION。

离线算法筛选与候选软件验证已完成。进入受控分批Online前仍需把**本包**通过现有项目发布流程登记/同步现场、核对当前部署负责人及模型状态，并做无动作身份检查。当前网页旧入口不会自动变成teacher5，本轮未接管EXPO或修改现场。

接续原则沿用已确认的分批流程：明确从数据采集路由进入，先闭合1个完整Episode的录制/trace/Replay/模型身份；再小批新经验更新，保留固定执行配置和位置/自主/HIL记录。更新到pending后做相同离线门槛和同条件冻结比较再晋升；入库错误、非有限值、保持越界或执行故障时停该批更新/晋升，返回已登记的Actor3640。尚不能承诺每一批Online都会提高成功率；下一步应获得真实新数据和行为证据，而非继续在同一个1集HIL DEV上堆超参数。

## 命令与复现

完整实际命令：`O/launch-v2.json`、`O/verification-commands.json`、`O/package-commands-v2.json`、`O/final-tests.json`。按新输出目录复现，不覆盖既有结果。模型计算使用P/.venv/bin/python加`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu taskset -c 172-175`；绘图使用系统python3。

- `scripts/run_teacher_continuation.py --study critic --assets P --output <新目录> --prepared P/outputs/teacher-continuation-20261013/preflight`
- `scripts/summarize_review_validation.py <实验目录>`（父目录需有awbc-retention-review.json，此为本任务专用汇总）
- `scripts/review_online_training.py --journal <journal> --norm-stats <norm> --upstream-src P/third_party/openpi-rlt/rlt_online_rl/src --split-label TRAIN_REUSED或DEV_REUSED --checkpoint 名称=完整路径 --output <新JSON>`
- `scripts/prepare_teacher5_candidate.py --assets P --study <实验目录> --output <新candidates目录>`
- `scripts/verify_teacher5_candidate.py <候选目录> <新验证JSON>`

回滚本轮研究开关只需不选择它们；生产训练源码和默认未改。恢复本任务先读本文件和coordination，不重复运行已完成的9组实验。
