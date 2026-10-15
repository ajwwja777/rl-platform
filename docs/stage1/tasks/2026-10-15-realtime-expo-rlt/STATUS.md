## DELIVERY COMPLETE 2026-10-15T19:19:07

Verified: both Web entries installed and no-motion field loads passed. Current Online is ready/disarmed, emitted_commands0, Replay551, Learner7000/Actor3500, pending budget0. Only two wrist camera subscriptions. User may proceed with controlled physical acceptance/collection; autonomous improvement remains unproven and gripper DEV retention limitation remains. Canonical final evidence and paths: [RESULT.md](RESULT.md). Earlier progress below is historical.

## Final offline candidate (2026-10-15T19:04:24)

Verified: original delta10 Warmup7000 remains selected. Single-factor delta0 comparison completed 10k updates and deployed-input evaluation: selected7k DEV joint18.7913mrad/gripper4.6126mm vs native18.7425mrad/4.6135mm. No meaningful gripper-retention improvement; no algorithm default changed. Comparison and limits: outputs/realtime-expo-rlt-20261015/warmup-single-factor-comparison.json.

Verified on Cobot CPU/NumPy1.26.4: portable release loads, Actor3500 (Learner7000), 32 deployed-input outputs exactly match A6000, zero ports/publishers/robot commands. 99 final RL CPU tests and 11 Web tests passed. Stage1 large copy remains in progress; Web activation and paused runtime acceptance remain pending. Source65ac7d4 and Webc500d3b were serially merged/pushed, preserving concurrent EXPO and chart changes.

## Delivery integration progress (2026-10-15T18:57:47)

Verified: Stage1 selected 4999; native Warmup 10,000 CPU updates completed in 317.6 seconds. Checkpoints 500/1000/2000/3000/5000/7000/10000 evaluated. Current descriptive reused-DEV selection is 7000 (44 TRAIN Episodes / 5 DEV Episodes, no TEST). Deployed float32 single-input evaluation: DEV joint MAE 18.7425 mrad vs Reference 18.8040; gripper 4.6135 mm vs Reference 3.7101. This is NOT autonomous-success evidence; gripper retention remains a limitation. Delta-weight-only 0 control is in progress before final selection.

Verified: native exported Actor matches deployed evaluator exactly on 32 inputs; a copied offline branch starts with zero pending budget, synthetic 50 arrivals earn exactly 50 updates, step7050/checkpoint/history saved, Actor3525 observed, all restored state leaves exactly equal. No production Replay or robot used.

Resolved: replay float16 reference vs deployed float32 reference caused max448 microrad difference in the first export comparison; retained the original failure and reevaluated all candidates with actual deployed inputs. NumPy2 pickle namespace was incompatible with field NumPy1.26.4; release serialization now uses public ndarray constructors without dtype/value changes. Field native unpickle and 13-file SHA validation passed.

Implementation: independent left-arm 2-wrist Stage1 server, task launcher, frozen/Online configurations and separate Replay roots; no middle/head model input; native 10Hz logical time, optional shared 50Hz/RTC/EMA publication. Online UTD1, Actor every2 updates, save/publish/history every50. Existing right-arm RLT and EXPO unchanged. Web routing and loaded/published identity are task-specific.

CPU regression: 71 RL/data/async tests, 4 release-isolation guards, 11 Web routing/directory tests passed. Stage1 selected parameter/asset copy verified from HPC to A6000; Cobot large transfer still running. No new model entry installed yet, no paused field load yet. Next: finish comparison, serial Git merge/push, SHA-guarded field source install, no-motion loading validation.

Evidence root: `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/realtime-expo-rlt-20261015/` (`warmup-native-10k/`, `release-native-validation.json`, `release-bundle-cobot-portable-sha256.json`, `release-*-tests*.log`). All prior status below is historical.

# HPC训练与交付状态

2026-10-15 18:26 CST 实时核验：721065正式5000更新完成（55分17秒，退出0），721154完整五checkpoint评测/选型/专家种子完成（31分52秒，退出0），GPU已释放。在已比较1000/2000/3000/4000/4999中，4999按预登记DEV指标选中，无指标冲突；DEV完整Episode等权C10六关节MAE18.741mrad、夹爪MAE3.486mm（非TCP误差），仍是重复DEV模仿代理指标，不是自主成功证据或全局最优。551条TRAIN专家transition、62条DEV单独保存；未开始Warmup训练，也未登记Web新RLT，当前不可交付Online。小报告和两张图已取回A6000 outputs/realtime-expo-rlt-20261015/stage1-comparison/；未复制整份权重。下一步：复核选型/专家种子，完成Warmup多checkpoint与恢复/导出验证，再做独立Web入口和现场无动作验收。

2026-10-15 16:09 CST 实时核验：721065已于15:59:32启动，ACD1-19，4GPU/48CPU/384GB，实际最长到19:59:32（4小时）。三次collective门禁后进入stage1-run2-train，日志16:09:06到553/5000更新，稳定约1.7更新/秒；最近step500的loss0.4598（VLA0.0145、RLT0.4453）、grad_norm6.9533，均有限。暂无最终checkpoint/模型能力结论。按当前吞吐剩约44分钟，不含后续保存和评测排队；721154仍等待afterok依赖。先前排队预计时间均为历史，已被本次实际启动取代。

2026-10-15 继续核对：正式训练重试721065仍排队；**尚无可交付的新Online权重**。HPC登录使用jhe724@hpc3login.hpc.hkust-gz.edu.cn，不沿用旧IP别名。

HPC输出：/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/realtime-expo-rlt-20261015
A6000输出：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/realtime-expo-rlt-20261015
worktree：/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/realtime-expo-rlt-20261015/rl-platform
分支：train/realtime-expo-rlt-20261015；训练代码0bfb297，推理/执行/选型4b85c8b，专家Replay准备4f9ef69，固定上游8cef77e。

## 已验证

- 720997在4×H100完成2次更新、保存及独立进程恢复到第3次更新。smoke/verified.json；不声称精确数据顺序恢复。
- 721124单GPU完成5个重复DEV Episode、591帧评测；同部署单条推理，约0.049–0.050秒/观测（不含加载/编译）。仅短训checkpoint2，不能当最终模型质量。
- 721135单GPU无ROS推理/RTC验证通过；固定输入重复结果一致，RTC与普通推理RL Token一致，双腕输入且拒绝多余base相机。稳定RTC约140.8–142.9ms，首次普通编译约142s、RTC约58s；HPC时间不能代替Cobot实际deadline。
- 左臂双腕ROS边界、10Hz逻辑/50Hz发布、HIL左侧时间戳、可变夹爪、拒绝旧任务回位等93项CPU测试通过；选型7项、专家Replay边界5项通过。
- Replay逻辑动作等于对应逻辑段最后实际发布命令的测试通过；完整50Hz发布trace仍须保存，不能声称每条50Hz子指令都成为独立10Hz训练样本。
- 模型原始float32 checkpoint资产不转换；推理沿用既有RLT loader的BF16恢复行为，已在metadata明确。

## 失败与修复

- 720997正式训练初始化NCCL/CUDA error999，11:46退出，未产生正式训练checkpoint；stage1/train.log/slurm-720997.log保留。短训成功不能排除此运行时故障。
- 721070批量8评测动作一致性超门槛停止。721124记录最大关节差6.03mrad、夹爪0.519mm、token差0.0433；正式选型全部固定batch1，不放宽门槛，也不将批量近似当部署动作。eval-smoke-batch-consistency.json保留。
- 首次评测重提遇调度账号查询超时；后误把分区内部debug_qos作为用户QOS被拒。使用debug分区+已授权formal-user提交成功；未产生重复运行作业。

## 已排队的自动后续

1. 721065：acd_u，4GPU/48CPU/384GB，最长4小时，排除ACD1-33。原sbatch记12小时，提交后已scontrol改4小时，以scontrol实际值为准。三次独立进程collective检查后从同一base、batch32/FSDP4训练5000更新；stage1-run2复用smoke编译缓存。当前无分配节点，稳定训练吞吐仍未测得。最近预计开始16:04为调度器估计，不保证。
2. 721154：afterok:721065依赖，1GPU/8CPU/96GB，最长2小时；未满足依赖不占GPU，前序失败自动取消。hpc-selection.sbatch依次评测1000/2000/3000/4000/4999，每个完整TRAIN44/5331帧、DEV5/591帧；产物evaluations/<tag>/、selection/。DEV选型存在指标冲突则停止，不自动发布。
3. 若选型无冲突，仅生成warmup-seed/train_expert_journal.pkl、独立dev_expert_journal.pkl与TRAIN-only norm；保留UUID/负专家ID/指令目标/参考动作/成功来源，不复制旧右臂Replay。全成功人工示范不能证明Critic能区分自主失败。

## 仍未完成

正式Stage1、多checkpoint模型结论、Warmup训练与选型、左臂Web完整启动/暂停无动作验收、Cobot校验同步、真机成功率与Online收益。RTC目前只是HPC接口验证，不是现场平滑验收。没有独立TEST。

下一步按721065实际日志核对通信/吞吐/checkpoint，721154自动做Stage1评测和专家种子；随后完成Warmup与部署入口。不要重复提交已有作业，不抢A6000/Cobot GPU，不改现场服务/机器人/旧Replay。
