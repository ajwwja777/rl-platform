# HPC训练与交付状态

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
