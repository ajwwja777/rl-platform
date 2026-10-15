# realtime_expo 左臂RLT训练（2026-10-15）

状态：短训/恢复通过；首个正式作业720997初始化NCCL失败。4卡重试721065排队，单卡评测721070暴露批量推理一致性失败，正在修复。真实进度见[STATUS](STATUS.md)。未交付新Online权重。

## 任务与范围

用户要求从Cobot `/media/agilex/Getea1/xiqing/data/datasets/realtime_expo` 训练新RLT用于Online rollout，并于2026-10-15确认49条完整示范均成功。重新建立左臂0:7、双腕RGB、10Hz和可变夹爪合同；不能续用旧右臂134集的Actor/归一化/固定夹爪/20Hz执行假设。保持原RLT Stage1网络与loss，现有EXPO与右臂RLT资产/Replay/默认入口不改；不启停现场服务或运动。

负责人：当前RLT对话。分支train/realtime-expo-rlt-20261015；worktree `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/realtime-expo-rlt-20261015/rl-platform`，初始main497466f。独立环境在同级env，使用只读EXPO环境依赖并在独立overlay补pytest；未改EXPO环境。固定上游保持8cef77e。

## 数据与实际输入：已验证

- 原始49 HDF5/16,421,504,459字节/5931帧；两条deferred/error未作为完整示范使用。原件只读保留Cobot。
- state=`observations/qpos[:,0:7]`，动作=`rollout/coordinator_command[:,0:7]`。只取HUMAN(2)、左臂反馈/协调器/双腕有效且数值有限的连续片段，去9帧边界，剩5922帧；不跨无效片段构造未来动作。
- 沿用既有EXPO分集：TRAIN44集5331帧，DEV5集591帧；DEV native IDs 5/12/20/24/26。该DEV已用于EXPO研究，不是独立TEST。TRAIN实际loader只返回TRAIN，并非仅写split文件。
- 无完全相同的state/action整集哈希、UUID重复；不据此断言没有近重复场景。所有原文件完整SHA及派生NPZ SHA见输入manifest。
- 成功标签来源是本次用户确认，原HDF5没有outcome，不伪造原始标签。人工专家示范不计为自主成功；后续Warmup仅能使用观测成功回报，不能由全成功数据证明Critic能区分失败。片段截断不能每段都当成功终局。
- 原始RGB480×640，用固定上游PIL bilinear保持比例补边到224×224，NPZ无损保存所得输入；未照搬EXPO的320×180拉伸。双腕输入有效，未使用的base槽补0且mask=false。用于RLT的双腕选择是与该任务EXPO输入一致的适配，非原三相机插孔合同。
- 10Hz sample间隔中位约0.09997s、范围0.06436–0.13781s；各集双腕时间差p99最大0.02359s。时间戳保留，未声称严格等间隔或相机硬同步。夹爪动作范围约0–0.0756m，必须保留可变夹爪。
- 数据与输出统一在 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/realtime-expo-rlt-20261015/`。只取431,011,840字节训练派生包，未复制16.4GB原录制；source-inputs.tar和inputs保留校验依据。

## 训练入口与验证

新增 `methods/openpi_rlt/realtime_expo/{dataset,configuration}.py`、`scripts/{export_realtime_expo_rlt,train_realtime_expo_rlt}.py`。调用固定原生train_rlt，不改固定上游；仅针对本repo ID安装进程内dataset hook。数据加载按完整集拆分、未来动作H50，片段末尾保持最后动作（原版padding语义）；六关节相对chunk起始state为delta，夹爪absolute，模型动作pad到32。TRAIN全帧计算独立norm，DEV不参与norm。抽样真实模型预处理通过，绝对/delta往返最大误差1.49e-8。

配置：原π0.5 base `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/assets/pi05_base/params`；H50（10Hz为5秒）、RLT1token/2层/2048、alpha1、原AdamW/学习率日程/EMA0.99；计划5000更新、batch8、FSDP2、seed42、每1000保存。batch8不同于历史右臂batch32，是明确的资源配置差异。上述是准备配置，不是已执行训练。

5项CPU测试通过，覆盖TRAIN/DEV隔离、片段边界padding、数据SHA拒绝、重复UUID、transform不污染输入、六关节delta/绝对夹爪与空base mask。检查真实batch的结果保存在stage1/preflight.json（不能把测试成功当训练/能力通过）。原版抽象CPU state形状：参数15.505GiB、优化器31.010GiB、EMA15.505GiB，共62.020GiB；梯度约15.505GiB另计，未计激活/临时buffer，单48GB不够。CPU只抽象追踪，无模型加载或GPU分配。

失败记录保留：首次共享依赖缺pytest→独立overlay补齐；首次输入往返断言暴露新适配数组引用被DeltaActions原地修改→独立copy修复并加回归；默认下载tokenizer因DNS失败→复用已存在tokenizer并SHA核对，不改tokenizer内容。见import-check、prepare各版本log及tokenizer-copy.json。

## 可恢复入口与未完成项

```sh
B=/data/LFT-W02_data/jiaan/jiaan
P=$B/projects/rl-platform
W=$B/scratch/rl-platform/realtime-expo-rlt-20261015/rl-platform
O=$P/outputs/realtime-expo-rlt-20261015
PY=$B/scratch/rl-platform/realtime-expo-rlt-20261015/env/bin/python
# prepare已经完成，接管先看preflight与SHA，不重复转换。
# 下行仅在双卡归属明确/空闲后执行；启动器会拒绝占用中的GPU。
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=8 "$PY" "$W/scripts/train_realtime_expo_rlt.py" train --assets "$P" --data "$O/inputs" --output "$O/stage1" --base /data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/assets/pi05_base/params
```

唯一下一步是完成已授权HPC四卡迁移核验/Slurm申请，再执行真实小步/保存恢复验证与正式Stage1；不足时调整batch并记录，不改算法挤单卡。之后才可提取同权重z_rl/reference、构建独立10Hz Warmup Replay、训练Actor/Critic并离线验证，最后实现独立左臂10Hz/可变夹爪的RLT部署入口。当前右臂运行器有硬编码，尚未适配，不得把本次新权重将来直接挂到右臂入口。没有自主失败样本/独立TEST，能力评估限制保留。

本批HPC作业已启动，是否完成真实更新以STATUS和运行日志为准；尚无已验证新权重或现场部署，不要使用“训练完成”或“可以Online”描述。

## 2026-10-15 用户追加的交付要求与选型协议

授权范围扩展到实测训练提速、checkpoint离线选型、独立左臂双腕RLT的Warmup/Online接续与Web可选入口同步。用户自行真机rollout；不自动机器人运动，不替换现有EXPO/旧RLT，不覆盖生产Replay。部署不输入中臂/头部相机，真实输入仅left_wrist/right_wrist，base占位为全零且mask=false。

先按固定候选1000/2000/3000/4000/4999比较；最后一步不是自动赢家。若最佳仍在预算边界且DEV持续改善，再有记录地扩展预算；不凭TRAIN loss无限续训。DEV五完整Episode、TRAIN44完整Episode，保留训练见过/重复开发标签，无独立TEST。

离线主指标：C10有效未来帧的逐Episode等权六关节物理MAE；同时报告H50、各关节/夹爪MAE及p95、相邻重规划差、动作二阶差与TRAIN/DEV差距。尾部padding不作为真实误差样本；不把重叠窗口当独立样本。以完整Episode bootstrap描述不确定性，五集DEV结论强度有限。夹爪单列米，不与关节弧度混算。

选择流程：先排除非有限值/输入资产不一致/缺集候选；以DEV主指标最佳为参照，若其夹爪MAE或关节p95相对候选最优差超过25%，标为指标冲突，保留Pareto候选供追加离线检查，不自动宣称最优。主指标差1%以内优先更早checkpoint；连续性作为退化诊断，不以过度平滑换低误差。每候选相同观测、batch、推理步数和逐观测固定noise；先验证单条与batch一致性。

效率：记录编译冷启动与稳定每步用时、GPU利用率/显存、读数据/保存时间；只据实测改变workers/prefetch或卡数，batch变化单独记为训练配置变化。现有4H100 batch32先获得稳定基线，不中断正在执行的小步门禁。

Online交付还需同一选中Stage1提取z/ref、独立TRAIN专家Replay/norm，Warmup多checkpoint比较、Actor导出/恢复与左右臂/夹爪/时基/最终执行动作一致性验证。全成功专家不能证明Critic有失败区分能力，实际自主改进仍待Online新经验。Web明确标注冻结/Online版本与独立数据目录，候选晋升可回退，不静默继承旧任务权重/硬编码。
