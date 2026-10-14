# Teacher5：恢复后首批 Online 学习审计（2026-10-14）

**已验证：数据进入了真实采样和更新，Critic 在新增数据上的行为回报残差下降，Actor 有小幅变化；证据不足：新 Actor 自主成功率提升。** 本次只读审计和 CPU 前向，无新训练、发布、Replay/服务/机器人修改。现场后来由其他对话切换 EXPO，本报告不描述 EXPO 学习效果。

## 数据、版本与实际配置

- A6000 项目 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`；产物 `outputs/teacher5-online-review-20261014/`。采证起点 main `dd3a1ee`（Teacher5 现场修复 `6867ac6`）；写报告时 main 已推进至 `f421ef5`，保留其他会话 EXPO 改动。运行训练源逐文件 SHA 位于 `training-config-metrics.json:profile.training_source_sha256`，不是以当前 main 冒充运行代码。
- Cobot runtime `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013`；录制 `/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/teacher5-8127-online-v1`。
- TRAIN：实际 Replay 3,730 windows / 235 Episode；旧数据 231 集，今日 4 集。重复选型 DEV：20 集 / 370 windows；实际 `(phase, episode)` 身份与当前 Replay 无交集，但不是独立 TEST，也不能据此排除 Stage1 历史使用。独立 TEST 缺失。
- 实际配置：C10、delta_chunk、gamma=0.99、batch128；BC5、Q0.1、delta10、teacher5、MC0.3；Actor lr=1e-5、Critic lr=1e-4、tau=.005、Actor 每2次更新一次、reference dropout=.5、std=.002。固定夹爪，诊断只计算6个活动关节。CPU Learner；Stage1/在线 Actor GPU。50Hz RTC/滤波为本轮登记执行方式，不以配置代替物理时序验收。
- 更新预算：每新增 transition 1次 Critic 更新；候选 staged，不自动晋升。20:55 采证：served Actor4063/step8127，pending Actor4117/step8234。
- 用户补充：插头与夹爪相对位置稍有变化；不能将这3轮视作严格同条件重复，更不能据其结果归因新 Actor。

| 原始 Episode | Replay Episode | 标签 | 入库 windows | HIL |
|---|---:|---|---:|---:|
| 1（先前故障前） | 10040 | 失败 | 18 | 0 |
| 2（恢复后） | 10041 | 自主成功标签 | 29 | 0 |
| 3（恢复后） | 10042 | 失败 | 23 | 0 |
| 4（恢复后） | 10043 | 自主成功标签 | 37 | 0 |

成功依用户结果标签且无 HIL 记录；未逐帧人工复核视频。录制 UUID、标签时间、每集终止与窗口身份见 initial.json。89个重叠窗口不是89次独立尝试。兼容标注 failure→bad/keep=false 不是独立质量复核，不能因此删除有效失败 RL 经验。

## 实际更新与采样

**已验证**：107条连续唯一更新 8128–8234；其中首轮故障恢复18次，新增3轮89次，新增 Actor 更新45次。全部日志数值有限，最终预算0。新增3次更新突发分别29/23/37次，首末 metric 间隔1.20/0.72/1.28秒；结果标签到末次更新约11.44/7.43/12.10秒，多数等待在学习前。尚无完整阶段 trace，不能把这段等待精确归因某个特征提取或磁盘操作。

| 互斥来源 | 最终 Buffer | 新89次 Critic 实际抽取 | 45次 Actor 实际抽取 |
|---|---:|---:|---:|
| Warmup 专家 | 31.80% | 34.70% | 32.43% |
| Warmup rollout | 31.93% | 21.35% | 20.14% |
| 较早 Online | 33.40% | 38.02% | 42.05% |
| 今日第一轮失败10040 | 0.48% | 1.39% | 1.28% |
| **新增3轮10041–10043** | **2.39%** | **4.55%** | **4.10%** |

Critic 11,392次抽取 / Actor 5,760次抽取，重复抽取不是独立样本。最近20集配额并非最新3集配额；最新数据已被使用，不能称被完全忽略，但占全部训练的比重仍低。最终 Buffer 只能作末态参照，不能替代每次更新的动态池分布。

- 按完整 Episode 结果归类的成功来源抽取比例：Critic69.14%、Actor70.78%；失败30.86%、29.22%。没有固定80:20。
- human/expert 标记抽取42.48%/46.18%，与上表来源重叠，不相加。额外 Actor HIL8 槽从旧近期 HIL 池抽取，新3轮没有 HIL，不能称模型学到了这3轮的新人工纠正。
- 行级 `sample_success_ratio` 约5%是终末成功行，不是 Episode 成功率。Actor 非更新步的批次不计入 Actor 实际训练比例。

![数据构成与实际抽样](../../../../outputs/teacher5-online-review-20261014/sampling.png)

## 同一输入上的 checkpoint 对比

5个完整 checkpoint：8127/8145/8150/8200/8234，冻结 CPU 前向；Q1、Q2、min-Q分开保存。Episode 内平均后对 Episode 等权，DEV配对 bootstrap 按完整Episode抽取10,000次，seed42。图中末3窗口仅指时间末端，不声称语义阶段；3个新Episode仅作描述统计。

| 指标 | 上机初始8127 | 恢复后8145 | 最新8234 | 判断 |
|---|---:|---:|---:|---|
| 新3集 TRAIN：record-action Q1 对观测回报 MSE | 0.19195 | 0.14019 | 0.06553 | 已验证：训练内残差下降 |
| 新失败10042：末3窗口 record Q1 | 0.84166 | 0.76701 | 0.51807 | 已验证：失败评分下调；仍有残差 |
| DEV20：record-action Q1 对观测回报 MSE | 0.04717 | 0.04854 | 0.04165 | 已验证：重复DEV指标下降 |
| DEV20：六关节 recorded-action MAE(mrad) | 0.92055 | 0.91581 | 0.91144 | 已验证：均值未退化，不代表精度/成功率提升 |
| 新3集 TRAIN：J2 recorded-action MAE(mrad) | 2.69178 | 2.52924 | 2.39269 | 已验证：模仿误差小幅下降 |

最新 Actor 相对上机初始，在新3集 J2平均绝对变化0.652mrad，其他活动维约0.035–0.444mrad；DEV J2变化0.270mrad。变化可测，但不是巨大修正，也不能由拟合判定纠正方向物理正确。

新失败10042的观测回报为0；最新全程 record Q1=0.388、末3窗口0.518。两成功集末3窗口Q1约0.842/0.831、观测回报约0.893/0.881。失败/成功分离有所改善，但只能说明已见数据的结果拟合；同状态 Actor/Reference 的Q差并无替代动作实际执行结果验证。

本批实际 reward 合同核对为单次终末1、其余0，gamma=.99；沿观测轨迹去重重叠窗口后 return∈[0,1]，全部255个TRAIN+DEV Episode满足终末/reward/重叠一致性与完整回报重建。此范围只适用于这个数据合同；Critic 的目标策略回报不等于某条随机行为轨迹的一次观测回报，失败样本Q不为0不能单独证明算法bug。

DEV Q1-MSE配对差95% Episode bootstrap区间约[-.01137,-.00024]；min-Q差区间[-.00565,.00089]跨0。DEV已多次用于选型，区间仅是这20集的描述性不确定性，不是新泛化证明。固定夹爪不纳入误差、退化结论。

![固定输入权重对比](../../../../outputs/teacher5-online-review-20261014/checkpoint-comparison.png)

![真实训练曲线](../../../../outputs/teacher5-online-review-20261014/training-curves.png)

Critic日志loss范围0.00234–0.01038，末值0.00312；没有数值发散迹象。loss低或Actor自评Q高不是放行依据。梯度范数本批未记录，不能声称梯度检查通过。加权loss项大小不等价参数梯度贡献。

## 结论与剩余边界

- **已验证**：记录→Replay→真实batch→更新→checkpoint/pending链闭合；本批结束是用户正常停止，未重复之前 Learner SIGSEGV。长期稳定性仍需更多运行证据。
- **已验证**：新数据有实际学习响应、旧重复DEV平均指标保持，保存8127/8145/8150/8200/8234可追溯。读前后Replay/latest/served/pending四资产SHA一致；分析未训练或晋升权重。
- **证据不足**：新候选自主成功率、位置/抓持泛化、同状态动作级Critic正确性。本次3轮都执行旧Actor4063，不能把2次成功归因未发布的4117。当前没有新的HIL效果证据。
- **证据不足**：以最新8234立即替换上机权重的完整门槛。本次评估不是完整尾部/发布一致性/冻结真机验收。失败Q仍有残差，不能宣称Critic全面正确。
- 后续建议（不是已执行流程）：保留8127与8234对照，以固定抓持/插孔条件分别做小批冻结比较，再决定是否晋升；后续采集按位移/抓持条件标识，保留自主失败与HIL，持续核对新增集实际曝光。当前现场已由其他对话转EXPO，先保持该任务归属，不能混用两算法数据或自动切回Teacher5。

## 产物与复现

A6000完整目录：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/teacher5-online-review-20261014/`。

- initial.json：20:55现场采证，含录制、标签、runtime、served/pending和日志尾。
- training-config-metrics.json：实际profile、107条metrics、batch样例；analysis.json：全部实际batch身份聚合、逐Episode逐checkpoint结果、SHA保护收据。
- input-identity.json：DEV路径、SHA、分析脚本SHA；summary.json：Episode等权摘要和DEV配对区间；final-web-status.json：随后现场已为EXPO paused，不将此状态冒充Teacher5。
- run_analysis.py / analyze_worker.py：CPU只读前向，env显式CUDA空/JAXcpu，2CPU affinity，150秒上限；summarize.py：A6000制图。

复现命令（先核对现场与冻结资产身份；脚本会读取当时最新Replay，不保证未来仍是本快照）：

```bash
cd /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform
.venv/bin/python outputs/teacher5-online-review-20261014/run_analysis.py
/usr/bin/python3 outputs/teacher5-online-review-20261014/summarize.py
```

本次 numerical exit0；所有预测有限、TRAIN/DEV身份无交集、107step连续、255Episode回报重建通过。制图首次用项目venv缺matplotlib，改用已有/usr/bin/python3成功，未安装或更改环境。报告/索引属于事实记录，不修改正式运行流程。

### 核心资产 SHA256

- Cobot `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/replay/replay_journal.pkl`：`d2a74cfce3f3e77e4ff23704dc09e0c9f541988acc014a53b030af0310e109c6`
- Cobot `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/checkpoints/latest.pkl`：`3d85d142f231ae1251be9f5f4755b220e19f50962602376845e7a006173104d6`
- Cobot `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/actor_snapshot/actor_snapshot.pkl`：`8a59565d2ba4538d13df2d49a047b8482707126dd38dacfe34a4e55ce4958f9f`
- Cobot `/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/pending_actor/actor_snapshot.pkl`：`08dd512f78bbedd7fa1ca7a76db9d9e1ad6e8857236f3ab3f2177f1ecde81ae6`
- checkpoint step8127 / Actor4063：`18b0dfa82e2c2978df524535eb03c0457a4afc3c97b469f89ab57c87a8a825d5`
- checkpoint step8145 / Actor4072：`791bc1579e821e2bbb0048b4a10eaa47e57db879eb3d4acfebbce0f9f6ae4d66`
- checkpoint step8150 / Actor4075：`5b7d386e1896272a7ad53133fff61d97b43a0f20fd8a47a869c50da391987e12`
- checkpoint step8200 / Actor4100：`b2e5160c15b9e2ca7d27256be073b3babe85f648b989f7260683232f0ed0230d`
- checkpoint step8234 / Actor4117：`3d85d142f231ae1251be9f5f4755b220e19f50962602376845e7a006173104d6`
- DEV：`edf3f3bc3d6315fbcb55d58ce1ad21a71c18e730fed62407f92d677d3f1fca21`。
