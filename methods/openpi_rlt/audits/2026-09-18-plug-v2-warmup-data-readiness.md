> 后续修复已完成，正式训练未启动：见[修复与验收](2026-09-18-plug-v2-warmup-replay-repair.md)。下文保留修复前只读审计，不代表当前脚本仍有相同阻塞。

# plug_v2 warmup 数据与训练前审计（2026-09-18）

本轮只读检查数据、代码、Session状态及GPU，无训练/资源申请/机器人动作；未访问HPC。

## 已核验数据

Cobot根 `/media/agilex/Getea1/jiaan/data/rlt/plug_v2/warmup`：95份终局rlt事实，55条成功（全部有HIL）、20条自主失败、20条放弃；75条success/failure均有validated官方LeRobot转换。主目录HDF5=0，incomplete=0；历史.failed文件不计训练。75份replay文件仅.source.npz事实，不是已提取特征。无learning/warmup/ready.json或actor.pt，新cohort warmup尚未训练。

转换后training_mask：成功组13467录制帧、8596有效帧、5133有效HIL帧；失败组2931录制帧、1742有效帧。共10338有效帧，6060暂停/无效帧排除；这些是转换后的mask数量，不混同trace的policy_frames+hil_frames=10322。按当前stride2/delay6构造，成功3993、失败812候选replay行（含尚待修正的边界行），不能把10338图像帧当独立transition。

当前网页Session phase=stopped，policy_paused=true，actor_mode=reference，actor_version=-1。4090显存18908MiB使用/5306MiB空闲，util0%；预加载Stage1在内存。状态是本次动态快照，不承诺训练时仍空闲。

## 训练前阻塞与限制

`projects/rlt/methods/openpi_rlt/plug_v2/replay.py`当前按有效/generation切段，但next_z由全局anchors的searchsorted取得；done仅以整轮最后有效帧决定。只读重算发现288个非终局候选的后继越过本片段或进入其他generation。应修正bootstrap边界，区分真实终局与暂停/接管截断，不给截断伪造失败奖励或跨片段Q后继。当前未修改该逻辑，不交付可直接训练结论。

另需审核HIL成功整轮BC：learning.Sampler目前仅按episode组生成successful，update因此也会模仿HIL介入前的策略动作；建议正向BC重点使用专家和HIL纠正动作，失败/错误自主动作用于Q学习，不能因为最终HIL成功就全部作为正确动作。

RTC replay目前统一offline_teacher_prefix（由未来动作构造），虽然source trace已存真实plan.prefix/context/z/ref，prepare未读取这些factual计划。训练前应核对自主rollout的真实承诺prefix与决策/动作/奖励的时间轴，不能把fixture通过等同于正式rollout的时间轴已验证。

success.labels的keep=true，failure.labels的keep=false/quality=bad是当前记录事实；本版RL准备按validated转换与training_mask筛选且明确包含失败组，不以该SFT保留字段排除全部失败。需保持录制损坏与任务失败的区别，不盲改标签。

## 建议训练结构（未启动）

冻结同一joint Stage1 3999的Pi05/token encoder/decoder；actor从零残差、critic从新初始化，不加载旧相机warmup。保留82条原成功专家（坏帧切分后83片段），三组采样30%专家/40%新成功/30%新失败，按episode均衡。这是采样比例，不要求采集数量恰好相同比例。

按原UUID固定留出，专家沿用已有74/8原UUID划分；新rollout可增加至约20%留出（55成功留11、20失败留4），剩44/16训练。当前未创建split_registry，可在首轮优化前定好；后续不得为了指标重新分配。

沿用batch128、actor/critic LR1e-4、每控制步gamma.99、tau.005、critic:actor2:1、reference dropout.5、BC10/Q.1/smooth10，RTC与命令条件化和reference/frozen/online一致。最多20000critic/10000actor更新，分段验证选优，不默认最后版本。建议先2k审核，再5k/10k/20k；对比留出专家/HIL动作误差、TD、动作速度/加速度及边界连续性，真实提升最终用独立固定场景自主评测。

优先用Cobot现有4090训练小RL模块、复用预加载模型提取特征并按Stage1/UUID缓存；不用为小模块默认重申请HPC。完整耗时需包含首次特征提取，旧优化器基准不能直接当端到端时间承诺。修正/验证以上阻塞后再交付warmup.sh启动结论；当前请勿据此自行启动旧warmup脚本。
