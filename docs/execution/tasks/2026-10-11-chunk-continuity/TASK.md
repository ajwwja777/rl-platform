# RLT chunk衔接与命令平滑

状态：2026-10-12 低频探索候选已实现、CPU验证、提交push并安装；正式网页可见。下一步验收LOW_FREQUENCY，不再优先叠加两级EMA。真机平顺性、接触性能与成功率仍证据不足。

授权与范围：用户明确优先执行基建，先在RL项目实施，autoresearch并交给用户rollout。固定Actor/Reference/RTC/50Hz/探索分布，不训练、不自动运动；共用VLA仅只读，不改其他模型或生产默认、权重、Replay。可选新入口采用独立录制和Replay。

负责人：当前RLT执行对话。基线5960622；分支fix/chunk-continuity-20261011；worktree `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/chunk-continuity-20261011/rl-platform`。

验收：实际RTC/50Hz/EMA证据；旧滤波器重构匹配已发布命令；同目标序列候选降低加速度且报告跟随滞后；暂停/HIL/终止重置、限速、实际发布动作进入Replay、旧入口回归；可选、可回退。真机前不得以离线平滑曲线认证任务成功。

资源：仅CPU/数值trace；A6000输出`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/chunk-continuity-20261011`。现场同步只在锁内空闲核查后进行，不启停服务、不加载模型/不运动。故障/忙碌/身份冲突时停止同步；用户负责实际加载与rollout。

唯一下一步：用户固定同一插孔位置和初始位姿进行少量旧/新入口对照；若更迟钝、接触更差、跟踪冲突或超时，切回旧入口。记录动作、反馈、HIL、自主/辅助结果，再决定是否保留或推广。此前离线BC/Q训练暂缓，先验收执行基建。

详细证据和交付：[RESULT.md](RESULT.md)。


## 2026-10-12 用户调整优先级：动作一致性与生成连续性

两级EMA候选保留历史安装/回退证据，但不再作为优先验收方案。只读核对发现20Hz逻辑action仅代表每段最后50Hz命令；5327逻辑步对应13291发布，中间7964条仍在trace但未逐条进入训练action。详见[ACTION_CONTRACT](ACTION_CONTRACT.md)：原RLT、Training-time RTC、XR-1和Real-Time EXPO-FT原文/代码核对，以及探索时间结构的证据边界。下一步是明确最终动作/时间契约与前缀条件训练的最小隔离方案；本次没有改代码/现场或训练。


## 2026-10-12 低频探索交付

最新[交付与验收](LOW_FREQUENCY.md)：0b9dd4a新增episode_constant，探索幅度/RTC50Hz/单级EMA/Actor3640不变；每Episode一个偏移并1秒平滑进入，Learner关闭。29历史Episode开环命令加速度RMS中位−9.86%，末3秒命令路程−19.40%，不能证明闭环抽插已消除。四源码/登记文件现场SHA及CPU检查、正式网页可见通过，没有加载/运动/服务启停。

唯一现场下一步：用户固定位置交替验收旧探索与新低频探索，实际发布/Replay身份核对后决定保留；新入口plug-v3-hil8-explore-constant。算法匹配结果另见[Online RESULT](../../../online/tasks/2026-10-11-online-learning-evidence/RESULT.md)，二者不混作成功率证据。
