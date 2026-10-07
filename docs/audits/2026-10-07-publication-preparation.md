# 2026-10-07：Online 控制发布准备与故障证据修复

结论：确定的调度开销问题已修复，代码在A6000提交推送并同步Cobot，现场实际模块CPU测试通过。真机恢复/持续发布尚未执行；不能将合成时钟通过称为真实50Hz或成功率通过。现场模型仍error/运行进程退出，用户下次恢复才进入新代码。

## 证据与修改

|项目|状态|证据/边界|
|---|---|---|
|正常准备开销被逐次累积|已验证|旧代码wait_until后还有控制权sample、重复sample、滤波、限幅，再把全部准备迟到加进后续时钟；可提前做的工作也占用C10累计50ms预算|
|可重复复现|已验证|固定合成采样.5ms、滤波1.3ms、安全检查.1ms、发布.5ms；旧40Hz第8逻辑步/50Hz第7步累计51.2/51.0ms退出。不是对现场全部耗时的精确归因|
|修复后同负载连续执行|已验证|20/30/40/50Hz各120逻辑步，命令数120/180/240/300，无跳步、无压缩物理间隔；旧40/50不能完成。合成50Hz间隔中位22.4→20.6ms，非现场测速|
|旧物理限速检查顺序|失败（已修复）|旧实现publish之后才检查安全clamp是否破坏相邻命令限速，回归证明异常目标先发出；现移到publish之前，零命令发出即停止|
|真实安全与抢占边界|已验证（软件）|单次/每C10累计50ms、200ms停顿、sleep过冲、暂停/HIL/terminal、陈旧RTC结果、限幅和拒绝发布回归通过。未放大阈值|
|故障partial命令留证|已验证|先暂停，再向模型进程日志写[rlt-execution-fault] JSON：已发物理命令、未完成发送尝试、版本、时刻、累计漂移。不构造Replay逻辑行。发送调用中异常的published=null表示未知|
|现场连续rollout|证据不足|没有机器人动作、Stage1前向或运行恢复。剩余CPU/ROS抖动仍可能触发真实超时，新的partial证据用于定位|

生产源码只修改`methods/openpi_rlt/cobot_adapter/async_execution.py`：
- 插值、因果EMA和夹爪速度约束提前到等待前。滤波器已由fresh_plan以实测状态初始化，后续依赖上次已发送目标；发送前仍用最新反馈做安全限幅。
- wait_active返回刚验证的反馈，去除发送前重复读取；保持等待前/后的权限检查及I/O原子拒绝。
- 物理限速检查移到发送前。
- 保存prepare起止、control_sample、safety_check、post_wait lateness、publish成本；partial fault单独写日志，不进入Replay。

## 验证

基线d808b4994e95b2cd2c5bc0ba78a800bff6b757dd。最初6项新回归在基线4失败/2通过，修复后通过；后增加两项发送状态未知和准备中暂停抢占回归。A6000 Python3.11相关125通过（19.57s），含真实EnvDriver/CollectionTrace/strict输入身份和临时Replay两次学习组合；Cobot冻结Python3.10对实际安装模块48通过（17.24s）。计数包含交集，不相加。所有I/O与权重为合成/临时CPU测试，不调用机器人/现场推理服务，不改固定上游。

完整命令：输出`validation-command.json`、`install-status.json.test_argv`。A6000 CPU172–175最多4线程，Cobot CPU20/21最多2线程；CUDA_VISIBLE_DEVICES为空、JAX_PLATFORMS=cpu。合成前后对照可复现脚本`compare_scheduler.py`，数据`synthetic-timing-comparison.json`；输入成本明确写在脚本及新回归中，无真实成功率数据。

## Git、现场应用与回退资产

源修复提交：`57b378f67d314e2557fe0318d7d6709c332be0d5`，分支`fix/publication-preparation-20261007`，先A6000提交push后同步。
现场模块：`/home/agilex/jiaan/project/rl-platform/methods/openpi_rlt/cobot_adapter/async_execution.py`。
- 旧SHA `7a46080d2e7ce23ecf48e09bb698947843842d77dbce6430e95cdc6c4bc6a4d9`。
- 新SHA `b04a782b7b86cfd334e39160d150c925b48fe437f846b369f17d1bf5751e71e5`。
- 旧文件保存在`/home/agilex/jiaan/project/rl-platform/runtime/verification/publication-preparation-20261007/async_execution.before.py`；如回退源码，仍需确认运行进程退出后执行，不能覆盖活动执行器。

同步前后都确认deployment error、运行进程退出、无活动操作/Session、录制stopped/complete，模型操作锁内仅替换上述已退出进程使用的模块，执行实际环境CPU测试；失败会还原原件。Stage1 PID2073785/start_ticks8759073保持。完整资产保护SHA见install-status.json，包括配置、两个候选Actor/norm、teacher/profile、生产Replay和latest/7281。Replay仍`7ec6da3ac6ffd66a1074b87444400cff46860c58018578997f7d1bad1889d0a5`，最新checkpoint仍`da3dda177f204bd1fced04ca566c715c9e209e36bc0747f9fa34f591f548f45e`，served Actor仍3500。没有发布3640、重启/加载模型或服务、执行运动，也没有更改50Hz/RTC/EMA、采样、训练参数。

## 对下一步学习的判断

已有Online数据并非只能画图：可在固定快照上做训练对照、历史到达重放、梯度/动作/Q诊断、完整状态恢复与导出一致性验证，先排除数据错误和显著退化。离线回报/模仿指标不能识别未执行动作在真实接触任务中的结果；没有可信且验证过的任务模拟器或充分覆盖的数据时，自主闭环收益最终仍需少量受控真机验证。

因此后续采用“离线定位和筛选→小批冻结比较/纠正采集→按批训练审核→明确发布或回退”，而不是边改多项参数边无限Online。当前权重不变；新Episode均衡六组有Critic保持代价，仍未采用。原下一轮策略见`2026-10-07-online-rollout-strategy.md`。此次可进入修复后的受控运行检查，不等于允许跳过故障观察直接长时间无人值守Online。

现场恢复使用已有网页运行恢复入口（保留Stage1），由用户手动操作；本任务未调用该POST。机械臂重新上电后的硬件检查按现有现场流程；源码同步本身不会使机器人动作。恢复后首次短轮重点检查是否正常结束、发布时序、Replay提交与训练版本，再按既定位置/HIL方案收集。

## 产物

A6000：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/publication-preparation-20261007/`，含baseline-tests.log、related-tests.log、field-tests.log、source-manifest.json、validation-command.json、synthetic-timing-comparison.json、guarded_install.py、install-status.json、REPORT.md和PROGRESS.json。
现场验证：`/home/agilex/jiaan/project/rl-platform/runtime/verification/publication-preparation-20261007/`。
