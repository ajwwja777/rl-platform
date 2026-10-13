# EXPO 在线学习版本 · 2026-10-13

当前：功能实现和隔离验证通过，现场安装/暂停加载待记录。真机成功率与长期稳定性未验收。

## 使用与数据路径

Web 采集页选择 `左臂拔插插头 · EXPO Online（SFT2000起点）`，ID `expo-plug-left-2cam-online-2000`。加载后保持暂停；用户开始采集，再进行 rollout/HIL、暂停/继续并选择成功或失败。结束回合后先保存录制和结果标注，再更新策略、保存检查点；显示“等待下一回合”后才能开始下一轮。每个回合固定策略版本，推理和训练串行使用 Cobot 的一张 RTX4090。没有 A6000 训练进程或网络权重同步依赖。

- 初始权重：`/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_2000`，只读。
- 原生代码：`/home/agilex/xiqing/projects/expo-ft`，只读；运行配置记录源码哈希，源码或训练配置变化时拒绝混用旧在线目录。
- 在线状态：`/home/agilex/jiaan/data/expo_ft/plug_left_2cam/online_from_rtc_sft_2000`；`episodes/` 为独立实际执行记录，`checkpoints/` 为在线参数/优化器/目标网络，`committed.json` 为已提交版本、回合身份和 RNG，`status.json` 为实时学习进度。
- Web 原始录制：`/media/agilex/Getea1/jiaan/data/datasets/plug_left_2cam/recordings/expo-online-sft2000`，和原 SFT 数据、RLT Replay 分开。
- 原固定 SFT2000 网页选项保留。切回固定项即只推理；再次加载 Online 会恢复最近已提交版本，不会默认重置训练。

## 数据与学习语义

左臂7D绝对关节/夹爪位置；两路腕部 RGB，非活动 base 图像槽 masked；逻辑10Hz，H16/r8，RTC延迟2步。Hz选项仅改变发布/插值频率，滤波仍可选；Online 的 RTC 必须开启，以匹配延迟训练输入。继续沿用原模型的逐步限幅、暂停/HIL保护和失效推理结果丢弃。

训练动作来自 `/master/joint_left` 的实际协调器命令，包含 HIL，配对前一观测和下一观测。模式切换、暂停、缺帧/过期与时间间隔异常会断开序列；不伪造跨中断动作。回合结尾保存实际终止观测，使用原生 Replay 的完整窗口；过滤 valids=0 的截断窗口，避免编辑网络对填充动作训练。

只有明确成功/失败的完整回合参与学习。成功末步 reward1，失败 reward0，真实终止 mask0；未知、放弃与中断恢复的不完整日志不加入 Replay。基座 Actor 沿用原生 success-only 训练；成功回合中的 rollout/HIL 均可入其 BC 样本池。尚无成功样本时，Critic/编辑网络等仍可更新，基座 Actor 保持不变，不把失败 HIL 伪标成专家成功。

当前配置：batch4、UTD1、至少64条真实转换开始训练、每新增25条转换获得一次更新额度，每回合最多8次更新；Replay 容量6000（含每回合一个终止观测占位），按完整旧回合淘汰内存样本。原始日志保留，磁盘不足16GiB时拒绝新回合。未导入历史离线示教，生产 Replay 从空开始。该保守调度是功能起点，不代表已选出最佳超参数。

## 内存适配与检查点

保留原生 RealTimeEXPOFTLearner 的 N8、Q ensemble10、edit8、filter8、损失及更新顺序，调用其 `_update_jit`。适配层消除 self/agent 重复输入和初始目标网络别名后进行 buffer donation。目标 Actor 在当前源码中只维护 EMA、未被损失或推理读取，因此将可训练目标 EMA 分块放 CPU，冻结部分与种子共享；源码检查防止把此适配直接用于已改为消费目标 Actor 的实现。未降低参数精度或缩减网络。

更正最初摘要：“LoRA配置”不等于“仅 LoRA 参数训练”；当前原生 trainable filter 也包含视觉模块等参数，适配保留这个范围。

原生 Orbax 整体保存曾触发主机内存保护，最终在线运行使用有 SHA256 的分块状态检查点：完整保存可变参数、优化器、目标网络和计数器，冻结参数引用只读 SFT 种子。逐文件 fsync 后发布目录，再提交元数据，保留最近两次已提交检查点。**此在线格式不是可直接交给原生 Orbax restore 的单独权重包**；应由本适配加载，并保留原种子与 norm_stats。不能只拷一个 safetensors 文件声称可恢复在线优化器。

## 验证证据与边界

- 原生未经内存适配的 batch2 更新 OOM，保留此前失败证据。
- 优化后 batch4 连续5次完整更新通过，Actor/Critic参数改变、全部标量指标有限，热态更新约2.05–2.06秒；更新后推理110ms。含保存的测试 GPU采样峰值19012MiB，主机可用内存最低约16.5GiB。
- 分块保存约11秒；新进程恢复 Actor/Critic step5、RNG，与保存前相同输入/随机种子的动作最大差0。
- 合成观测和实际命令接口模拟的 rollout/HIL 回合：未知回合排除，成功回合更新0→2；新进程恢复 Replay/优化器/RNG 后继续2→4，幂等结束不重复训练，新回合在更新期被拒绝；恢复续训采样 GPU峰值19027MiB。
- 无成功样本：失败回合仍完成两次原生非基座 Actor 更新，基座 Actor step0；恢复验证另行记录。
- Python回归：RL24项；Web58项通过、1项环境跳过；前端29项通过。包含暂停/HIL旧结果失效、实际动作、终止观测、录制后标注/学习顺序、RPC失败重试、本地HTTP身份检查。

测试均为隔离输出，无 ROS 运动发布、无真实机器人动作、无真实录制或生产 Replay 写入。JAX 首次训练或恢复后可能重新编译数分钟；上述热态推理不是连续真机 deadline 验收。没有声明与原生整图执行逐位数值相等，也没有将合成数据更新通过视为成功率提升。

可复现脚本：`scripts/verify_expo_online_lifecycle.py`，以原生私有 Python 运行，传独立 `--output`、原生 `--project` 和 `--bundle`；可分别 `--outcome success/failure`，完成首轮后使用 `--resume`。脚本不导入 ROS，拒绝覆盖已有非验证 run。

完整现场测试证据：Cobot `/home/agilex/jiaan/project/rl-platform/runtime/verification/expo-single-gpu-20261013/`；A6000 `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/expo-online-20261013/`。任何模拟检查点都不用于生产初始化。

## 实现入口

- RL：`native_online.py`（原生配置/Replay）、`memory_update.py`（捐赠/CPU EMA）、`stream_checkpoint.py`（有界保存恢复）、`online_learner.py`（回合版本/学习/事务）、`episode_journal.py`（实际动作与 HIL）、`online_protocol.py`（身份/数据协议）、`online_runtime.py`（ROS/HTTP协调）；均在 `methods/expo_ft/cobot_adapter/`。
- 启动与登记：`scripts/expo_online_up.sh`、`configs/deployment_models.json`。原 fixed `ros_runtime.py` 仅添加可覆写入口，固定模型回归通过。
- Web：`collection_model.py`、`deployment.py`、`execution_options.py`；`segmented_capture/api.py` 与采集/执行选项 JS。通过本机HTTP交接，不反向 import Web 或复制其录制实现。
