# EXPO Cobot 单卡在线学习交付 · 2026-10-13

状态：用户已授权继续实现；单卡内存适配、连续更新、检查点恢复与模拟回合续训通过。现场安装与暂停加载待完成。当前实现和证据见 [RESULT.md](RESULT.md)。

以下保留最初原生直接更新失败的历史记录，不能代替后续优化结果。

用户要求先验证全部使用 Cobot 显卡的同步方案。测试使用 `/home/agilex/xiqing/projects/expo-ft` 的原生 `RealTimeEXPOFTLearner`，从 RTC-SFT2000 参数初始化在线 Learner；在线优化器从0开始，不把 SFT 优化器步数当作在线更新数。

## 实测结论

Cobot RTX4090（24GB 标称）可以初始化完整 EXPO 并推理，但当前原生完整梯度更新在 batch size2、UTD1 时 OOM。不能据此声称 Web 已能在线学习，也不能把同步调度等同于显存问题已解决。未验证过的内存优化版本仍可能可行。

| 检查 | 结果 |
|---|---|
| 原生初始化 | 通过，约17.6秒 |
| 普通推理热身后 | 两次约104–105ms |
| RTC候选生成 | 两次约107–108ms |
| RTC编辑与Q选择 | 两次约4.3–4.4ms |
| 默认JAX分配器下完整更新 | 第一步OOM，申请2.25GiB失败 |
| platform分配器、MEM_FRACTION0.98对照 | 第一步仍OOM，申请510.68MiB失败；采样峰值23868MiB（约23.31GiB） |
| 模型恢复 | 原SFT2000已恢复，ROS PID2164624，paused；50Hz/RTC开启/滤波开启，无活动回合或录制 |

推理耗时仅为同一保存观测的短时热身测试，不是持续真机 deadline 或成功率验收。内存峰值由 nvidia-smi 约0.5秒采样，可能低于瞬时峰值。默认分配器测试峰值17220MiB；其分配上限不能作为物理卡容量的结论，因此补做了直接释放显存的 platform 对照。

## 测试边界与配置

- 源码、权重：xiqing 原目录只读；种子 `/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_2000`。
- 保留原生 N8、Q ensemble10、edit samples8、filter_N8、Critic/encoder/edit/temperature/base Actor 更新；双实际腕相机，左臂7D，H16/r8/delay2。
- Actor 使用 LoRA 可训练参数，目标 Actor 从相同 SFT 参数初始化；没有为本次测试缩小网络、关闭基座 Actor 更新或替换算法。
- 独立96容量 Replay，64条合成转换，batch2/UTD1；测试性成功标记仅用于触发 Actor 梯度路径，不是现场成功数据，不进入任何生产 Replay。
- 推理、更新串行；无 ROS 导入或机器人发布器。两次都未完成第一步更新，没有生成或发布新权重。
- 通过网页暂时释放原暂停模型后独占GPU；测试退出后通过原网页加载接口恢复同模型与执行配置，保持暂停。

## 证据

A6000：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/expo-online-20261013/single-gpu-probe/`，含 probe.py、native_online.py、monitor.py；`field-evidence/` 内保留两次 result.json/probe.log/memory.jsonl/exit.json、恢复前后状态和六个原生源码 SHA256。

Cobot：`/home/agilex/jiaan/project/rl-platform/runtime/verification/expo-single-gpu-20261013/`。

在线接入的两份未提交原型保留在独立 worktree `feature/expo-online-20261013`，未装入生产；A6000 权重复制仍按此前暂停要求挂起。当前缺口是完整更新的内存峰值；单卡内存优化或跨GPU方案尚未进行后续验证。
