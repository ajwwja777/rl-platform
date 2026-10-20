# EXPO SFT2000 50/30/20 HPC 四卡交付事实

2026-10-20：用户要求将新建532模式准备为Cobot本地推理、HPC四卡Learner，验收后交付网页Online采集。隔离更新、恢复、全量权重回传与Cobot推理已经通过；正式Web暂停加载和用户追加的吞吐对照进行中。不启动Session或机器人。

## 配置与隔离

源码e3ca100，基线c1df5a9。新可选ID `expo-plug-left-2cam-532-hpc4-2000`，方法 `realtime-expo-ft-532-hpc4`，原A6000 532入口保持。初始SFT2000，25条已确认成功TRAIN示教/3301步，在线计数、Actor、Critic与VLA更新均0。旧Online121/166/98及其Replay不继承。

固定RTC d5、Edit scale0.01，原生global batch64、UTD20、K30、10有效回合热身、之后5有效回合触发。Critic每1280条实际采样640示教/384在线成功/256在线失败；缺成功或失败池时等待，不伪造结果。基础Actor成功样本筛选、夹爪反馈、HIL与采集接口沿用既有实现。仅新增配置和启动入口，未改学习算法。

Slurm作业740494，ACD1-67，4张H100 80GB HBM3、48CPU、384GB，2026-10-20 10:34:20启动，截止22:34:20。资源与资格仅对应本作业；未来状态以实际Slurm为准。

## 验证进度与边界

- CPU目标回归51通过；完整EXPO回归255通过、8跳过、1失败。该失败在未修改基线c1df5a9同样复现：旧fork迁移测试fixture缺device_layout.py，legacy双digest表达式提前求值；本次fresh SFT路径不调用该迁移工具。排除该既有失败后255通过/8跳过/1取消选择。没有声称全测试通过。
- 四GPU后端/collective通过；隔离HTTP10回合热身不更新，3/5退出恢复、4/5不更新已通过。第5回合触发真实四GPU学习：40Critic/2VLA与每1280条准确640/384/256配额通过。完整checkpoint新进程恢复通过。正式HTTP下载2017658362字节/184.63秒，12次RPC5.86–11.43毫秒，Range与全量SHA核验通过；返回Actor2在Cobot加载预热34.41秒，推理0.218秒且8×7动作有限。
- 验证目录与生产目录分离；合成测试回合及Actor2不进入正式Replay或作为正式策略。没有真机rollout或成功率结论。

## 实际路径

A6000证据：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-532-hpc-20261020`。

HPC产物根：`/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-532-hpc-20261020`。正式状态 `online/`，测试 `verification/online/`。资产和依赖复用10月18日已校验的包；36个适配源码SHA已核对本次版本。

Cobot缓存：`/home/agilex/jiaan/data/expo_ft/plug_left_2cam/native_remote_532_hpc4_20261020`；真实录制目录 `/media/agilex/Getea1/jiaan/data/datasets/plug_left_2cam/recordings/expo-532-hpc4-sft2000`。现场证据 `/home/agilex/jiaan/project/rl-platform/runtime/verification/expo-532-hpc-20261020`。

正式Learner已ready：Actor/Critic/VLA/online/completed均0，Replay3301，热身剩10，四GPU。资格文件 `runtime/expo-remote/532-hpc4-ready.json` 与Web暂停加载待安装。用户追加尽量提高吞吐，正在同一隔离Actor2起点比较原platform内存分配和显存池复用；正式数据不参与测试。

验证脚本首次snapshot_root误指向上级目录，Actor路径守卫拒绝；仅修正测试脚本父目录后加载通过。原始失败日志保留，非产品代码修复。HPC验收脚本沿用模板source_commit字段已按36个实际SHA校正为e3ca100，回执保留原模板值供追溯。
