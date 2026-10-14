# EXPO 初始 Online 抖动诊断 · 2026-10-14

状态：已定位离线动作段中主要增量来源并交付独立低扰动本地预览；真机平顺性尚未验收，原 Online 配方未修改。

## 现场事实与范围

用户报告第三个 native-remote Online 明显抖动，而 RTC-SFT baseline 平滑。两次已录制 Online journal 均 actor_version0，尚无梯度更新：3de226f7-7142-4680-86b5-7361fd8eeb09（125帧，aborted）、b7dc20d9-b7ac-4f74-8ea9-b2ccfa818e29（214帧，unknown）。不把这两轮重新标成成功/失败，不修改原 Replay。随后再次加载失败的直接原因是 LFT1 启动器报 Reserved EXPO GPU is already in use；未停止其他 GPU 任务。

本次复用原任务 worktree 和 outputs/expo-native-remote-20261014/verification/jitter。Cobot 显卡空闲、Web error/无Session/无录制时运行独立无ROS推理；无训练、机器人动作或服务重启。

## 源码与对照

原 native 在线从版本0就用32基础候选+32编辑候选、未训练目标Critic选最大Q。前10回合只禁止梯度更新，并未禁止探索；这与收到的 train_pi_robo.py / AsyncChunkSampler 调用一致。edit_scale0.1来自 dynamic_pick/run_server.sh；它是归一化动作扰动，不能直接解释为物理动作10%。Cobot全部7维参与编辑，不采用DROID的xyz旋转mask。

相同实际画面/状态、相同物理RTC前缀、相同32个VLA候选和选择RNG，比较base第一候选、Q-only、Edit0.1/0.03/0.01。两个Episode各均匀选8个窗口；delay0/2/5，每个delay16个窗口、总240次选择。恢复现场版本0Actor快照并逐叶SHA校验。所有组使用现有EMA08和每逻辑步0.03rad/0.004m限幅，没有新增滤波器或改动上游。脚本53bc47b，methods/expo_ft/experiments/jitter_probe.py。

| Candidate | Limited second difference RMS (mrad) | Reversals per 8-step window, summed over six joints |
|---|---:|---:|
| base | 2.251 | 0.1250 |
| q_only | 2.682 | 0.0625 |
| edit_010 | 16.016 | 7.4375 |
| edit_003 | 6.710 | 1.1875 |
| edit_001 | 3.364 | 0.0625 |

表中统计为RTC5；显著反向要求相邻两次关节增量均大于0.005rad且符号相反。Edit0.1的有限幅二阶差分为Q-only约5.97倍；Edit0.01相对0.1降低约79%，显著反向从7.4375降到0.0625/窗口。delay0/2出现一致趋势。因此编辑扰动是这些窗口内粗糙动作的主要来源；未发现必须先改变RTC5才能降低该现象的证据。

边界：仅固定观测开环反事实比较；指标不拼接不相关窗口，不覆盖真实chunk边界、接触/反馈、任务成功率或学习后的行为。两条同一现场Episode不是独立TEST。Q-only平顺不证明未训练Critic排序正确，Edit0.01也不代表最终最优探索参数。

## 可审查交付

源码da73e72新增本地 `realtime-expo-ft-edit001-preview` / expo-plug-left-2cam-edit001-preview-2000。冻结现场version0、SFT2000权重，仍原生32+32/Q/RTC5，唯一采样参数变化edit_scale0.01。服务没有Learner/Replay训练/权重安装接口；复用现有ROS执行、暂停/HIL与采集接口，独立录制/评测目录。原三个入口、原Online源码/参数/数据和SSH配置均未改。仅将一个registry条目及两个新运行文件在模型操作锁下安装，原文件SHA/新SHA和所有旧模型逐字段相同检查通过。

验证：49项EXPO CPU回归；真实Cobot GPU/HTTP40次推理，slow最大162.405ms、fast最大10.528ms，分别小于500ms/50ms预算。错误前缀、错误版本和训练/安装端点拒绝。两次测试进程均退出，最终GPU无计算进程。Web /api/site/options已返回新模型available，但未自动加载/开始推理；现场仍原失败入口error、无Session/录制。用户负责真机短轮对照。

后续Online应在真机预览通过后将0.01作为独立任务配方，同时用于Actor及Learner中的编辑采样/目标/损失，并保留独立配置身份和Replay；本次尚未接入低扰动Online，更没有把预览冒充在线更新。LFT1空闲卡和原生完整学习验证仍是该后续版本的条件。

## 证据

A6000：outputs/expo-native-remote-20261014/verification/jitter/{report.json,actions.npz,jitter-ablation.png,cpu-tests.log,preview-http.json,install.json,web-catalogue.json}。
Cobot：runtime/verification/expo-remote-20261014/jitter，保留脚本/日志/安装前registry；原episodes/actors只读。图用系统Python matplotlib生成（项目venv未安装，不改共享环境）。
