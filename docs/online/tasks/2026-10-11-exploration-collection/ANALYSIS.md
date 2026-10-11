# 2026-10-11 新采7轮与模型目录闭环核对

状态：保存显示故障已修复并同步；固定探索采集链路已有真实数据证据；持续 Online 学习收益证据不足。

## 结论与数据身份

- **已验证**：7个 complete、labels_complete 的HDF5，共661帧；操作员标签为6失败、1成功；标签/录制HIL字段和trace均无人工介入。另5次abort没有提交Replay，单独记账，不计失败或独立测试。
- **已验证**：独立Replay提交回执30/37/48/29/35/21/51条，累计251条；严格输入审计251/251通过、原生episode_id为2/4/5/8/9/10/11，run_id=7603ed4e1d284fc2a22f80bc36fed294。回执检查state/feature及数组有限性，不等同检查所有语义或训练效果。没有历史batch可统计：本入口Learner明确关闭，训练batch数0。
- **已验证**：1252逻辑步、3121次去重发布均来自Actor3640；249次完成推理有探索偏移；全部偏移在2σ内，J2最大绝对值6.85868mrad。每次发布可追到本trace的request_id。它是规划目标偏移，不是TCP实际位移或滤波后偏移幅度。
- **已验证**：执行为logical20Hz/publish50Hz、RTC=true、replan5、EMA tau=0.08、关节速度限0.6rad/s。逐Episode发布间隔p50=20.359–20.399ms，最大31.966ms；epoch内3112个间隔无>40ms。排除暂停/epoch边界，只覆盖7个完整trace；40ms是描述阈值，不等于系统deadline。推理p50约143–145ms，最大170.005ms。不能据此证明插入能力提升。
- **失败（通用HDF5动作直接训练入口）**：7份HDF5的通用`action`全NaN、联合valid_mask/action全false。capture_core/sampler.py中_pair要求左右命令均新鲜；固定左臂无新命令使coordinator_ok=false，action整行置NaN。独立`rollout/coordinator_command[:,7:14]`仍有640/661帧有效且有限的右臂命令。21帧右臂命令无效，包括每轮起始2–3帧及5/7号录制中的暂停段。后续只按右臂valid_mask读取该字段，并保留无效边界；不要填0、伪造左臂动作或跨缺口拼接。原始数据不改。本次RLT Replay从数值trace构建，不经这个通用HDF5 action；不能把该限制归因为本次Replay全NaN。
- **证据不足**：成功率改善、位置泛化、Critic动作偏好改善、探索的因果收益。记录1/7=14.3%，Episode级Wilson95%区间2.57%–51.31%，仅描述本批；连续收集不保证IID，同一场景相关性未建模。位置/偏移标签缺失、无同时期冻结无探索对照，无真正独立TEST。251个重叠窗口不能当251次独立试验。
- **证据不足**：图像坏帧/语义任务完成。只检查三相机有效标记、形状/时间戳和人工结果标签，未逐帧视频复核。相机时间戳跨度各轮p50约16.6–20.1ms，最大38.25ms，是异步相机观测事实而非同步精度验收。

## 原始录制（UUID身份）

真实根目录（Cobot）：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/exploration-hil8-7281-v1`。

| 文件序号 | Episode UUID | HDF5帧 | 操作员结果 | 有效右臂命令帧 |
|---|---|---:|---|---:|
| 1 | 164b2bd2-4cee-47ca-89e8-a8b3c5a4e99a | 82 | failure | 79 |
| 2 | bcaef048-d842-47ce-8b55-ab11fc95f397 | 98 | failure | 95 |
| 3 | 01187a85-44df-4b52-9ec8-b8380d3d34a3 | 126 | failure | 123 |
| 4 | ebd35e5b-05c4-4f0e-b13b-d914f609e927 | 79 | failure | 77 |
| 5 | d20b5155-1756-4777-ad98-4e8c52bb5fed | 96 | failure | 92 |
| 6 | cb3b0c13-3a8e-43e9-a512-50d7a7a6850e | 57 | failure | 55 |
| 7 | e48ed143-ca1e-4299-8930-4563c96af464 | 123 | success | 119 |

表中HDF5序号和原生Replay episode_id属于不同命名空间。本报告没有凭排序伪造UUID与Replay ID的一一映射；各自身份、计数和时间证据分别保留。后续逐Episode跨资产训练清单必须补显式关联。

失败HDF5的`episode_quality=bad`与`keep_for_training=false`保持原值；这是通用示范筛选标签，不是RL失败经验必须删除的依据。独立RL Replay已有失败经验。二者不能静默混用。

## 代码、配置与命令

实际采集运行源码来自121f0f8交付；trace记录async_execution/execution_runtime/exploration逐文件SHA，见data-analysis.json及原trace。Actor checkpoint SHA：`020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5`；stats SHA：`4583a59496d4c459a155c960adf4baac6af0ccbd536a2168914a28809f45366c`。现场启动日志明确RLT_DISABLE_LEARNER=1；不是仅根据frozen标签推断。

新目录注册RLT提交7bd6cda；前端提交2936dad。此前f39745c允许显式自选路径的Python源码修复仍未热切换到活动模型；本次按照用户偏好的模型专用目录方案，网页与既有实际录制路径对齐，无需重启。不能声称活动Python已加载f39745c。

产物（A6000）：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/exploration-collection-20261011/followup/`。

- `reduce_field.py`：由A6000 stdin通过SSH送Cobot web .venv Python，只读HDF5数值/标签/trace/log，返回小JSON；不读图像张量，不加载模型，不复制完整资产。
- 复现：A6000执行 `ssh agilex@10.7.165.64 /home/agilex/jiaan/project/cobot-web/.venv/bin/python - < reduce_field.py > data-analysis.json`（在独立复核目录，勿覆盖已有证据）。
- `data-analysis.json`：文件路径、UUID、size/mtime、标签/trace/input-receipt SHA、逐Episode统计、实际配置、日志片段。
- `plot_analysis.py`：A6000系统Python3（matplotlib3.1.2）执行生成`episode-audit.png`；图仅用收集/开发数据，无TEST曲线。
- `field-sync.json`：4个文件新旧SHA、目录POST回执、7条history、受保护Actor/独立Replay及进程身份未变。
- `http-verification-final.txt`：实际服务的新registry字段、JS HTTP200/SHA；`frontend-regression.txt`：77项前端回归通过。
- `pre-sync-status.json`：ready/waiting_scene、policy_paused、recorder stopped，实际录制路径。早期HTTP诊断误受代理影响的输出留档但不作为服务停止证据；改用ProxyHandler({})核实服务始终同PID。

## 接下来（采集与训练分开）

1. 用户刷新数据采集页，仍选“插孔 · HIL8 探索采集（固定权重／不训练）”；网页自动使用专用目录，下面可见现有7条。先下一轮确认新序号出现，无需重新加载权重。
2. 保持同Actor、50Hz/RTC/滤波和探索幅度，优先补3–5条在常见失败处及时纠正的HIL轨迹；记录目标位置（中心/左/右及可测偏移），分别标自主/辅助结果。HIL标签保留；不能把辅助成功伪装自主成功。继续累计约20–30个完整Episode只是首批分析预算，不是统计放行门槛。
3. 执行方先补显式UUID→Replay身份清单、右臂字段/暂停边界复核，再按完整Episode预先固定新DEV和训练部分；旧DEV保留检查保持能力。不要自动把失败HDF5 keep=false映射为RL删除，也不自动导出通用action。
4. 再做已有A0与W2/AWBC方案的匹配离线对照：同初始化/Replay快照/预算/随机种子/执行设置；改变训练配方时列清teacher、BC/Q/delta、学习率，不把多参数配方对照冒充单因素因果证据；先逐项再组合。检查6活动关节尤其J2的纠正方向/幅度、同状态Actor/Reference/HIL配对Q、旧DEV保持和异常动作尾部。HIL不是所有状态的最优反事实。
5. 达到预先记录的离线门槛后才交付可回退冻结候选；采集期间不自动训练、晋升或开启持续Online。当前7轮只能验收采集链路，无法认证“越学越好”。
