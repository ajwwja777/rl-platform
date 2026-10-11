# 2026-10-11 探索采集与模型目录闭环核对

状态：保存显示故障已修复并同步；现已累计11轮，最新4轮均为同位置成功/无HIL。最新增量结论见末节；首批7轮证据保持。持续 Online 学习收益证据不足。

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

## 第二批：新增4轮成功（原始标注时间21:03–21:06，UTC+8）

**已验证**：网页历史API现返回11条完整记录；新HDF5 8–11完整、545帧，均标success，interventions=[]且右臂HIL帧0。用户确认新增4轮在同一个插孔位置。这不自动扩展为第一批和第二批场景一致，日志也不能排除未由HIL系统记录的物理调整；自主分类指当前人工结果标签+系统无HIL的口径。

| HDF5序号 | UUID | 录制帧 | 有效右臂coordinator帧 |
|---|---|---:|---:|
| 8 | f154fff4-c772-4055-9351-56bd7e675921 | 110 | 108 |
| 9 | 14060abb-12bd-4fe3-ae3d-a74ff26ff90c | 177 | 174 |
| 10 | 56da005e-d27f-4cea-9dd5-dea937cb24d4 | 142 | 139 |
| 11 | 8319b3c8-426a-44b0-984e-9e74350cfa67 | 116 | 114 |

**已验证**：新Replay原生Episode12/13/14/15分别写入41/67/54/44条，合计206；总457。严格输入回执457/457通过，旧7个trace的SHA未改变。新增权重版本始终Actor3640，learner_enabled=false，配置仍logical20/publish50/RTC/replan5/EMA0.08，没有在线更新或新权重晋升。

**已验证**：独立采集buffer按完整Episode为成功5/11=45.45%、失败6/11=54.55%；按transition为成功257/457=56.24%、失败200/457=43.76%；新增4轮贡献206/457=45.08%。这是独立探索buffer构成，不是旧生产buffer或真实训练batch；Learner关闭，实际batch数0。5次历史abort仍单列，没有新增abort。

**已验证**：新增1047逻辑步、2614次发布，210次推理，所有记录探索偏移均在原界内、发布request_id均可追溯。J2各轮平均规划偏移+0.056/-0.547/-1.444/+0.163mrad，有正有负；不能据4轮成功认定单向J2偏置更好。新旧批次首状态6关节均值差为[-0.556,+0.077,-0.050,-0.066,-0.274,+0.299]mrad，起始关节均值接近，但不能据此证明插孔位置或视觉条件相同，也不代表误差量级的任务意义已验证。

**已验证（时序事实）**：新增发布间隔各轮p50=20.361–20.375ms；2610个epoch内间隔中1个>40ms，47.738ms出现在原生Episode14，该轮成功。上一发布结束至下一次prepare开始空档约28.897ms；下一次prepare耗时0.033ms、发布调用0.104ms，现有trace不能定位未打点空档的具体阻塞因素。不能误称发布函数本身花了47.7ms，也不能用很小的post_wait_lateness（时钟已平移）掩盖实际间隔。未见新增控制超时/回退；单点延迟继续记录，不据此改RTC/滤波/频率。推理p50约141–144ms、最大181.631ms。

**失败（通用HDF5动作直接训练入口仍不适用）**：新4份通用action仍全NaN；右臂coordinator535/545帧有效且有限，缺失10帧均在开头2–3帧，需按右臂mask处理。所有三相机有效标记存在；未做图像语义复核。此问题与独立Replay来源分开；没有修改原始数据、掩盖NaN或为固定左臂补造动作。

**证据不足（能力/因果）**：第二批同位置4/4说明该固定探索策略在这个位置出现重复成功，是积极的行为证据；不是学习收益，因为没有训练。4/4的Episode级Wilson95%区间约51.0%–100%；累计5/11约21.3%–72.0%，连续同场景样本未保证IID，区间仅作描述，不能声称真实成功率100%。无新位置覆盖、无HIL纠正、无匹配无探索对照，因此不能判断泛化、纠正方向最优性、探索因果收益或Critic动作偏好改善。本轮未做模型前向，不补造Q指标。

本批分析沿用已批准的采集→准入→匹配离线对照计划；增加同位置成功数据后，仍缺失败位置的纠正轨迹。后续最有信息量的采集是实际偏离处的HIL纠正和明确位置记录，不能仅靠继续重复同位置成功来证明Online；保持当前Actor和执行配置，离线对照候选不自动晋升。

证据、图和复现命令：A6000 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/exploration-collection-20261011/batch2/`。snapshot.json、data-analysis.json、motion-analysis.json、summary.json、operator_context.json、validation.json；命令为A6000 `ssh agilex@10.7.165.64 /home/agilex/jiaan/project/cobot-web/.venv/bin/python - < reduce_field.py`及`reduce_motion.py`，输出另存独立目录；`python3 plot_compare.py`生成batch-comparison.png。trace/receipt/labels路径与SHA、HDF5 UUID/size/mtime均在JSON；实际代码SHA、资产和执行配置与首批相同。项目分析基线10ae12e。全部只读Cobot、CPU统计；无现场启停/模型加载/GPU/训练/Replay写入。
