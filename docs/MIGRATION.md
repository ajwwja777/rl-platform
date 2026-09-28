# 统一 RL 实验平台：迁移记录

日期：2026-09-27。当前业务迁移进度见文末；以下初始化内容保留为历史记录。

## 已有位置与成果

以下是已读项目记录与前序目录核验的入口清单，不表示本轮重新完整验证每项资产。执行某批迁移前须核对实际目录、符号链接、Git 状态和使用者。

| 机器 | 旧位置／已有依赖 | 保留事项 |
|---|---|---|
| Cobot | `/media/agilex/Getea1/jiaan/projects/rlt` | RLT 现有部署、模型、环境与运行记录，顶层与嵌套 Git 分别核验。 |
| A6000 | `/data/LFT-W02_data/jiaan/projects/proj-20260904-cobot-realworld-rl` | RLT 适配、审计、数据契约及实验配置；保留未提交修改。 |
| A6000 | `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/runs/plug_v3_yyshadow/warmup_20260925_trials` | 有价值的 warmup 对比产物，不能按临时文件直接清理。 |
| A6000 | `/data/LFT-W02_data/jiaan/jiaan/projects/expo-ft` | 已存在的独立仓库，作为固定版本依赖接入；本次不移动或提交其工作区。 |
| Cobot | `/media/agilex/Getea1/jiaan/data/evaluations` | 现有模型现场评测记录，不能误作具备完整 action replay 的训练数据。 |

## 首个候选验收范围

先登记并验证当前 RLT warmup 5k 模型及现有评测的完整来源，做固定输入的离线加载对照；不启动新的在线学习。

验收要求：权重与预处理校验一致，恢复运行所需 actor／learner 版本明确，现有评测记录完整可读；忠实复现参数不因框架接入而静默改变。

## 逐批迁移约定

一次只处理一个明确范围，记录来源、目标、依赖、版本／校验值和回退入口；先复制与验证，再切换，最后清理对应旧文件。未验收不切换，仍被依赖或缺少可靠备份的原件不清理。共有目录按文件实际归属处理，保留其他对话的未提交修改及共享资产。

迁移批次记录至少包括：范围、来源与目标、验证结果、切换状态、可清理清单及实际清理结果。初始化完成仅表示入口与 Git 可接管，不代表运行环境或业务功能已验收。

本轮没有迁移／删除旧文件，没有安装项目运行环境、启动训练、加载模型或控制机器人，也没有变更当前网页服务。

## 初始化发布记录

- 2026-09-27：项目目录与维护入口已建立，基础提交已 push 并核对远端 main 一致。
- 仓库：https://github.com/ajwwja777/rl-platform（独立仓库，非 GitHub fork）。
- 首次发布提交：`3063ae62a7043184e7331d877d1e89bb067b18cc`。
- 本记录在首次发布验证后追加并单独提交；最新版本以 main 为准。
- 运行状态：源码／文档基础已发布，业务迁移、环境安装及新位置运行验收尚未开展。

## 2026-09-27 RLT 业务迁移（进行中）

本批已取得迁移授权，guide Git 由另一个 agent 管理，本会话不提交／推送 guide。

来源为 Cobot /media/agilex/Getea1/jiaan/projects/rlt 和 A6000 旧 proj-20260904-cobot-realworld-rl。两边原件、差异与复制核验置于 outputs/migrations/20260927-rlt/。生产冲突以现场版本为基础；A6000 独有工具和测试保留。当前 cohort、模型计算、奖励、数据比例和 warmup 预算不变。

新运行布局见 README。代码主工作区在 A6000，通过 Git 发布后同步 Cobot；新模型、Replay、环境和数据均使用 Cobot /home/agilex/jiaan。模型历史归 A6000，现场只留当前所需模型。旧文件通过完整验证前不得清理。

代码调整：路径与所有权拆分；推理环境隔离；输出目录可配置；加载前要求已完成 warmup 的 checkpoint；Python 3.8 网页契约导入兼容；补充 Trace 仅在终止时 fsync（HDF5 和 Replay 提交语义不变）。A6000 可选 Actor pinning / 动作平滑保留，但当前 profile 不启用，默认动作裁剪保留现场公式。

实时进度与验收证据以后续“验收／切换”记录为准，不能将准备工作视为完成。

## 2026-09-27 23:55：已验证范围与现场连接中断

- 主源码提交 1f57443c19ac3ae948ca8dcf922b2c29ba0d6cbf、自有 upstream 8cef77eb7c5211b45382bf9199c9cdf0aaf60a19 已 push 并核对远端。upstream 是独立仓库 ajwwja777/rlt-openpi，保留 c1e40ac 作者历史；仅部署补丁纳入自己的提交，不向上游提 PR。
- Cobot 新项目同步 501 个源码文件，逐文件 SHA-256 一致。Stage 1 31 个文件、固定 5k 8 个文件、在线资产 135 个文件与旧现场原件逐项相同；Python 3.10 在线环境与 3.11 推理环境复制、重定位，不升级算法依赖。
- 新目录 preflight 实际通过：learner=5000、actor=2500、warmup_ready_adds_total=2567；7D/C10/z2048、配置非路径字段不变。
- A6000 测试：adapter/Session/录制 243 passed、upstream 45 passed；硬件项目 342 passed；网页 576 passed / 11 skipped。没有把测试等同于真机成功率。
- 新代码的独立在线恢复验证通过：实际 checkpoint / Replay 副本从 5000/2500 开始，无新数据时不更新；添加一个模拟 transition 后恰好更新 5 次至 5005/2502，指标有限，重启后不重复更新。原始三份输入 SHA-256 不变，零 ROS 发布者。脚本 scripts/validate_online_resume.py；证据 outputs/migrations/20260927-rlt/resume-validation-a6000/report.json。
- A6000 原 warmup 对比历史复制到 outputs/rlt/plug_v3_yyshadow/history/，1,731 个文件 SHA-256 一致；选定 5k checkpoint / actor / norm 归 models/rlt/plug_v3_yyshadow/warmup-5000。原件仍保留至全链路切换。
- Cobot 约 148 GB 数据已经复制，逐文件校验尚未取回完成报告；旧 RLT 全量归档至 A6000、旧 cobot-platform 至 vla-platform 的归档也未完成。不得清理原件。
- 用户已明确机械臂安全断电，可重启节点。23:44 前确认网页 idle、模型 offline；随后在新路径发起 Stage 1 --validate-only 固定输入验证，无 ROS 发布者。23:46 左右 Cobot SSH / ping 不再响应，至 23:52 仍连接超时，尚未取回加载结果。原因未定，不能判断为模型通过、磁盘故障或已关机。
- 尚未重启硬件节点、尚未部署本轮网页路径变更、尚未删除旧 cobot-platform / rlt / data。现场现状以连接恢复后的重新检查为准；先核对验证任务、数据校验、模型与录制状态，再接续验收。
- Guide 摘要只修改 MD，不由本会话提交／推送；其他 agent 的 expo-ft / storage-cleanup 修改保留。

当前未交付完成项：现场权重加载／新硬件节点验收、网页切换、全部数据／历史资产校验和对应旧目录删除。不得把已复制／已 push 写成已经能够在线真机运行。

## 2026-09-28：现场恢复、新路径运行验收

网络恢复后确认 Cobot 已重启、旧节点已退出。本轮不归位、不启动真机 Episode，不改变奖励、模型计算、warmup 步数或数据比例。

- 新 Stage 1 在 Cobot 实际加载成功：总耗时约44秒；首次固定输入推理16.80秒，随后76.29/76.37ms。动作和 token 尺寸、有限值通过，无机器人发布者。A6000 保存同一 Stage 1，31文件/15,428,192,821字节跨机器 SHA-256 相同。
- 正式8015已使用新配置、新 rl-platform/control。真实共享模型入口分别加载固定 Warmup 5k 和最新在线 Actor，均 ready/disarmed/policy_paused=true，Session未开始；随后正常释放。
- 在线恢复5000/Actor2500/Replay2567，无新数据不更新。Cobot Python3.10独立副本测试通过：一条模拟transition→5次更新→5005/2502，重启不重复更新；正式checkpoint、Replay、norm SHA与迁移前一致。
- 数据 rlt、evaluations、cobot-platform、record、test 已复制到 /home/agilex/jiaan/data；11,123文件/158,037,688,841字节、2内部链接验证完成。HDF5、标签、provenance不改写；内部链接重定位。网页配置和浏览器最近目录按注册前缀迁移。
- 历史读取：warmup原始历史170条，训练历史筛选111条；episode171的标签、视频和首尾六张相机图均可读。两个计数属于不同筛选口径。
- rlt_up/down/status 不再被旧网页配置重新指向旧项目；中臂home入口归同级control，9项相关测试通过。
- configs/rlt/plug_v3_yyshadow/initial_assets_20260927.json 记录迁移基线，不限制今后合法在线变化。

证据：outputs/migrations/20260928-cutover/stage1-cross-machine.json、cobot/stage1-validation.log、cobot/resume-validation-cobot.log；网页证据在相邻 cobot-web/outputs/verification/20260928-cutover/cobot/。原始副本测试报告在 Cobot 同名迁移目录 resume-validation-cobot/report.json。

仍在完成旧 cobot-platform/RLT 历史资产跨机器归档与全量校验，未通过前保留原件。新运行路径、加载、数据读取与独立学习恢复已通过；上电后的 HIL/动作/结果提交需现场短轮次验收，不构成新成功率。

## 2026-09-28：数据与A6000旧源码清理

- 五个旧数据根完整校验后删除，新Cobot数据位于/home/agilex/jiaan/data；11,123文件共158,037,688,841字节保持原样，2内部链接重定位。网页历史与媒体再次读取通过。删除回执 outputs/migrations/20260927-rlt/data-cleanup-receipt.json。
- A6000旧proj-20260904-cobot-realworld-rl已清理：479个文件/链接与完整tar归档及可读副本一致，无活动引用。tar和legacy-a6000-source位于outputs/migrations/20260927-rlt，回执outputs/migrations/20260928-cutover/a6000-source-retirement.json。没有提交旧父仓库或其他项目的改动。
- Cobot旧RLT源文件全量SHA清单完成，39,001条目；跨机历史归档仍在复制。复制时复用已存在的同一文件只用于减少传输，最终按完整SHA清单验收，不能仅以大小相同判定通过。
- Cobot envs现场快照已生成，包含online/stage1/python311与overlay，排除可重新生成的pyc/cache；7,485,824,462字节，SHA-256 214403a0eeefa5f31bbc7505808fe93ec344d5e656ff4b6d815d92eae7445f2b。A6000备份传输排在历史归档之后，完成前不删除现场暂存。
- configs/environments/cobot-online.txt（70包）与cobot-stage1.txt（202包）是冻结版本清单，不包含凭据或下载地址；不是已验证的pip从零重建锁文件。现有overlay、ROS、CUDA和SDK依赖需按启动脚本保留，不升级依赖。

## 2026-09-28：RLT 归档、冷启动与旧目录清理完成

本批迁移已收尾。主代码/Git在A6000，现场代码、环境、选定权重和Replay在Cobot新项目，数据在/home/agilex/jiaan/data。本次未开始真机Episode，算法参数和正式训练资产不变。

- 旧RLT完整归档到A6000 outputs/migrations/20260927-rlt/legacy-rlt-source：38,655普通文件、346链接，共122,208,969,870字节；全量SHA及原链接文本通过，条目无缺失/额外文件。复用重复内容仅减少传输，最终每个路径单独验收。
- 历史模型实体归models/history/stage1-legacy40/4999、stage1-plug-e78/step_2000、step_4000、stage1-plug-v2/3999；当前Stage1/4999重复副本逐文件SHA比对后复用models/rlt/plug_v3_yyshadow/stage1/4999。datasets归data/history/legacy-rlt（165条目、609,616,285字节）。configs/assets/legacy_rlt_models.json保存来源和文件哈希。
- 原归档通过相对链接保留原查阅位置；343个绝对内部/别名链接重定位，2个原相对链接保留，1个原pytest临时目录链接仅作历史证据保留。跨旧VLA别名的4个脚本经解析链及源manifest核验后指向已验证的相同归档文件。整理后38,655普通文件全部仍可访问、大小一致；原命令与provenance文本未改写。
- 现场冻结环境快照已在A6000 outputs/environments/cobot-runtime-20260928.tar.gz完成SHA验证，7,485,824,462字节。包含online/stage1/python311及overlay，不升级依赖。校验后删除Cobot本次暂存tar，保留现场实际envs和两机JSON回执。版本清单不能替代已安装ROS/CUDA/SDK或保证从零重建。
- 隔离旧路径后重启8015（153484→222607），通过真实共享入口加载plug_v3-online-latest：ready/disarmed/policy_paused=true、Session未开始、step=0；learner发布新鲜状态5000/actor2500/Replay2567、pending_update_budget=0、training_frozen=false。释放后offline，正式learner/actor/norm/Replay SHA全部不变。
- 清理前复核39,001源条目元数据未变化、无活动进程引用；冷启动通过后删除Cobot旧/media/agilex/Getea1/jiaan/projects/rlt及其cobot-realworld-rl链接。对应旧cobot-platform、五个旧数据根、A6000旧RL源码已在各自批次验收清理。共享cobot_magic及π0.5/驱动资产保留。
- 删除后9类只读接口、37个页面资源和episode171标签/视频/首尾6张图片通过；6个共享模型入口可用。工控机空间查询/，剩余约77.6GiB。无GPU计算进程，模型offline，无活动Session/评测轮次。
- 当前硬件状态与早先被动测试不同：网页后来启动arms PID148006和cameras PID158179，5臂反馈、5CAN及3相机可用。本会话保留这些新任务；先前测试停止记录对应其他PID，不把后续启动判成停止失败。这些状态不等同于本会话完成上电HIL/动作验收。

证据：outputs/migrations/20260928-cutover中的rlt-source-files.json、rlt-archive-sha256.json、archive-layout.json、organized-archive-access.json、platform-cross-project-links.json；现场完整回执和加载/释放证据在outputs/migrations/20260928-retirement/cobot/。outputs/environments保存环境备份及SHA。当前冷加载/恢复通过，不构成新的模型成功率。

接续使用现有docs/RUNBOOK.md：在线目录选/home/agilex/jiaan/data/rlt/plug_v3_yyshadow/online，模型选plug_v3-online-latest。网页保留原warmup目录选择，没有强行重标历史数据。现场先短轮次确认暂停/HIL、结果提交与所选复位，再连续在线采集。未进行浏览器目视动画验收；两个π0.5共享部署、FluxVLA适配和完整驱动环境重建属于后续批次。


## 2026-09-28：Getea1 统一存储迁移（进行中）

Cobot 数据与模型统一在 /media/agilex/Getea1/jiaan/data/ 和 /media/agilex/Getea1/jiaan/model/。数据按场景分、模型按项目/模型分；本轮不新增 A6000 权重备份。代码、安装环境、运行日志与 PID 留在 /home/agilex/jiaan/project/<项目>/。完整路径与批次状态见相邻 cobot-web/docs/STORAGE.md。

已在 A6000 接入新存储配置及旧路径映射；逐文件复制/校验正在进行，正式网页已在空闲状态正常停止，机械臂/ROS 进程保留。本段不代表旧源目录已经删除。位姿、回放、示范、RLT rollout/Replay、评测和部署权重按 STORAGE.md 归类。最终运行验证及删除回执待本批完成后追加。

## 2026-09-28 20:00：迁移中遇到 Getea1 USB 掉线

已完成主体 12,059 条目、351,844,176,546 字节及 6 个恢复验证资产、117,047,594 字节的迁移、SHA 校验、运行验收和对应源文件清理。Warmup 与在线模型在新路径加载/释放通过；在线状态 5000/2500/2567，正式权重和 Replay 的 SHA 不变，未启动 Episode 或真机运动。历史读取、92 条有效评测和媒体通过；主副本清理后再次读通。系统盘当时剩余约 404 GiB。

剩余 FluxVLA 环境复制到 libcublasLt.so.12 时出现 I/O error。内核在 19:59:53 将 sda 下线，随后 USB 设备枚举失败；20:00 检查已无 Getea1 块设备和挂载。不能把它归因于单个 Python 包或仅网页错误，也不能仅凭这些日志判定是线缆、供电、硬盘盒或盘本体。

所有迁移进程已退出；正式网页 PID 366090 正常停止，无 GPU 模型进程，临时 ROS master 已停止。本轮未做运动。尚未验收的 FluxVLA 旧目录、暂存副本未清理，**Getea1/jiaan 仅保留 data/model 的目标尚未完成**。已验证结果仅代表掉线前状态，恢复连接后仍须核对文件系统并按迁移收据重新校验新资产，不能直接继续删除或开始在线训练。

证据：相邻 rl-platform/outputs/migrations/20260928-getea-storage/cobot/，现场同目录不带 cobot/。包括 retirement.json、validation/retirement.json、cutover-verification.json、extras/copy-status.json、disk-disconnect.json 和 disk-disconnect-kernel.log。源码和证据位于系统盘/A6000，本轮没有新增 A6000 数据/权重备份。
