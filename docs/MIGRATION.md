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

## 2026-09-28 20:56：Getea1 存储迁移完成

本批已完成复制、哈希与运行验收、切换和对应旧文件清理。Getea1/jiaan 只保留 data、model；旧系统盘数据/模型目录移除。数据按场景/用途/方法归类，位姿与动作回放归 data/motion；模型按项目/模型/场景/版本归类。代码/环境/日志/PID 留在 /home/agilex/jiaan/project/<项目>。

USB 掉线重连后已完成已迁移资产的全量收据复核；尚不能据此认定硬件链路根因已消除。RLT 新路径暂停加载、在线状态恢复与历史媒体通过；FluxVLA 固定版本离线 baseline/prefix-RTC 通过；π0.5 两入口只做 dry-run。本批未启动真实 Episode 或机器人动作。

完整路径、占用、各项验证边界及回执见实际 cobot-web/docs/STORAGE.md。证据位于 rl-platform/outputs/migrations/20260928-getea-storage/cobot/（Cobot 去掉末尾 cobot/）。同批源码与项目记录已按各自仓库发布；guide Git 保持由其他会话管理。

## 2026-09-29：职责边界与部署材料

按实际源码、只读现场状态整理，代码先在A6000开发。结构、安装、依赖来源及验证界限见docs/DEPLOYMENT.md；跨项目关系见cobot-web/docs/ARCHITECTURE.md。数据/模型实体未迁移或删除；公共厂商工作区未删除、硬件未重启。guide只写事实、不提交其Git。现场切换与版本见后续发布回执。

## 2026-09-29：正式切换、清理及交付验收

运行边界首次发布e5fea09：shared_model_env、evaluation_env、profile_storage已归integrations/cobot_runtime；运行脚本不再source或import web配置。录制HTTP仍由web提供，领域实现由dagger提供，单writer保留。configs/local.json登记recorder_url，runtime/storage-selection.json接管目录选择。

现场preflight通过：Learner5000、Actor2500、warmup锚点2567；引用Getea1既有Actor/learner/Replay/normalization，robot_publishers=0。在线契约78项通过；A6000冻结环境独立恢复与模型/训练配置导入通过，Stage1契约35项通过。未更新生产Replay、模型或新一轮真机成功率。

RLT与EXPO-FT保持独立实现和环境；EXPO-FT登记为待适配，不伪造可运行状态。listen、Session-ready与inference_verified分别显示。环境恢复说明见DEPLOYMENT.md。

主代码位于 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform；现场副本 /home/agilex/jiaan/project/rl-platform。后续收尾版本以Git main和现场.release.json为准。guide仅更新事实摘要，不提交其Git。

## 2026-09-29：只优化RLT启动，保持正在使用的π0.5

用户要求先优化RLT，不影响正在使用的in_the_pot π0.5。改动在A6000主仓库，未改固定上游子模块、训练循环、控制参数、Pi05项目或Cobot模型进程；本批不搬迁、导出或删除数据/权重。

已确认当前Stage1参数：VLA 3,353,433,872，RL-token encoder 404,338,688，训练decoder 404,338,688，原checkpoint为float32，原推理恢复为BF16。部署只调用encode，但旧加载器真实初始化整套RL-token网络、再完整恢复权重。新实现整体nnx.eval_shape，只构造推理分支，Orbax选择性恢复VLA+encoder，跳过约4.04亿decoder参数（未压缩float32约1.62GB）。不是完整读取后再删decoder。原checkpoint保留供训练/回退，推理参数精度与值不变。

Stage1启动默认按需分配显存，保留原.72上限及显式恢复预分配的选项；增加项目内JAX持久编译缓存和分阶段计时，不跳过真实推理预热。新源码文件stage1_loading.py、验证入口scripts/validate_stage1_loading.py；原serve_stage1.py、scripts/rlt_up.sh与CLI/网页入口兼容。

17项相关冻结环境回归通过，bash语法与git diff检查通过。测试直接记录Orbax实际反序列化参数名，证明decoder没有被读取；验证保留原始文件哈希、抽象结构和encoder数值。固定Flax环境的ShapeDtypeStruct弃用提示与Orbax旧转换API提示不代表测试失败，本次没有升级环境。

A6000已有同版4999权重验证：全部3,757,772,560个保留参数逐值一致；实际2048D encoder输出逐值一致；修改前后的完整固定输入ref_chunk与z_rl均逐值一致，未连接ROS或机器人。CPU参数恢复单独比较为8.67秒→5.41秒（原版本先执行，页缓存未控制，不作为冷启动加速倍率）。

独立CPU进程、4个CPU核完整加载比较：原版32.60秒，新版19.80秒；首次完整推理28.02秒→25.68秒。再次启动新版加载20.10秒，明确命中jit_fun持久编译缓存，CPU首次执行23.50秒。这些是A6000 CPU实测，不能宣称Cobot原约9分钟加载已降到20秒；Cobot机械盘冷读和GPU峰值/编译时间留待下一次实际切换RLT测量。新版仍需读取必要VLA/encoder权重，尚未制作紧凑BF16部署文件或改用NVMe。

验证环境：/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/runtime-restore-verification/envs/stage1，沿用已恢复的固定包。证据 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/startup-optimization-20260929/，包含parameter-equivalence.json、{before,after,warm}-full-inference.json/.log及固定输入输出。Cobot同步代码/文档，不复制CPU编译缓存，不重启任何服务。发布版本与现场PID保护核验在发布后追加；guide Git不由本会话提交。

发布与现场保护核验：源码8c3cc48已push并核对origin/main，Cobot同步528文件SHA一致。π0.5 supervisor2019008、GPU服务2019058、客户端2025877，机械臂1318293、相机1317979及start_ticks均保持，网页无错误。未加载RLT、未创建机器人发布器，未改数据/模型资产。再次缓存命中的完整固定输入输出亦与原版逐值一致。

发布回执：A6000 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/startup-optimization-20260929/release.json；Cobot /home/agilex/jiaan/project/rl-platform/runtime/verification/startup-optimization-20260929/release.json。后续文档提交仅补记录，最终文档版本见Git main和.release.json。下一次用户正常切换RLT后核对实际GPU冷读、预热、峰值显存，不因A6000验证通过而自动释放π0.5或启动新模型。

## 2026-09-29：首次优化版 Cobot 冷加载实测

用户16:46:18点击加载在线RLT，16:51:08开始RL角色，约290秒。Stage1 imports3.501秒、配置0.349秒、抽象结构1.664秒、恢复259.279秒、绑定0.107秒；首次固定输入编译/推理21.016秒，后两次76.79/72.18ms。约4分50秒而非A6000 CPU的20秒，主要瓶颈仍是Getea1权重恢复。Stage1 PID2139241，supervisor2139119，日志outputs/rlt/plug_v3_yyshadow/logs/model-20260929T084619Z.log。模型自检通过不代表真机成功率。

随后 Session 开始失败来自历史未标注示范目录，而非模型加载；修正归cobot-dagger/web，保留RLT模型和算法进程。未修改训练/Replay/动作配置。现场用户后来切换online目录，已自行完成轮次；agent不为验证自动开始推理。

## 2026-09-29: read-only training diagnostics

Added lightweight telemetry aggregation and offline Replay posture/action projection; four regression tests passed. No training/driver changes. See docs/ANALYSIS.md. This batch deploys only the new analysis files; concurrent Cobot NVMe/probe edits remain untouched and are not included in this commit.

Live verification: the registered Cobot journal produced 3917 valid transitions, zero skipped; PCA axes explain 19.61% and 11.90%. Output: /home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/analysis/replay_projection.json. Web e775308 consumes it read-only via /api/analysis/rlt. The 9 hardware process identities were unchanged; no inference, learner update or Replay write was requested. The first analysis HTTP read completed in 0.20 seconds. Only newly added analysis files were synchronized; concurrent NVMe/probe files were preserved. Detailed deployment receipt is in the sibling cobot-web runtime/verification/rlt-analysis-20260929/release.json.

## 2026-09-29: Replay provenance and offline learning diagnosis

Added real-batch identity observation at the project Learner entry, composition refresh on journal growth,
archived Actor/Critic matching-step audits, whole-episode Online holdout sampling experiments,
loss/input sensitivity and recorded-image occlusion. Pinned upstream/train_step and production assets unchanged.
A6000 tests: 9 lightweight regressions; 1 frozen-JAX test with two real Learner updates compares
every state leaf and metrics exactly against an unaudited run. No re-sampling/RNG change.
12 experimental runs (4 sampling recipes x 3 seeds x 2000 updates), 14 archived Actors,
checkpoint11000 gradients and 6 recorded frames / 306 occluded Stage1 forwards completed on idle Cobot.
Full definitions, results, limitations and commands: docs/DIAGNOSIS_20260929.md.
Reports are ignored outputs; only small report copies returned to A6000. No production weights saved.
Actual historical batch identities unavailable; production logging starts next Learner launch.
New GPU version audits require explicit invocation; no automatic live GPU hook.
Code is developed/pushed on A6000; selected-file Cobot sync follows remote hash checks.

### Live release verification

The audit and diagnostic scripts are synced on Cobot, preserving concurrent local NVMe/probe
configuration. Formal8015 returns3917 transitions,14 retained versions,12 sampling runs and6 image frames.
Actual batch history is correctly empty until next Learner launch. Fixed-cohort regression added:
--versions-only reuses saved episode IDs as Replay grows and refuses changed cohort size.
10 lightweight tests plus1 real-JAX equality regression passed. No model/policy was started.

## 2026-09-30: registered credit candidate and execution diagnostics

A6000 development -> tests -> project Git push -> selected Cobot files with SHA256 verification. Added method-owned sampling/MC experiments, seven recipes x three seeds x 2,000 updates. Native upstream and journal rewards remain unchanged. MC30 candidate at /media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/history/candidates/credit_20260930/mc_30/online_candidate; runtime config /home/agilex/jiaan/project/rl-platform/runtime/experiments/credit_mc30/online.yaml. Actual Learner restore/update/publish/restart passed; isolated spawned Replay/Learner/Actor passed without EnvDriver. Fixed candidate shutdown ordering so Learner flush completes before Replay stops. Recorded 7D Stage1 RTC inference passed, but RTC/high-frequency publication remain disabled pending integrated control/Replay tests and field acceptance. Full results, paths, limitations and CLI: docs/EXPERIMENTS_20260930.md. Source: cobot_rlt conversation; no robot motion, no guide Git submission.

## 2026-09-30: frozen Critic guidance and Q-loss ablation

Added scripts/audit_critic_guidance.py and actual algorithm/config provenance to
credit experiments. Six frozen states audited on the same whole-episode cohort;
MC30 raises fixed-action Q1 human preference 23% -> 36%, but most windows still
disfavor it and gripper MAE worsens ~4.5%. Three matched Q-loss-off runs improve
HIL MAE ~0.98% without proving autonomous success. Online Replay has only six
unassisted successful episodes; 23 assisted successes confound terminal credit.
Whole-episode AUC uncertainty includes zero. Evidence, reproduction and remaining
work appended to docs/EXPERIMENTS_20260930.md. No robot publishers, production
weight writes or Replay changes; only small reports/figure copied to A6000.
Source: actual offline audit and cobot_rlt dialogue; guide Git not submitted.

## 2026-09-30: optional asynchronous RTC execution
Added project-owned EnvDriver/environment seams and execution registry; reuse VLA
RTC queue/sampler/physical-time utilities. New MC30 frequency entries share one
candidate branch. Original production algorithm/config/weights unchanged.
A6000 76 related tests and frozen Stage1 loading 3 passed; GPU audit and robot
acceptance pending. Latest feedback control peek avoids camera-frame gating only
for the opted-in path. No hardware restart or robot commands. See RUNBOOK.md and
EXPERIMENTS_20260930.md. Publish/sync receipts follow after release.

### Execution release acceptance
VLA bda74ad and RL77671fc published before selective21-file SHA sync; benchmark
613ae31 published/synced next. Real-policy synthetic GPU comparison7/7 passed,
related tests77 and pinned Stage1 loading3 passed. Four new entries were read
from the live8015 API as files-present/available, not process-ready/robot-verified.
Final HIL/version/timestamp/status refinements preserve right-arm semantics.
Detailed metrics/limits are in EXPERIMENTS_20260930.md; final release/sync receipt
in outputs/rlt-diagnosis-20260930/execution-final-release.json. Original defaults,
source assets and other-dialogue local/probe changes preserved.

## 2026-09-30: RTC runtime failure and retained-model recovery

Investigated the actual19:50:31 log: MC30 async_rtc50 EnvDriver rejected a late
RTC result; final supervisor traceback was a consequence. Stage1 remained
PID233064; recorder reported stopped/complete/committed. No recovery, inference,
robot action, asset deletion or re-labeling was performed during diagnosis.

RLT owns stdlib root-cause classification and detailed actual/allowed timing.
Web collection/deployment share runtime-failure display and guarded explicit
/api/rlt/recover-runtime. It uses the original launcher, verifies no live owned
runtime/pending writer, retains same-checkpoint Stage1, rejects implicit weight
reload and starts no episode. Recording recovery stays a separate action.
A6000 source/tests/push precede selective Cobot sync. Release and live verification
are recorded in the following deployment receipt; guide Git is not submitted.

Live release verified: rl-platform d4c798f and cobot-web4f5db63 pushed;18 selected
source/docs files match Cobot SHA256. Full web backend709 passed/6 skipped,56 DOM
checks passed,40 selected RL checks passed; additional orphan-shutdown/CLI32 passed.
Only the idle/completed orphan web service was restarted: Stage1 PID233064 and
14 hardware process identities/start_ticks unchanged. Formal8015 reports
rtc_delay_exceeded, retained Stage1 and recovery allowed, but whole runtime is
not ready. No recovery POST, Session start, inference or motion was performed.
The recovery mechanism is covered offline; end-to-end runtime restart and RTC
timing contention still need operator validation. Receipt on A6000:
projects/cobot-web/outputs/rtc-runtime-recovery-20260930/{release,live-verification}.json
under /data/LFT-W02_data/jiaan/jiaan; Cobot:
 /home/agilex/jiaan/project/cobot-web/runtime/verification/rtc-runtime-recovery-20260930/.

## 2026-09-30: pause-clock and pending-recording recovery follow-up

A6000 source changes: pause/HIL cancels obsolete publication deadlines; optional
async execution moves recorder HTTP off inference critical path with bounded
monitoring and generation guards. Web adds explicit finalization and scoped
runtime restart, preserving Stage1 and unlabeled recordings. Zero-option API
contract remains strict. No algorithm, ROS/action contract or asset layout change.
Operator procedure and code paths: cobot-web/docs/WEB_RECOVERY.md and
rl-platform/docs/RUNBOOK.md, latest sections. Offline regression, Git publication,
selective SHA sync and actual paused restart results follow in the release receipt.


### 2026-09-30: actual keep-model recovery acceptance

Source release: rl-platform3902fa8 / cobot-web6b84116, both pushed and13 selected
source/doc files SHA-verified on Cobot before switching. Web full backend715
passed/6 skipped, frontend58 passed, RL50 selected passed. Follow-up recorder
mode-conflict handling returns409 mode_not_selected/writer_busy instead of a
generic500; unknown start failures retain stable API codes and log their cause.
Its API/recovery regression suite73 passed.

At21:04 the failed runtime was rebuilt with Stage1 PID233064/start_ticks20025113
retained. At21:06 a real 13-frame in-progress test recording was handed to the
new general recovery endpoint; the owned runtime stopped, the writer committed48
frames, its lease cleared, and the new runtime355612 became ready/disarmed with
policy_paused=true and emitted_commands=0. No Session start/resume, homing or
robot motion was requested. Seven sampled hardware PID/start-tick identities
were unchanged. Replay remained3917 and Learner7000/Actor3500.

Retained unlabeled test HDF5 (no labels sidecar, no Replay insertion):
/media/agilex/Getea1/jiaan/data/datasets/test/runtime_recovery_20260930/episode_000001.hdf5
(133151819 bytes; UUID43361fc7-485a-4590-a342-9bc683e806b0).
The earlier operator episode_000035 reached its3000-frame ceiling and had already
committed before the web restart; no label/deletion was applied by this task.
The new pending-writer path was verified separately using the test episode.

Receipts:
- A6000 /data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/outputs/pause-recovery-20260930/
- Cobot /home/agilex/jiaan/project/cobot-web/runtime/verification/pause-recovery-20260930/

This verifies actual process/recorder recovery without reloading Stage1.
Pause/HIL timing faults are covered by injected offline races. Continuous live
50Hz publication, real pause/resume/HIL cycles and insertion success still need
operator acceptance; do not advertise them as passed from this non-motion test.
The selected MC30 async_rtc50 profile is retained; faithful synchronous20 remains
the documented rollback. Project handoff/concurrent-dialogue guidance now lives
in each project AGENTS.md and the five laptop entry directories.

Final pause race check also covers a pause arriving between the authority precheck
and deadline evaluation; current selected RL regression51 passed.


## 2026-09-30：独立可选发布Hz/RTC/平滑与保留任务的轮次跳过

执行配置新增严格execution_options契约，发布20/30/40/50 Hz、RTC与因果平滑可独立组合；
逻辑Actor/Replay仍固定20 Hz，关闭平滑保留现有动作/速度限制。模型默认profile未改变。
网页只传递该领域契约，不复制执行算法；COBOT_RLT_EXECUTION_OPTIONS由运行组件解析。
录制健康HTTP失败/超时转terminal_pending并撤回策略权限，保留Actor/Learner/Replay任务；
健康worker保持单个在途请求，旧episode/generation结果不影响新轮次，超时处理不再同步访问录制HTTP。
Session skip验证episode/generation及故障归属，先暂停并确认自己的writer关闭/保留，
再清理本轮在线trace并进入waiting_scene；不归位、不自动开始下一轮、不插入Replay。
跳过结果未确认时留terminal_pending；控制/执行故障与Replay提交中的轮次不能作为小录制问题跳过。
启动响应丢失通过本次prepared index及latched identity匹配owned recorder，避免操作其他轮次。
隔离RL相关回归79 passed，覆盖16种配置组合、失联/超时、stale generation与非录制fault拒绝。
现场源码同步与网页运行证据归web迁移记录；本批未进行真机频率/插接成功率试验。

## 2026-10-01：复用跨模型执行选项合同

RLT execution_profiles 的 enabled/publish_hz/rtc/smoothing 校验改为调用 VLA 的 integrations/cobot/execution_options.py，支持通用 COBOT_EXECUTION_OPTIONS，兼容 COBOT_RLT_EXECUTION_OPTIONS。原 RLT profile、逻辑/Replay 20 Hz、默认模型、Stage1/Actor/学习及录制恢复均未改。

A6000 CPU 离线 test_async_execution 21 passed（固定第三方源码 PYTHONPATH、JAX_PLATFORMS=cpu）；无新 GPU 模型/真实 Episode/Replay 改写/运动。web 配套精简步数列表与通用运行选项；现场 SHA 发布见 web outputs/execution-compact-20261001/ 回执。

## 2026-10-01：固定 5000 的独立在线分支与 NVMe Stage1 修复

来源：用户要求保留原 5000、在线采集产生新步数，同时区分原版与 MC30。部署注册增加训练方法字段；固定 5000 登记有完整状态的在线种子能力。integrations/cobot_runtime/online_seed.py 校验已登记种子 Actor／Learner 步数、版本、优化器／critic／RNG 和 warmup 状态，在明确采集加载时创建独立分支；种子文件不作为写入目标，原 Replay 仅引用。新分支的 Actor／checkpoint、配置与运行输出独立；恢复复用同分支，模型目录只呈现最近新建分支的已发布实际步数，旧分支资产保留。未修改固定第三方训练循环、奖励／采样／优化算法。

Stage1 清单 checkpoint 曾回退到 USB，本批恢复 /home/agilex/jiaan/data/rlt/plug_insertion/reference_4999，并新增独立 stage1_root；在线 model_root 仍按既有现场配置引用。preflight 分别校验 Stage1 与在线资产，拒绝清单不一致，不静默回退 USB。没有重新复制／移动原资产。

离线 CPU：在线分支完整状态／种子保留／不匹配拒绝／NVMe 登记与执行配置共 24 passed；配套 web 目录／发布步数和训练指标回归通过。未启动现场训练分支、未新加载 GPU、未真机动作；真正在线更新与连续 50 Hz 表现未由此验证。同步 SHA 与现场只读预检回执见 web outputs/catalog-results-20261001/。

现场发布事实：2026-10-01 代码已 push 并逐文件 SHA 同步；web 209 个运行文件、RL 6／dagger 2 个本批运行文件一致。仅重载 8015，网页 PID 970937；模型 offline、recorder idle、无 active writer／lease，9 个采样硬件／模型 PID 和 start_ticks 不变。实际 CPU preflight 使用 NVMe Stage1 路径且归一化 SHA 与发布清单一致；两个旧目录入口的 8 条历史标签 GET 通过，现场目录和标签响应的 DOM 结果通过。固定 5000 文件与原历史标签 SHA 不变。未运行现场在线分支、真实动作或连续 50 Hz；真实浏览器视觉验收未完成。回执：cobot-web/outputs/catalog-results-20261001/。

## 2026-10-05：真机 Online 前离线复现、输入合同与候选验证

本批在独立分支 audit/q-guidance-20261005 完成；不部署、不启停现场、不运动，固定第三方 8cef77eb7c5211b45382bf9199c9cdf0aaf60a19、生产 Replay/权重/默认配置未修改。结果位于 outputs/offline-model-selection-20261005/，逐项数据身份、实际命令、配置、源码 SHA、产物和边界见 delivery.json 及各 launch/report.json。此前三阶段审计与 Critic 目标机制报告分别位于 outputs/full-chain-audit-20261001/、outputs/preonline-verification-20261005/。

Stage1 已实际前向全部134训练见过 Episode/19022帧；加载器动作还原与转换数据最大差约2.8e-17。120专家/1186 transition 的20Hz动作目标全部重建，确认旧参考为30Hz前10点：参考覆盖0.30秒，而目标约0.45秒。重采样参考的训练内关节拟合约改善32%；同一旧5k Actor改用该参考反而约变差16%，该混用组合未放行。转换120/14并未提供Stage1独立测试。

早期两个插入轨迹记录锅盖任务提示词，按完整C10动作/source/起止状态精确追到Warmup Episode3/7的29条训练记录。已增加插入环境显式任务检查，错误默认在构造ROS前失败；现有启动脚本已设置正确任务，默认配置未静默替换。六组配对原生冷启动5000 Critic/2500 Actor对照使用独立输出，原Replay保持只读。累计30组Actor继续训练、6组原生冷启动与此前6组Critic-only目标反事实重拟合完成；局部监督拟合改善不等于自主收益，没有新Actor自动发布。

额外精确追到1751条历史rollout记录中的1728条唯一trace窗口，11条身份歧义、12条缺失未补造。仅完整覆盖Episode的记录时间跨度均值：策略约0.88秒、人类约0.27秒、混合约2.4–2.9秒；这些不是硬件发布时钟，却明确显示其不对应统一0.45秒动作合同。Warmup归档专家ID为100000以上，当前生产Replay专家已重映射为负ID，不因符号不同而混淆身份。读取单条原始HDF5的五帧候选图像，共2.49MB压缩输入，仅在A6000 GPU0做实际服务前向，未能唯一复现旧Token/参考；未把最近帧当成原VLA输入。原生数据堆叠类型差异已修正后重跑六组，修正前四组保留为被替代诊断记录，未冒称生产问题根因。

可选staged发布已用真实固定上游多进程/5k完整状态CPU链路验证：3条私有合成经验产生15次Critic/7次Actor更新；中断剩余预算准确恢复，候选Actor2507而执行Actor2500不变。分析工具修复图像归一化设备路径一个float32 ULP偏差；3个固定真实输入在两个实际服务独立恢复进程中动作/Token完全一致，但批处理缓存与服务的严格动作阈值仍有失败，失败记录保留。Stage1训练入口只接受已知干净版本且受保护源码与作者版本相同，实际validate-only通过。

回归：methods测试337通过、集成测试75通过，共412；在线冻结环境相关28通过。上述测试和GPU0只读离线推理不证明现场时序、插入成功率或Online持续改善。没有独立自主测试，当前现场状态接口不可达；无动作一致性、受控冻结真机验收和Online放行仍缺证。原不可变Warmup5k及其旧参考合同保留作回退基线，研究权重未替换生产资产。现场可独立转交的验收Prompt在同一输出目录operator_acceptance_prompt.txt；正式操作流程MD未在本批改写。

## 2026-10-05：HIL 命令/反馈溯源补充与相机输入合同修复

本轮延续 q-guidance 隔离工作区，读取现有 216 份 HDF5 的低维字段。31 个 Episode 的 2,586 个严格身份配对 HIL 步骤中，2,585 个 action 等于 next-state 测量反馈；786 步有反馈完全相同的时间兼容原始帧，722 步候选命令相同，但缺逐步命令发布回执，未恢复为可直接训练的历史命令动作、未改 Replay。固定上游 HIL 常规路径使用 cmd_topic 收到的命令，首次无命令时才回退反馈；本地反馈目标与其语义差异已验证，自主收益因果证据不足。

修复 RosTask2IO 两条相机采样路径的 RGB 条件性合同错误，并在同步/异步 trace 记录步骤前后 I/O 证据，保留左右独立 coordinator 命令与源/接收时间，右臂有效性不依赖左臂新鲜度。该证据不是执行确认，不改变 Replay 动作或生产默认。现场只读查询三路相机当前均 rgb8，五帧 BGR 对照不匹配旧 Token/ref，不能把颜色称作已定位现场根因。

418 项整套回归、3 项分析专项测试通过，最后修正后 49 项定向测试通过；初始缺依赖环境失败日志保留。A6000 GPU0 只做五帧已有图像对照后释放，GPU1 未动；未启停现场、加载现场模型、运动或切权重。未新增训练或放行 Actor；模型独立能力与受控冻结验收仍未通过。详细身份、命令、图和边界：`outputs/hil-command-audit-20261005/`，上一批交付仍在 `outputs/offline-model-selection-20261005/`。

HIL 配对 Critic 补证（同日）：CPU 原 5k Critic、229 个窗口/31 个 Episode；全 Episode 终端标签为辅助成功 30、失败 1。仅替换已观察命令候选相同的槽位，并与同槽位 FP32 反馈基准比较以分离量化影响。完整 trace 覆盖的 12 个训练 Episode 平均 ΔQ1=-0.01036，Episode bootstrap 95% [-0.01821,-0.00273]；反复使用的开发集 3 个完整 Episode 为 -0.00827，[-0.01691,-0.00224]。min-Q 也平均下降。缺失命令使其仅是条件性冻结 Critic 响应，不证明命令最优、精确历史动作身份或自主收益。初版私有脚本把选中非终端 success 标记当整轮结果的错误已修正并保存旧报告；Q 数值与区间完全不变。详细身份/配置/CPU命令/图：`outputs/hil-command-audit-20261005/command_critic_report.json`。

## 2026-10-05：Online 历史身份审计与可选训练输入回执

独立 q-guidance 工作区 CPU 审计当前生产 Replay（SHA 0fb87e9ecc3b4e0ede50208caa3ceea93db3330bf976edf682cd6c21f684525a）：4013 条中的75个 Online Episode/1446条，按全轮终端标签为自主成功8、辅助成功23、失败44；不是独立评测。现存46份原始 Online HDF5 标签15成功/31失败，UUID一致，但与75轮 Replay 不是已证实的同一漏斗。录制标签服务自动将失败标为 keep_for_training=false，不据此排除 RL 失败经验。

旧trace仅172条核心动作/source/起止状态唯一匹配、2条歧义、1272条无匹配；11轮有唯一匹配、10轮核心完整。各匹配轮末段 Replay/旧pending trace终端字段有差异，未认定Replay错误。修正分析工具曾误将锚点VLA ref_chunk与逐步执行计划ref_action视为同物的比较假设，旧私有结果保留。当前保存 Learner step11000/Actor5500，而Actor快照step11500/版本5750；release仍是首次5k安装清单，不作为当前验收证明，未构造虚假配套恢复状态。

新增可选 COBOT_RLT_INPUT_AUDIT=off/record/strict（默认off）：原生窗口构建后记录当前/next原始输入及实际transition.to_numpy序列化指纹，兼容可选FP32 action；strict拒绝state不一致或存储非有限字段。单独replay_inputs回执不替代提交ack、GPU处理、真实权重加载/执行证据，不改变生产默认或Replay。整套426通过，补充record/溢出边界后相关17通过；私有三路480x640 RGB哈希p95约1.97ms，不是现场deadline结果。详情与实际命令/身份/两图：outputs/online-provenance-audit-20261005/。

本轮不训练、不占GPU、不部署、不启停现场或运动；生产Replay/权重与固定上游不变。输入工具完成不等于模型放行，独立自主能力与受控冻结验收仍证据不足。

## 2026-10-05：HIL 命令目标与实际原始观测的可选合同修复

CPU固定上游复现确认：原始append忽略trace当前observation，本地重采样后可出现实际起点state5、原始起点0、缓存proprio5；第二步重规划缓存还能把上一动作Replay next-state从实际1换成5。新增 COBOT_RLT_RAW_OBSERVATION_CONTRACT=trace（默认legacy）保留实际起点、上一next-state和重新规划的独立输入，并重定位缓存锚点；无暂停空白的伪造动作。复现是私有合成Episode/mock特征，不是历史全量受影响数或真实网络能力证据。

固定上游HIL读取周期初命令、本地原路径取周期末反馈的差异已追到实现。新增 COBOT_RLT_HIL_TARGET=coordinator_command（默认feedback），限定右臂7D/显式logical20/trace，使用同一接管epoch内新鲜/master/joint_right命令；左臂无效不影响右臂。缺失/过期/未来/接管前命令及策略周期中途接管明确拒绝本轮Replay提交，不伪造纠正动作、不发布人类命令。最新收到命令不是执行确认；首步topic传递次序与真实拒绝率尚待现场验证。

新异步测试发现sample包装丢弃io_evidence，修复后默认/可选路径均传递；修正此前“写trace字段即完整传递”的过强推断，真实失败日志保留。完整C10经过原生trace/raw/Replay/实际FP32 serializer/严格输入回执验证，保留VLA锚点reference及周期末next反馈；HIL无新Actor推理时版本-1。454整套、82最终相关测试通过；实际命令/身份/边界见 outputs/hil-target-contract-20261005/。

无训练/GPU/现场启停、模型加载、部署或运动，生产Replay/权重/默认配置和固定上游未修改。该批是数据链候选，不放行新Actor；命令执行时基、算法收益和独立自主能力仍证据不足。正式流程MD未改。

## 2026-10-05：私有原生输入、学习与 Actor 导出链路核验

在 audit/q-guidance-20261005 隔离工作区新增 scripts/validate_input_learning_chain.py；真实不可变 Warmup5k Actor/Critic/优化器/RNG与归一化资产，CPU私有Replay/Actor RPC、原生EnvDriver/Learner，600条起始合成记录和1个30步HIL合成Episode。启用可选logical20、coordinator_command、trace观测、strict输入收据和FP32动作；生产默认、Replay、权重和固定第三方未改。

最终verified_publication运行验证3条新C10记录与输入收据、实际归一化JAX batch逐字段指纹一致；原生预算15次Critic/7次Actor，Learner5015/候选2507，完整恢复且剩余预算0。新记录Critic抽样5/4/2次、Actor2/2/0次：UTD5不保证每条新记录进入Actor更新。实际配置500次更新定期导出，15次更新完成后flush前导出仍2500，flush后2507；独立执行服务读取冻结2500，确定性动作不变。所有自有私有服务/端口已退出，原5k与归一化SHA不变，无GPU/Stage1加载/机器人发布/现场部署。

基于既有4013条Replay及75个Online完整Episode长度的固定池条件计算，单个记录未进入Actor的概率约0.9%–52.8%；各轮5倍更新预算均不足首次500更新定期导出。这不是历史batch/时序恢复，也不代表实际75轮均无导出；跨轮累积、其他预算及flush会改变发布。自主/辅助/失败仅为操作员终端标签，既有数据为训练/反复开发资料，独立测试留空。

报告/图/身份/命令/配置：outputs/input-learning-chain-20261005/{REPORT.md,delivery.json,sampling_exposure.json,sampling_and_publication.png,publication_launch.json,verified_publication/}。工具初始fixture/CLI/核对API错误及中间成功运行均保留；最终带退出及导出时序核对的verified_publication为主证据。未将更多采样或导出当作动作身份/credit缺证的修复，未晋升模型；自主提升与受控冻结真机验收仍证据不足。仅取回小报告，不取私有Replay/权重。正式运行流程MD未改写。

## 2026-10-05：近期经验原生分层采样六组模型对照

独立输出outputs/recent-sampling-model-study-20261005，代码基线32f513a/固定上游8cef77e；真实不可变5k完整状态、固定历史训练/反复开发资料，CPU-only六组配对native uniform/stratified，种子41/42/43、各2000Critic/1000Actor更新。两条件共同排除错误提示词Episode3/7，目标/特征/参考/reward/loss/budget相同；原生分层默认0.4近期/0.3Warmup/0.2human。专家归档正大ID仅在私有采样元数据负数化，不改原始数据、不声称生产UUID映射。直接原生train_step与索引函数，不冒称现场Replay/Learner。

研究池2496训练窗口（2377Warmup/119Online8Episode）、189开发窗口，六旧Online辅助成功Episode仅作反复开发诊断。不得将池比例当生产4013 Replay。实际近期抽样约1.88%→40.03%；类别重叠、human池包含专家。六旧辅助成功Episode的分层减均匀关节MAE=-0.00013317rad，整Episode配对bootstrap95%[-0.00021492,-0.00007675]；夹爪+0.023511mm，[+0.014321,+0.032502]。相对均匀关节改善5.6%、夹爪变差8.8%；相对初始5k夹爪变差32.2%。联合代理门槛失败，未晋升Actor；幅度未验证任务影响，不宣称真机更差/更好。

trace动作身份可追的Online训练仅1个人类动作Episode、对应Online开发0个；六旧诊断全部辅助成功。cached z/ref及FP32反馈的匹配不证明精确历史命令/VLA输入/因果时序，独立测试缺失。可继续诊断夹爪分维度/source/loss梯度，不能直接把更多近期采样当修复。CPU先导20更新×两条件通过，六组完成且源资产SHA不变，PID720279退出；GPU0外部IsaacSim任务占用、本轮不占GPU，GPU1/现场/默认/生产Replay/权重/固定上游未动。

完整数据/代码SHA、配置、命令、逐Episode/关节/夹爪误差与区间、checkpoints路径：delivery.json、main/study.json、summary.json、launch.json；四图及REPORT.md/PROGRESS.md同目录。六份研究权重留A6000，不复制整份资产；回退仍原5k/均匀采样。正式流程MD未改写。

## 2026-10-05：原生 Actor 来源/夹爪梯度与单因素继续训练诊断

独立 audit/q-guidance-20261005、基线95edbf7，CPU原生7完整状态×6来源batch共42案例；aux-only梯度捕获和原函数参数差0，梯度加和最大误差3.34e-6。夹爪为absolute、前六维delta，往返夹爪误差0，不存在本研究路径夹爪delta mask错误；分位归一化/Actor输出均不裁剪。初始5k专家/策略更新增加旧六个HIL开发Episode夹爪误差，HIL来源减少；六类固定batch去Q均略差，不能将现象简单归因于Q。零梯度仍恢复Adam动量更新，未把非线性更新误算成可相加贡献。

六旧辅助成功开发Episode177窗口、同状态当前冻结Actor提议与反馈目标的分维替换：初始5k仅关节ΔQ1=-0.0504，完整Episode区间[-0.0819,-0.0229]；仅夹爪+0.000855，区间跨0。不是历史介入前提议、有效执行命令或动作最优性证据；Q1/Q2/min-Q分开记录。

本批六组CPU单因素非HIL夹爪BC0.5 vs既有原生1.0，各sampler/种子配对采样索引SHA一致，从不可变5k各2000Critic/1000Actor更新至7000/3500。stratified旧六条夹爪MAE相对原生下降0.03490mm，但120专家夹爪上升0.02430mm，相对初始5k仍回退；uniform也有取舍，联合代理条件失败。lambda1完整更新与原生数值完全一致，不晋升研究Actor，不继续无目的扫描。

已验证条件性来源/loss拟合冲突；真机根因、自主改善、冻结验收仍证据不足。实际数据身份/SHA、配置、命令、42案例、6训练日志与三图见outputs/gripper-update-diagnosis-20261005/{REPORT.md,delivery.json,summary.json}。训练2496/开发189，Warmup被5k见过，旧六条反复诊断；没有独立测试，区间以整Episode计算。研究权重只留A6000。无新运行代码修复、GPU/现场启停/部署/运动/生产Replay/权重/默认/固定上游修改，原5k与旧reference保持回退。

## 2026-10-05：原始录制图像存储、同步与采样身份排查

基线60e53aa隔离工作区，仅CPU只读Cobot216份HDF5元信息/时间/validity，648相机实际存储均未压缩uint8，无有损JPEG。Warmup170轮31598帧、Online46轮4674帧，三相机源ROS跨度>100ms为0、源回退为0；按完整Episode的p95跨度中位数27.71/29.53ms，不代表现场实际推理/发布延迟。录制fps均10但59份有65个>0.2s间隔，最大3.860/5.490秒，不补均匀时间，暂停/负载成因未恢复。

确定坏帧：Warmup episode_000009.hdf5、UUID1d7cf69d-c316-485a-96be-966444a82344、第155帧high/left valid=false、像素全零；到达单调时钟比sample晚4.18/3.03ms。只在Cobot读该帧像素并取统计/hash，未复制图像。初始5k归档2567记录的数字Episode9不存在，不用数字ID代替全UUID溯源或宣称所有训练均未受影响。当前collector支持复制后取clock，历史v1版本因果缺证，本批不改采集项目。

已排除本批有损压缩与大范围>100ms相机跨度，历史实际VLA输入身份仍证据不足：录制/RLT独立取样、10Hz录制不能替代实际模型调用输入。图明确原始混合/成员未恢复队列、无独立测试；sample/arrival单调与source ROS分域，不相减混域时钟。完整清单、实际命令、脚本/SHA、坏帧/大间隔、图及边界在outputs/input-image-identity-20261005/{REPORT.md,delivery.json}。初版Cobot部署Git假设失败已保留并改报告实际源码SHA，不猜历史提交。无新运行代码修复、训练/GPU/部署/服务启停/运动/生产资产修改，模型/冻结验收未放行。

## 2026-10-05：冻结Actor首步限幅与HIL纠正方向Q核验

隔离基线1c0fe7e，CPU13个完整冻结状态×2891既有窗口：实际RightArmPolicyRuntime当前0.03rad/0.004m条件下首个提议限幅0；最大关节0.02352rad、夹爪0.001052m。只用首个已知起点，不伪造后续反事实反馈；不含探索/RTC/EMA/历史实际限幅配置，不能排除后续/现场问题。当前窗口由2496训练/189反复开发/177旧辅助成功开发/29已排除错误任务记录组成，没有独立测试，没有重新训练错误轨迹。

相同缓存状态上当前冻结Actor沿人类记录反馈槽位方向的原生JVP：初始5k六旧辅助成功Episode平均Q1导数全部负，mean=-0.04808、整Episode配对95%[-0.06678,-0.02938]；Q2区间跨0，min-Q平均-0.02873区间负。alpha替换不是执行轨迹/历史Actor/最优命令，不能凭此把HIL Q强行抬高。7既有状态端点逐Episode与上一报告一致；另外6 BC0.5状态保留诊断，不晋升模型。

新本批仅分析，无训练/GPU/现场部署/运动/生产Replay/权重/默认/固定上游修改。工人826360已退出，源/归一化/研究状态SHA核对；多维CI汇总轴序错误修正后重构图（不影响原始Q响应）。实际数据身份、配置、命令、边界、13状态逐Episode与两图见outputs/proposal-execution-diagnosis-20261005/{REPORT.md,delivery.json,final_verification.json}。输入无损身份、有效命令目标、独立自主能力仍缺证，冻结真机/Online未放行。

## 2026-10-05：有界无损Replay输入诊断与原生闭环验证

隔离基线bdf59dd新增input_snapshots.py、verify_input_snapshot.py及专项测试，online_runtime既有可选audit钩子接入。默认COBOT_RLT_INPUT_SNAPSHOT_COUNT=0；启用上限3份/EnvDriver、每份原始数组8MiB，要求显式独立绝对root、trace合同和record/strict收据。在Episode Replay构建阶段保存当前/next raw RGB/state/prompt/RTC、native feature/target和实际serializer数组，不使用pickle；外部receipt含metadata SHA，恢复核数组dtype/shape/SHA及原始指纹。超出数量明确未捕获，不补造数据；不在实时发布循环。CLI把输入合同检查与model_consistency_verified=null分开，不宣称实际GPU处理。

冻结online专项27通过，既有冻结stage1 CPU环境完整468通过。初次测试路径错误未运行及online环境缺h5py/pydantic的收集失败保留，未安装/修改生产环境。真实不可变5k完整状态的CPU私有native链路：600合成初始记录、30步合成HIL、3份快照逐字段恢复并与RPC journal及15实际batch一致；15Critic/7Actor、候选2507/执行2500不变、恢复一致、全部自有服务/端口退出。是软件验证，不是新任务训练或收益证明。

此前A6000已保存的真实尺寸候选RGB与合成feature fixture：6张480×640×3无损往返，原始5530213bytes/压缩1003299bytes，单次写加构建130ms/恢复29ms；不是现场p95/deadline，不能计入发布循环。完整身份/实际配置/命令/回退/容量与错误日志见outputs/input-snapshot-verification-20261005/{REPORT.md,delivery.json,final_verification.json}；快照/私有Replay/权重只留A6000，小报告取回。

算法、Actor/Critic默认、固定上游、生产Replay/权重不改，snapshot默认关闭作回退；无GPU、现场部署/启停/模型加载/运动。实际Stage1输入处理/有效命令/独立自主能力仍缺证，不晋升模型，后续同输入网络复算需单独核资源和边界。正式流程MD未改。


## 2026-10-05: actual Stage1 CPU lossless-input numerical closure

Isolated source base0b50423, actualStage1 checkpoint4999 restored in two independent CPU processes (4cores, seed42,10denoising,legacy30Hz C10,noRTC). Existing retained rawRGB/rightstate candidate frames0/2 from WarmupEp3, fresh correct plug prompt; one synthetic native EnvDriver transition, not historical input/command/task outcome. Six current/next z/proprio/ref native fields match exactly (maxabs0), and actualserializer casts match snapshot arrays exactly. Snapshot externalmetadataSHA b022d5ee3e2290a6488d21110f3eea00272c663f4e869a1eb07627be33e9ca36;31checkpoint/assetfiles SHA unchanged before/after. Actual3,757,772,560 inferenceparameters restoredBF16, no weightrewrite.

CPUworkers879350/882640 exited; outputs/stage1-input-snapshot-model-20261005/{REPORT.md,delivery.json,create_report.json,verify_report.json} retain exactcommands/config/source/dataSHA/load/calltiming. No runtimecode changes, training, GPU, field services/model/motion, production Replay/weights/default/upstream changes or Actorpromotion. FP16 feature serialization remains explicit/quantized; independent CPU equality does not prove GPU parity, historical image identity, validHIL commands, autonomous improvement or fieldtiming. This offline input check passed; model/Online acceptance remains insufficient. Formal operating flows unchanged.


## 2026-10-05: actual Stage1 features to frozen5k native Actor closure

Sourcebaseb138e79, two previously independently verified freshStage1 current/next diagnostic anchors from oldWarmupEp3 with correctplugprompt. Actualimmutable5k checkpoint and originalnorm SHA checked, Actor snapshot entireparameter tree exactly matchescheckpoint5000/2500. Four CPU native/storedfeature privateRPC calls equal directnetwork outputs exactly; nativeRightArm firstcommand .03rad/.004m clipping0. StoredFP16 feature vs liveFP32 conditionally changes output at most .000474rad/.0399mm, Q1 about+.0036 on these2anchors; no independenttest/task/field rootcause claim. Q1/Q2/minQ retained separately.

Private analysis worker repaired originalnorm vs researchformattedcopy identity check (numericstats exactlyequal) and inheritedfieldlogging path through unusedLearnerconfig (failedbeforelistening); both failedruns preserved, no runtime/production fix. Finalworker898496/privateActor898526 and failedActor896931 exited, privateports43105/35491closed; sourceweights/norm unchanged. No training/Replay/Learner/GPU/fieldservices/model/motion/production/default/upstream changes or modelpromotion. Details outputs/stage1-actor-chain-20261005/{REPORT.md,delivery.json,feature_precision_chain.png,verified_run/report.json}, actualcommands/config/sourceSHA/failurelogs retained. Existingjoint/gripper/oldscene candidate retention gate stillfailed; independentautonomous evidence insufficient. Formaloperatingflows unchanged.


## 2026-10-05: delivery entry staged-policy selection and rollback packet

Currentoffline handoff identified formalrlt_up seed block neverforwards existingnativeCLI publication/budget choices; realimmutable5k fullstate/readonlyarchived2567 Replay reproduced explicitstaged->automatic. Entry now forwards optionalCOBOT_RLT_SEED_PUBLICATION_POLICY andCOBOT_RLT_SEED_REPLAY_BUDGET_POLICY, defaultautomatic/new_arrivals unchanged. Explicitrequests on existingbranches checkseedmetadata+served/candidate paths and rejectconflicts/tampering withoutmutatingbranch. Nativeprepare/API/defaults/algorithm unchanged.

CPUactualseedblock/nativeCLI real5k comparison fixedstagedserved/pending separation, allparameters/optimizer/RNG retained; originalseed/snapshot/norm/Replay SHA unchanged, resumeunchanged.19targeted(10new actualshell/CLIcases),478full/61existingwarnings, bashsyntax/diffchecks passed; ownedtestPIDs911903/913454 exited. No models/services/ROS/GPU/training/field/productionReplay/weights/default/fixedupstream changes or modelpromotion. This fixes optionalentryavailability, not a proven historicalQ/robotrootcause.

Outputs/preonline-delivery-package-20261005 containsREPORT, manifest/delivery, sources/patch, tests/realstateevidence and standaloneoperator_acceptance_prompt.txt; privatefull5k branchassets remainA6000, onlysmallreports retrieved. ConsolidatesStage1/Warmup/Online findings and rollbackidentities; no candidatepasses joint/grip/oldscene retention, historicalvalidcommand/input andindependentautonomouscapability stillinsufficient. Latestsourcefacts/packet are notcurrentfieldruntime/load/acceptance evidence. Formaloperatingflows unchanged; guidefactualsummary only.

## 2026-10-05：真机前现场静态交付与实际提示词 CPU 补证

用户要求完成真机 Online 前确认与交付。本批核对现场发布 bf111eb 与运行实现 5320b56：完整 622 文件中 75 个不同或缺失，未登记修改冲突为零；备份 19 个旧文件及原发布收据、新增 56 个文件后同步并全量 SHA 校验，默认配置与固定上游不变。现场九个关键模块导入通过，实际 Python3.10 环境输入/HIL/Replay/seed CPU 专项 68 项通过；网页 PID970937、RLT offline/recorder idle 保持，生产 Replay SHA0fb87e9ecc3b4e0ede50208caa3ceea93db3330bf976edf682cd6c21f684525a 未变。没有现场模型加载、GPU、服务启停或运动。

现场 NVMe Stage1 的31资产及固定 Warmup5k checkpoint/Actor/原件norm哈希一致。A6000 CPU4 补用现场实际短提示词，两个保留的旧录制诊断锚点、两次独立 Stage1 恢复六字段逐值一致；固定5k私有 Actor RPC四调用与直接推理一致，完整资产不变，所有私有进程/端口退出。这是新构造数值链证据，不是历史输入、独立测试、命令最优性或自主能力证明。

执行分析工具增加显式 --prompt/--profiles，报告保留实际 policy metadata 和配置 SHA；省略参数保留历史默认，空提示词在加载前拒绝。CLI检查通过；实际现场 GPU 前向/执行检查仍未进行。没有新 Actor 晋升或 Online 放行，独立冻结真机结果仍缺失。详细命令、配置、SHA、同步/回退与边界：outputs/preonline-field-verification-20261005/；现场源码备份在 runtime/verification/preonline-field-verification-20261005/pre-sync-source.tar.gz。正式流程MD未修改。

## 2026-10-06：真实 GPU 无动作检查与执行诊断修复

用户明确要求自动完成，无需再确认；本批仅接管Cobot空闲RTX4090的有界无动作检查，不启停现场服务、不运动、不改生产Replay/权重/默认或固定上游。实际GPU代码c1e8409071fc21f573513586f9c81d119502894d已提交push并同步Cobot623文件SHA一致。

首次真实检查恢复Stage14999并成功前向，async_rtc20合成路径通过；faithful20因分析工具错误使用Async PlannerBackend访问None.stats失败。工具已改走原生EnvDriver与现场适配补丁，同步用原生planner、异步用自己的backend；显式Online动作路径以实际调用5k Actor，20Hz/C10，eval_actor_only、不写Replay/不学习，30逻辑步限制并核对Actor版本。FakeIO生命周期补齐，真实任务结果为null，不将合成停止称作失败任务或成功任务。两项有意义回归防止faithful重新误走async后端及Actor被Warmup Reference替代。A6000完整480通过/61既有warning，Cobot Python3.10新回归2通过；中间失败保留。

第二轮真实RTX4090 backend=gpu/cuda:0，原固定5k Actor2500/norm、NVMe4999、实际短提示词、旧Online Episode1 frame0与理想反馈；faithful20/async_rtc20均30逻辑步/30模拟命令、RL30/BASE0/HUMAN0、fallback0、intervention0、Replay0、学习0，退出码0/25.195秒。faithful模拟有效17.924Hz，p95间隔98.422ms、chunk边界max130.445ms；async约20.000Hz/p95间隔50.038ms、max50.042ms。这是单条合成执行诊断，不是实际机器人deadline、跟踪或自主能力证据；没有RPC/Learner争用，RTC条件不同。

运行后31个Stage1资产、5k checkpoint/Actor/norm、原录制及生产Replay SHA均未变；实际web970937保持、RLT offline/recorder idle、自有进程退出、GPU空闲。第一次launch把历史部署登记PID754128写作web_pid，第二轮改用实际进程证据，旧收据保留并明确更正。源码同步未切换当前网页选择或模型服务，没有新Actor晋升。交付可转入受控冻结真机验收准备，独立自主测试仍0，持续Online学习未放行，模型改善目标未完成。

报告、8图、版本/配置/命令/SHA、原失败、退出码、回退与独立现场prompt：outputs/preonline-gpu-20261006/。Cobot对应runtime/verification/preonline-gpu-20261006/，pre-attempt2-source.tar.gz保留上版诊断脚本/release收据。小报告取回，不复制完整数据或权重。本节只记事实，正式操作流程未改。

## 2026-10-06：eRLT/开源实现核对与可选训练诊断

用户要求自主完成并允许离线训练比较。本轮只读eRLT2610.00913v1方法/实验/消融及附录，外部RL-Token-Pi05-open固定cccf949c源码；论文AUC是自主成功率学习曲线积分而非Critic ROC-AUC，辅助采集单列且计预算。开源Actor仍完整输出，非零residual被拒绝；未据截图或曲线更换算法、TD截断或默认配置。

原固定5k/归档2567训练Replay/204Episode CPU4分路审计确认已有256/64/256投影和LayerNorm，Q1七维动作梯度非零；只排查具体数值淹没假设，不能证明表示语义或HIL命令最优。新增默认关闭COBOT_RLT_DIAGNOSTIC_METRICS可选原生TD日志，以及离线训练曲线/AUC证据工具，缺指标留空、Actor只取更新行，正奖励chunk不称成功Episode。A6000真实完整5k两条私有CPU续训各8Critic/4Actor、相同8x256batch，参数/优化器/RNG/旧指标逐值相同，没有私有5008/2504导出或晋升。首次batch字段错误在训练前失败，保留失败。

代码9b7601e73b829fcf0d695a46ae2c15aeca482523提交push，隔离分支串行快进；A6000两目录495通过/61既有warning，Cobot实际Python3.10 CPU专项15通过，630源文件SHA一致。Cobot同步前623文件无冲突，备份旧bootstrap/MIGRATION/release及新增7文件清单保留。没有现场GPU/model加载、服务启停、运动、生产Replay/权重/默认/固定上游修改。原31Stage1资产、5k资产及生产Replay/录制原件身份复核不变；实际web970937/RLT offline/recorder idle，自有CPU任务已退出。

outputs/erlt-reference-review-20261006/保存REPORT、两份实际训练曲线、分路图、实际命令/配置/SHA、原失败、回退和独立现场验收prompt。固定5k仍为基线，没有新Actor通过联合保持或独立自主改善；可准备受控冻结真机验收，直接持续Online能力仍证据不足。本节是事实记录，正式操作流程未改，guide仅事实摘要、不提交guide Git。

## 2026-10-06：异步 Reference 对应修复与 HIL／采样／预算对照

用户授权自动尝试截图中的加速和采样/HIL建议。本批CPU4、独立工作区，原固定5k完整状态与既有缓存数据私有续训18组：保留HIL/rollout-HIL来源改RL/互斥四池各25%各三种子1000Critic/500Actor；同8个历史Online Episode新增119窗口的预算1/2/5各三种子119/238/595Critic。来源重标不改变Critic直接更新，却改变Actor的BC/delta目标；旧6条辅助开发Episode拟合变差。四池相对均匀改善开发拟合，但专家相对退化、原5k夹爪保持仍失败；较低预算减轻部分夹爪退化而有取舍。无独立测试或Actor晋升，生产采样/预算/权重未改。

确定代码错误：异步队列保留旧动作/版本/来源时未保留对应旧Reference，RTC开关均复现。修复完整action/ref/version/source前缀，覆盖实际延迟0/2/4；不能据此归因旧同步Online表现。新增默认关闭async20_no_rtc_no_smoothing并支持显式诊断CLI，default仍faithful20，replan5/delay4，不冒称第8步整chunk预取。真实墙钟/合成80/120/180ms特征延迟原生执行9格各30步，异步约20Hz；无真实模型/机器人动态/任务成功。

代码f5f11f5c32222270cdb3f083012c1a51ec525194已串行提交push，A6000两目录503通过/61既有warning，Cobot实际Python3.10 CPU29通过；630文件旧源无冲突核验后同步4文件，保留runtime/verification/acceleration-hil-ablation-20261006/pre-sync-source.tar.gz。原5k三资产与生产Replay SHA复核不变。现场挂载已恢复、模型目录存在，但网页8015只读查询连接被拒绝；旧PID/旧上线结果不是当前运行证据。没有现场GPU/模型加载/服务启停/运动或生产Replay/default/fixedupstream变化。

outputs/acceleration-hil-ablation-20261006保存REPORT、5图、18组真实状态/曲线/采样身份、完整Episode bootstrap、命令/配置/SHA、失败尝试及独立现场prompt。私有脚本错误数字RL=4在发现后终止自有worker，保留原尝试并改用真实枚举RL=1重跑，未进生产。软件与静态同步已验证；冻结真机与持续Online仍证据不足。本节只记事实，正式RUNBOOK流程未改，guide只追加事实摘要、不提交guide Git。


## 2026-10-06：共享采集／评测 trace 终结接口修复

现场16:14 MC30 frozen（Actor3740）和16:22 Warmup5k frozen（Actor2500）在轮次收尾时均出现 `CollectionTrace` 缺少 `finalize`，env_driver退出。原始AtomicEpisodeTraceWriter已有终结方法，共享包装器遗漏转发；独立评测NoTraceWriter亦缺少完整生命周期。

本批只补充CollectionTrace.finalize按本轮已锁定用途转发，评测不写采集trace；NoTraceWriter补齐start/discard/finalize空操作。成功/失败/放弃、Session身份和收尾先于推进、幂等、连续轮次及采集→评测不改写上一轮均有回归。固定上游、模型/执行参数、Replay和历史标签不变。

A6000 CPU：直接相关88项通过；9个新增回归在基线均失败、修复后通过。全量383通过，19失败与同环境基线374通过/19失败的失败集合完全一致（缺少openpi/lerobot与Stage1 Orbax环境版本不匹配），没有新增失败，不声称全量全绿。证据：scratch/rl-platform/trace-writer-lifecycle-20261006/pytest-{reproduction,terminal-session,baseline,fixed}.txt。

16:14故障轮次metrics记载transitions_written=0，日志Learner disabled，不能将其计作训练样本或已学习轮次。部署同步与实际运行切换另记；本批不补写历史pending标签，也不启动真机Episode。

现场交付：修复4ce0a88已push，锁内确认Frozen runtime退出、录制idle后仅同步4个文件，全清单630文件SHA一致，15个ROS/相机/机械臂/web/Stage1进程身份在同步前后相同。Cobot冻结online Python3.10的14项trace终结测试全部通过。

用户随后自行重新加载Warmup5k（16:25:44，supervisor PID83992）；本会话恢复调用的前置检查发现已有新加载，未发送POST、未停止其进程。16:26:04后只读确认新运行ready/disarmed/policy_paused=true、Session未开始、step0、录制idle、Learner disabled，Actor2500。此为启动及离线收尾回归验收，尚未替用户进行修复后的真机success/failure轮次。没有网页/硬件重启或机器人动作。现场证据runtime/verification/trace-writer-lifecycle-20261006，A6000证据scratch/rl-platform/trace-writer-lifecycle-20261006；历史pending trace未改写。

后续现场观测更新：用户自主执行的Warmup5k Frozen评测连续三轮点击success均正常收尾，metrics原生episode245/246/247为success=1、transitions_written=0，Actor2500；日志三次success提交，无Traceback，env_driver85189仍存活，随后用户开始第四轮。证明本次评测终结崩溃已在真实调用路径消除；不将操作者标签当成独立成功率验收，采集入Replay、failure/aborted真机路径仍未在本批现场试验。证据live-terminal-observation.json（现场）/同名txt（A6000scratch）。


## 2026-10-06：A6000产物归拢与可选按轮更新（未发布Actor）

按plug_v3_online_rl_review(1).md复核并实现可选CPU按轮更新：完整Episode/source/奖励、多次接管截断、四池归一化且Episode先采样、Actor全轨迹／无Q梯度及优势权重、原生两头Critic／独立导出。docs/ROUNDWISE_ONLINE.md正文已先展示，用户授权自动实现。未改生产默认／固定上游／Replay／原5k，未同步或重启现场。

两种子每候选Critic1000／Actor1500，对照末段LR1e-4、全轨迹LR1e-4、全轨迹LR1e-5；末段损害旧专家和自主拟合，全轨迹／低LR缓解但未过独立OOD／保持放行。恢复参数／优化器／RNG和双采样状态逐位一致；下一轮初始Actor预测继承一致；32状态CPU原生ActorService最大误差1.1921e-7，不绑定端口／加载Stage1／运动。

基线384通过18失败，候选417通过18同名环境失败；早期隔离依赖缺失26项补齐后消除。生产Replay只读4013行279Episode，SHA0fb87e9ecc3b4e0ede50208caa3ceea93db3330bf976edf682cd6c21f684525a，时间代理合格1800／混合37，非评审1861。新工具batch22.656/21.875/21.875/33.594%非生产历史采样或成功失败8:2。

1311既有笔记本文件逐SHA归拢A6000scratch/rl-platform/laptop-artifact-migration-20261006：1227相同／84补齐／0冲突。outputs忽略；Windows删除遭自动策略拒绝，保留旧副本。新增代码／CPU实验／报告／PNG-PDF仅A6000；身份／命令／验证／失败在outputs/roundwise-online-review-20261006/REPORT.md和JSON，最终提交收据delivery.json。

补正早期web只读探测：继承HTTP代理导致adhoc本机请求拒绝，禁用代理后200，不是现场故障。用户冻结评测Replay0／无Learner不算Online改善。现场所有权归用户；guide只事实摘要，不提交推送guideGit。


## 2026-10-06：笔记本旧产物可恢复清理完成

后续按用户要求改为 Windows SendToRecycleBin：仅在完整目录清单与 A6000 迁移收据一致、所有文件逐 SHA 验证且无新增／修改／链接文件后执行。1311 个旧产物（132236382 字节）及临时目录3文件已移入回收站，原路径消失；回收站内1311个文件与归档再次逐SHA核对一致，未清空回收站，可恢复。A6000副本、项目入口与代码未删除。前述永久删除的策略拒绝保留为历史事件，不能据此推断所有清理均被禁止；具体拒绝规则仍未知。

收据 scratch/rl-platform/laptop-artifact-migration-20261006/cleanup-recycle-20261006.json；交付报告及delivery.json已补记当前状态。本批只更新事实文档，无生产模型／Replay／默认配置／现场进程变化。
