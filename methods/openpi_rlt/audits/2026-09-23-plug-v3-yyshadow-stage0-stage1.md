# 2026-09-23 plug_v3_yyshadow Stage 0/1 启动审计

## 结论

用户决定停止沿用 `plug_v2` 的累计改动，从重新采集数据开始建立 `plug_v3_yyshadow`，优先忠实复现 Yyshadow/openpi-RLT。Stage 0 已完成：固定上游身份、源码哈希、原始 Stage 1/online 参数、隔离边界及 Cobot 允许适配。Stage 1 的机器可读场景合同和现场清单已完成；实际场景参考证据仍由操作员现场确认，未伪造为已完成。

本轮没有连接 Cobot，没有创建 Cobot 数据/运行目录，没有启动 CAN、ROS、相机、RLT、训练或机械臂动作；没有访问 HPC/trainer。

## Stage 0：已锁定

- 上游：`https://github.com/Yyshadow/openpi-RLT.git`
- commit：`c1e40ac360185778c98cf20da2820e22d2d415e7`
- tree：`15f7602f6e9ceaf5f4b70721a48088e52271e751`
- relay 只读 clone：`/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/openpi-rlt/upstream`
- clone 状态：`main...origin/main`，无工作树修改。
- online YAML SHA-256：`65c727270979f6d88e88f6bcb44c0d1397e77abfa3552fcbceff33be09ae93b8`
- `training/config.py` SHA-256：`98757f454c831ed48267a081f60cf860b1c8fa66e9246adeaf19bdf09c8099ae`
- `train_rlt.py` SHA-256：`7ec1cf99980e56d961777b43852642466f982f5c0222a76d02f44d8b6bf5f77d`

首轮严格保留上游 Ethernet 默认：右臂7D、C10、20Hz、delta chunk、gamma .99、std .002、reference dropout .5、delta10、warmup BC10/Q.1、online BC5/Q.1、actor/critic LR1e-4、tau .005、batch128、warmup600 transitions/20000 updates、UTD5、stride0。

Stage 1 使用上游 `rlt_pi05_agilexbag_image_delta_joint`：Pi0.5、内部动作32D、H50、1个2048D RL token、2层、alpha1、5000 steps。

首轮明确禁止复用或开启：`plug_v2` 数据/权重/norm/features/replay/release、MC-success critic、固定20% HIL sampler、五episode发布门、自定义actor gate、30Hz training-time RTC、delta100/300、Q.01、actor LR1e-5、dense replay和reward shaping。

## Cobot 必要适配

- 平台可保存完整双臂14D原始证据，但训练、proprio、actor/critic只使用右臂7D，物理索引7–13。
- 左臂关节、左夹爪和中臂state退出训练；左臂固定在`all/plug2`并作为相机载体，中臂固定在`mid/plug2`。
- 视觉使用mid、left、right三路真实画面；动作与proprio只使用右臂7D。
- 只允许ROS1、双相机键名、关节映射、HIL桥接、急停硬限位、生命周期和存储路径适配。
- 所有算法差异必须显式登记，不能伪装成上游复现。

## Stage 1：合同已建，现场证据待填

场景合同固定任务为“右夹爪已经持有插头，从近孔起点开始，到指定插孔真实落座结束”。排除抓取、长距离搬运、episode中移动排插、批次中重装相机或改变夹持。

现场必须补齐：

- 操作员已指定机械臂`all --pose plug2`和相机臂`mid --pose plug2`；仍需读回精确pose文件和关节值；
- 插头夹持深度/方向实体标记；
- 排插方向与位置编号；
- mid/left/right三路参考帧、文件哈希、topic频率和时间偏差；
- 起始14D state；
- nominal/train/held-out位置及训练比例；
- 操作员、时间与照片manifest。

成功定义为插头真实机械落座；发生过HIL的成功单独记 `hil_success`，不计为自主成功；故障、坏帧、错误复位或取消记 `aborted`，不进入正式replay。

## 产物

- `methods/openpi_rlt/plug_v3_yyshadow/README.md`
- `methods/openpi_rlt/plug_v3_yyshadow/reproduction_contract.json`
- `methods/openpi_rlt/plug_v3_yyshadow/scene_contract.json`
- `methods/openpi_rlt/plug_v3_yyshadow/SCENE_CHECKLIST.md`
- `methods/openpi_rlt/plug_v3_yyshadow/validate_contract.py`
- `methods/openpi_rlt/plug_v3_yyshadow/upstream_snapshot/online_rl.yaml`

## 验证

`python3 validate_contract.py` 已通过上游commit、clean状态、五个源码哈希、锁定参数和场景schema检查。`--require-scene-reference` 按预期拒绝，完整列出尚未现场填写的字段，因此不会在场景未冻结时误进入采集。

## 下一步门槛

操作员完成现场场景确认后，只填写真实证据并运行：

```bash
python3 validate_contract.py --require-scene-reference
```

只有该检查通过，才进入 Stage 2：在新的 Cobot cohort 目录建立采集入口并采集第一批专家数据。任何 Cobot SSH、服务启动和机械臂动作仍需当前任务现场授权。


## 2026-09-23 现场范围补充

操作员确认本cohort不使用左臂或左相机。唯一初始状态使用已保存的`all/plug2`与`mid/plug2`位姿和对应图像；训练/推理的动作与proprio缩减为右臂7D，视觉使用mid/left/right三路图像。该决定已写入机器合同。当前仍未连接Cobot读取文件，因此pose/image路径、哈希、实际关节值与相机频率继续标记为待现场读回，不能仅凭名称宣称参考证据闭合。

## 2026-09-23 Cobot 空目录登记

经用户当前任务明确授权，只在 Cobot 创建以下空目录；创建前已核对两个父目录的真实路径，创建后逐项核验均为 `agilex:agilex`。未创建文件、未移动或删除旧数据、未启动服务或机器人动作。

- 数据根：`/media/agilex/Getea1/jiaan/data/rlt/plug_v3_yyshadow`
  - `demonstrations/{recording_tmp,lerobot,rejected}`
  - `warmup`、`online`、`evaluation`、`manifests`
- 运行根：`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v3_yyshadow`
  - `collection`、`stage1`、`machine_a`、`replay`、`warmup`、`online`、`evaluation`

这些目录与 `plug_v2` 隔离。下一步是只读核验 `plug2` pose/三相机参考及 topic，再实现新的采集 profile；本轮尚未开始采集。

## 2026-09-23 首批专家示教质检与归档

用户完成3条只覆盖插入关键阶段的成功专家示教。只读质检结果：episode 4/5/6分别为141/108/111帧，约4.65/3.55/3.65秒，采样率均约30 Hz；三路480x640图像完整、非冻结，相机topic时间偏差p99最大约31.14 ms；action/qpos finite且帧索引、采样时间连续。三条均为complete/committed，质检结论overall_pass=true、	raining_eligible=true。

任务合同进一步固定为单右臂：平台原始HDF5保留双臂14D证据，左侧7维在各episode内保持不变；后续转换只取索引7–13，即右臂六关节与右夹爪7D。视觉保留mid/left/right三路；左臂、左夹爪和中臂state不进入norm stats、Stage 1、actor或critic。

专家采集不要求额外点击success。完整提交且没有明确拒绝、硬件/传感器故障、无效复位或不完整commit的专家episode，隐式记为success=true。该规则只适用于expert demonstration profile，不延伸到warmup、online或evaluation rollout。

三条原始记录及对应.segments和已有.previews已从demonstrations/成组移动到demonstrations/recording_tmp/；未转换LeRobot、未删除原始HDF5、未启动训练。质检证据保存在Cobot：/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v3_yyshadow/collection/pilot_quality_episodes_000004_000006.json。

## 2026-09-23 采集计数补充

首批3条质检后，操作员又保存episode 9/10/11，分别108/115/112帧。只读结构检查确认约30 Hz、三路480x640图像存在且非冻结、action/qpos为finite 14D、帧索引连续，左侧7维在episode内不变。当前数据根共有6条完整HDF5：4/5/6/9/10/11；7/8为0帧放弃记录，已修复删除流程并清理。训练仍只允许取右臂索引7–13。

## 2026-09-23 专家数据增量质量门与7D转换准备

操作员继续采集期间仅执行只读数据审计和离线工具准备，未重启8015、未改动recorder状态、未启动模型/训练或机械臂动作。96条完整HDF5快照中92条通过、4条拒绝、1条只裁剪无效尾部；通过集三相机时间偏差p99最差约31.7 ms，采样频率约30 Hz，最短80帧，满足H50基本长度。

拒绝原因：episode 36和64各含一帧右臂action与实测qpos不一致的巨大尖峰（约1.42/0.68 rad；其余健康记录最大约0.033 rad）；episode 96固定左臂发生明显移动；episode 107在episode中段出现右臂qpos NaN。episode 122末尾两帧action无效，插入终态已在此前有效帧完成，因此选择区间固定为`[0,166)`。正常松开示教后的有限尾帧不再被误裁剪。

新增工具：

- `plug_v3_yyshadow/tools/audit_expert_hdf5.py`
- `plug_v3_yyshadow/tools/analyze_expert_trajectories.py`
- `plug_v3_yyshadow/tools/convert_experts.py`

转换器固定原始14D索引7–13为训练7D，保留mid/left/right三路视频，拒绝质量清单与原始inventory不一致的输入。Cobot合成2 episode/6帧烟雾测试已通过：Parquet state/action与源7D逐值一致，6个视频共18帧全部解码，抽样RGB最大MAE为1.0，转换前后源HDF5和sidecar哈希一致。测试证据位于`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v3_yyshadow/tests/converter-smoke-v1`；正式专家集仍在采集，尚未冻结、转换或删除原始HDF5。

当前动态清单为`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v3_yyshadow/collection/expert_selection_current.json`。目标按有效条数计约100条，不按文件编号计；采集结束后重新完整审计并冻结清单，再进行正式LeRobot转换与Stage 1准备。

## 2026-09-24 数据冻结、转换与传输

- 冻结清单：141条原始记录，134条接受、7条拒绝、1条裁剪；接受集19,022帧，均为成功插入关键阶段。
- 冻结清单SHA-256：`25f455ad59289ddd04ebeb4090c046ed940736f9330fae47c7f5200825491910`。
- 转换结果：134个Parquet、402个视频、57,066个解码视频帧；state/action均为`Sequence[7]`，源向量逐值一致，抽样RGB最大MAE 1.258368。
- 数据payload tree：`96e6e2b30016d75fdaf099a256f324398fcd472d0c8326bb2ce033edf49af101`；完整tree：`c160a1d239a4ee43247cbf7ba3ddefe1f2b86ca33eb3e0cb71fea19a1574d23f`；manifest：`c1c3fd4ad2ca0d1602d611c99f9ad97c4792a17882aca22224c7d375e76f1171`。
- Cobot、A6000、HPC三份数据均以manifest、tree hash和542文件数核验一致。
- 后续最快传输路径已验证为Cobot→A6000增量rsync→HPC；manifest最后写入，使用`--partial --append-verify`，不压缩视频，不把HPC凭据复制到Cobot。

## 2026-09-24 Stage 1 训练

训练固定上游commit `c1e40ac360185778c98cf20da2820e22d2d415e7`，使用`rlt_pi05_agilexbag_image_delta_joint`同构配置：Pi0.5、H50、内部动作32D、右臂物理7D、1×2048 RL token、2层、alpha=1、global batch32、AdamW、EMA .99、seed42、5000步。仅将单卡改为4×H100 FSDP以保持global batch并缩短墙钟时间，关闭W&B；loss、优化器、学习率、数据和目标均未改变。

作业证据：

- 预检job `646558`：134 episode/19,022帧校验和594个stats batch完成，`COMPLETED 0:0`。
- 正式job `646566`：4×H100，`COMPLETED 0:0`，耗时`01:01:03`，最终checkpoint `step_4999`。
- loss从step0的19.4323降至step1000的0.3593、step3000的0.2303、step4000的0.2080、step4999的0.1843。
- RL-token MSE从19.3637降至0.1804；VLA loss从0.0686降至0.0039；最终grad norm 1.4817。未出现NaN、OOM、作业重启或末段系统性反弹。
- 推理包仅包含EMA `params`与归一化`assets`，不带39GB optimizer/train_state；共31文件，约15.43GB，tree SHA-256 `e73ef3f30c72bc946e7f3e9b5f73d5e97c99021716c0004f8d90aefe40e31270`，norm SHA-256 `901a3eec8ee83940ae993695cb5432e1ce9e6c91a4e8072f9cd8288af4f09f8b`。

本次134条全部参加Stage 1，因此early/late cohort离线对比是确定性一致性诊断，不声称未见场景泛化。真机效果仍待现场授权和操作员验收。

离线选模job `646663`在固定early 0–13与late 120–133 cohort上比较2000/3000/4000/4999四档，`COMPLETED 0:0`。从step2000到4999：early前10步归一化action MAE由0.03768降至0.02505，late由0.05191降至0.03381；early/late RL-token loss由0.25213/0.23725降至0.19877/0.16735。最终档的early动作加速度指标亦由0.00479降至0.00434，late为0.00501，与step4000的0.00494差异约1.4%，没有与动作拟合改善相抵的平滑度退化。因此选择`step_4999`，不机械回退到中间checkpoint。

## 2026-09-24 online/warmup 准备

- 从134条成功示教计算独立7D delta-chunk action stats：1,959个C10窗口、19,590个动作向量；未复用plug_v2 14D统计。
- online配置保持上游Ethernet默认：C10、20Hz、gamma .99、std .002、reference dropout .5、delta10、warmup BC10/Q.1、online BC5/Q.1、actor/critic 2×256、LR1e-4、tau .005、batch128、warmup600 transitions/20,000 updates、UTD5、stride0。
- 新右臂适配只把14D Task2反馈切成物理索引7–13，策略只发布右前臂7D；HIL以右后臂为专家。安全步限、Session、Task5和replay终局语义继续使用已有审计边界。
- Cobot入口：`rlt_v3_up.sh`、`rlt_v3_status.sh`、`rlt_v3_down.sh`；Ctrl+C停止Session/在线进程但保留Stage 1模型，down才释放模型显存。
- 尚未启动CAN、ROS、相机、持续模型服务或机械臂动作；需操作员现场执行`WARMUP_HANDOFF.md`。

## 2026-09-24 Stage 1 发布与 Cobot 无动作验证

- `step_4999` 推理包已从HPC经A6000完整同步到Cobot；A6000与Cobot均为31个文件、15,428,192,821字节，`params+assets` tree SHA-256均为`e73ef3f30c72bc946e7f3e9b5f73d5e97c99021716c0004f8d90aefe40e31270`。
- 修复Cobot混合Python环境中的初始化顺序：必须先加载PyArrow/Torch再加载JAX，否则LeRobot data config可能在`libarrow.so`中段错误。该修复只影响服务启动可靠性，不改变模型、权重、变换或推理数学。
- Cobot `validate-only` 完成真实checkpoint恢复、参数树匹配和三次固定输入推理，结果为`passed`；首次JAX编译17,106.47 ms，随后为76.71/76.84 ms，输出合同为`50×7`动作、前10步reference chunk和2048D RL token。
- 验证回执SHA-256为`9ae65a008b80dd8194766b996b0e8f7c8ccdbdf8f568f12108facab41ff0d975`，明确记录`robot_publishers=0`。验证后进程、端口与GPU占用均已退出。
- 发布状态为`offline_validated`，表示可进入操作员现场reference/warmup验收；不表示真机插入成功率已经验证。

## 2026-09-24 统一控制台、模型对比与warmup入口

- 8015默认profile改为`plug_v3_yyshadow`，数据根按`demonstrations/warmup/online`显示；转换清单回填专家计数134，当前warmup/online均为0。
- 新模型目录登记`plug_v3-stage1-reference`、`plug_v3-frozen-latest`、`plug_v3-online-latest`。切换只写下一次启动选择，不热切运行Session。reference已`offline_validated`；frozen/online在actor snapshot和`ready_for_online`成立前显示`awaiting_warmup`并禁止启动。
- Reference、Warmup、Frozen、Online分别映射固定`rlt_v3_up.sh reference|warmup|frozen|online`；Stop只结束Session/online角色，Down调用`rlt_v3_down.sh`释放模型。Warmup复用Stage 1 reference作为基线，不能与frozen/online模型错误配对。
- 页面显示35项核心合同：Stage 1、7D/三相机模型、C10/gamma/std/dropout/delta、warmup与online权重、actor/critic网络及学习率、600 transition/20k、batch/UTD、replay、20Hz/H10和验证延迟。
- 诊断数据来自`runs/plug_v3_yyshadow/online/metrics/learner_metrics.jsonl`和`learner_status.json`。已接入critic loss、Q1/Q2/target Q/actor Q、actor objective、episode位置Q、决策chunk Q、actor修正、batch组成、进度和发布状态。没有telemetry时保持空图并解释等待条件，不伪造曲线。
- 验证：定向Python 77 passed（4.12s）；Node的camera sync、capture profile、device control、diagnostics、path picker、replay timeline、unified console七组通过；Python/Node静态检查和实时HTTP合同通过。8015仅重启网页服务，当前模型仍未加载。
- 存储：USB盘`/dev/sda2`为13T，使用3.4T，可用9.4T。10:48曾发生UAS I/O/断连，现场重新插拔并于10:50重新枚举；此后没有新的块设备I/O错误，当前数据与UI均可读。该状态是动态快照，训练或真机前仍应复核。
- 安全边界：本轮没有配置CAN、启动ROS/相机/RLT、申请GPU、训练、归位或发送机械臂动作，也没有把离线验证称为真机成功率。

下一步：现场先固定`plug2`位姿、三相机和右臂夹持条件，用Reference做少量基线并采集warmup；以transition计数达到600后执行20,000次warmup。离线检查critic区分、actor偏移和有限性后才解锁Frozen/Online，先Frozen A/B，再开始在线RL。

## 2026-09-24 Reference 现场启动门禁修复

首次从统一网页启动Reference时，Stage 1模型已恢复，但Machine B在20秒后退出，网页保持Start禁用。根因有三层：网页设备任务的工作目录令YAML相对artifact路径落到错误的`/media/agilex/Getea1/runs`；现场`http_proxy=127.0.0.1:7898`把127.0.0.1 actor/replay/Task5请求送入代理；上游readiness把actor version -1（warmup前合法的reference fallback状态）误判为未就绪。

修复后`rlt_v3_up.sh`固定在配置目录解析artifact、清除loopback代理并为reference/warmup显式允许未初始化actor；上游启动器在创建learner前主动探测replay，learner启动请求使用现场配置的超时。统一API在plug_v3没有lifecycle supervisor文件时以真实Session端点回填`ready_disarmed`，启动RLT会自动切换控制台模式，但仍保留显式“现场 Arm”安全门，绝不自动arm或开始动作。

现场只启动Reference后端验证：replay ready size=0、actor ready version=-1、learner等待600 transitions、Session=`disarmed/policy_paused=true`、网页=`ready_disarmed`；没有调用arm/start，没有发送策略动作。平台回归77 passed，Node控制台/设备测试通过。
