# Teacher5 8127 现场交付（2026-10-13）

状态：**已验证无动作Online加载，可交由用户首轮受控采集**。用户明确批准结束EXPO会话后，通过正式网页API结束其会话、释放模型并加载Teacher5 Online；真实Stage1元数据、Actor RPC和Learner状态核对完成，未开始Episode或机器人动作。独立自主收益仍证据不足。

## 改动与来源

实现提交 `f4bae5ca8e7efb3c40e5bcecc749c490e6e98bb7`，A6000独立worktree `scratch/rl-platform/teacher5-delivery-20261013/rl-platform`；先提交push，再按SHA清单同步4文件。

- 登记 `plug-v3-teacher5-8127`（冻结）与 `plug-v3-teacher5-online`（分批Online），方法名 `supported-mc30-hil8-teacher5`，初始Learner8127/Actor4063。保留旧5k/7k/HIL8/EXPO入口和默认。
- 修复复制工具：HIL8-v2包必须携带初始pending_actor，缺失时在创建目标前失败；复制保留待晋升身份，不以served伪造pending。
- 原生CPU验证脚本移除离线训练runner依赖，可在现场直接验证。未改训练数学、固定上游或已有生产Replay/权重。

候选来自 `projects/rl-platform/outputs/review-validation-20261013/candidates/teacher5-8127-v2`。训练版本/逐项实验、旧review错误及能力边界见 [REVIEW_VALIDATION](REVIEW_VALIDATION.md)。相对HIL8训练仅teacher50→5；global7281续846次Critic/423次Actor更新；固定原5k teacher、LR/MC30/HIL8/BC5/Q.1/delta10保持。

## 交付身份和独立目录

Cobot不可变种子：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_20261013/`。

Cobot独立Online运行目录：`/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/`。内含profile、norm、teacher、checkpoint、served/pending Actor、Replay和online.yaml。仅YAML路径按目标重定位，其余资产逐SHA等于已验证原包。

网页采集专用目录：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/teacher5-8127-online-v1`。

初始Replay为TRAIN3623窗/231Episode（旧2777+新846），排除旧20Episode DEV和新6Episode DEV；历史DEV反复用于选型，无独立TEST。新数据动态比例以真实batch收据为准，不固定承诺成功:失败8:2。

执行沿用async_rtc50：逻辑20Hz/C10、发布50Hz、RTC、单级EMA tau=.08，固定夹爪。不额外启用冻结探索入口的episode-constant偏移。逻辑Replay仅保留每20Hz区间最后发布动作，中间50Hz命令留trace，不能称全50Hz轨迹训练一致。

## 已验证

- A6000和Cobot各5项目录复制/登记专项回归通过。
- 重定位后A6000与Cobot原生CPU加载、8vs4+4续训149状态叶完全一致，双采样RNG及8批身份一致；32真实输入Actor最大绝对差 `1.1920928955078125e-07`。
- 3623条Replay键无重复，当前最大Episode10039，原生下一Episode10040；不会新小号误入旧数据次序。
- 旧数据训练预算0；合成新增1/18/39条预算分别1/18/39。验证仅内存演练，不写运行Replay/checkpoint。
- 12个profile锁定训练源码SHA与现场一致；导出/运行norm及teacherSHA保持。
- 网页正式目录两个入口均available，身份明确冻结/Online；静态安装期间EXPO进程与录制状态保持。
- Online checkpoint每50更新保存，pending每50更新写出；staged不自动晋升served，初始两者Actor4063。

## 验证边界和下一步

真实Stage1/GPU及Online各角色已加载，`ready/disarmed/policy_paused`、执行发布0、推理请求0、录制idle。Actor RPC4063，Learner8127/caught_up/budget0，Replay3623。尚无新Episode、真实入库/学习或自主收益。沿已批准流程，由用户从数据采集启动首个完整Episode；再核对录制→trace→Replay新增身份/窗口→预算→实际batch→pending权重，不直接连续放任训练和自动晋升。停止条件和回退沿既有分批Online流程，旧HIL8/5k入口保留。

## 可恢复证据

A6000完整产物：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/teacher5-delivery-20261013/`。

命令/结果：`install-manifest.json`、`install_candidate.py`、`install-command-result.json`、`field-install.json`、`unit-tests.json`、`field-unit-tests.json`、`relocation-native-verification.json`、`field-native-command.json`、`native-verification.json`、`bootstrap-check.json`、`catalog.json`。最初嵌套SSH shell引号导致taskset相对Python路径失败，未执行验证；改用Python结构化SSH参数和绝对Python路径后通过，不隐去失败。

现场同步原件及收据：`/home/agilex/jiaan/project/rl-platform/runtime/verification/teacher5-delivery-20261013/`，`before/`保留旧源码。加载操作回执同目录。笔记本无新增文件；guide仅摘要、不提交推送其Git。

## 真实暂停加载补证

2026-10-13约21:40CST加载完成。主进程2489739；Stage1握手checkpoint为NVMe `reference_4999`、RTC支持true；Actor9141实际version4063、Replay9142实际3623/maxEpisode10039；Learner实际8127/4063/新增budget0。执行profile=user_options，逻辑20Hz、发布50Hz、RTC和单级EMA .08启用；session_use=collection，session disarmed/policy_paused，emitted_commands=0、inference_requests=0，recorder idle/0帧。没有自动准备/启动Episode、归位或运动。

Stage1元数据模型/参考采样率30Hz、H50，执行逻辑20Hz/C10；这是沿用的既有时基边界，本次未重新训练RTC或更改时基，不能将加载通过当作30/20Hz语义等价证明。

初始served Actor、完整checkpoint、Replay、profile、teacher和norm逐SHA保持。原生Learner启动会重新导出当前pending Actor；第一次“全部字节不变”检查因此失败。进一步确认仅pickle字典顺序不同：12个Actor参数叶完全一致，版本4063/global8127/rl_config完全相同；没有训练或晋升served。失败回执 `paused-load-byte-check-first-attempt.json` 与纠正后的 `paused-load-verification.json` 均保留。Actor `/healthz` 是纯文本OK，不能当JSON解析；身份改用原生 `/version` 二进制RPC确认。

旧preflight启动日志仍检查原生产11000/Actor5500资产，并非本候选身份；本次以随后真实supervisor运行目录、Actor RPC及Learner状态证明8127/4063，不采用preflight旧资产输出作当前版本证据。

网页已有方法 **supported-mc30-hil8-teacher5**，选择 **8127 · Online · Actor 4063** 对应ID `plug-v3-teacher5-online`（另一条8127冻结版不学习）。加载时已明确collection用途/50Hz/RTC/滤波；存储API已设置并GET核对专用 `teacher5-8127-online-v1` 目录。用户刷新数据采集页后准备会话并开始首轮；动作只由用户触发。

新增回执：`expo-session-stop.json`、`expo-unload.json`、`load-request.json`、`storage.json`、`paused-load-command.json`、`verify_paused_load.py`、`paused-load-verification.json`。EXPO原入口及资产保留，现场当前改为Teacher5；guide/live记录已交还用户。
