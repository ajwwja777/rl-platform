# EXPO 左臂双腕 RTC-SFT 网页接入

日期：2026-10-13。负责人：/root。状态：完成网页接入与暂停加载；真机首轮待操作者验收。

## 授权与范围

用户指定 `/home/agilex/xiqing/models/expo-ft/plug_left_2cam/rtc_sft_20261013/step_2000/`，要求网页部署并沿用其他模型的操作/HIL。只接入固定 RTC-SFT Actor，不启动在线学习或机器人动作；加载保持暂停。现场初查 model offline、无活动录制/operation、GPU 空闲。

分支：RL/Web 各 `feature/expo-web-20261013`；独立 worktrees `scratch/rl-platform/expo-web-20261013/{rl-platform,cobot-web}`。基线 RL d138611 / Web 872b384。VLA 现有 docs/MIGRATION.md 修改保留。

## 已确认的数据与模型

本批为左臂 `0:7`、两腕 RGB、10Hz，原始49集，实际 TRAIN44集5331帧、DEV5集591帧；不是旧右臂134集/30Hz。H16、replan8、训练前缀0..10、绝对关节rad/夹爪m。原生 Orbax RTCLearner agent+actor_params，独立 loader `/home/agilex/xiqing/projects/expo-ft/scripts/cobot_plug/deployment.py`，使用包内 norm。无在线EXPO critic/edit actor。

## 实现

RL `methods/expo_ft/cobot_adapter/{core,policy_server,ros_runtime}.py` 与 `scripts/expo_left_up.sh`：原生现代Python模型进程和系统ROS Python分离，以loopback8043通信；只发布左臂策略话题，启动暂停。默认逻辑/发布10Hz、RTC两步200ms前缀异步预取；前缀取限幅后实际计划端点，沿用原生input transforms归一化。加载时预编译boot/RTC，不将编译停顿放入动作执行。

部署守卫每逻辑步六关节0.03rad/夹爪0.004m、夹爪0..0.08m，默认无滤波。网页可选20/30/40/50Hz仅插值物理发布，训练时基仍10Hz；可选EMA08和无RTC。部署限幅/调度/可选滤波属于工程适配，不能称原作者完整在线算法复现。非有限输出、过期输入、RTC前缀预算超时、发布迟到50ms触发保护暂停，不补发旧动作。

接管采用既有协调器pause服务和mode；手动/保护暂停与HIL分开，generation使旧chunk/迟到GPU返回失效。交还不解除手动暂停；正常交还从新RGB/真实左臂反馈重规划。仅沿用现有control归位，不自动使用旧右臂位姿。

Web `deployment.py`、`execution_options.py` 薄适配：EXPO独立family/id，采用现有受管理VLA生命周期与普通模型采集/评测；展示真实PID/model_id/心跳、暂停和故障。录制、HIL标签、成功失败、历史、用户所选归位沿用现有普通采集/部署入口，算法不import web Python。

登记 id `expo-plug-left-2cam-rtc-sft-2000`，场景 `plug_left_2cam`，family `EXPO`，method `rtc-sft`，step2000。记录与权重仍由原本资产目录归属，不改RLT数据/权重/默认入口。train=false；HIL可操作不等于在线学习。

## 验证证据与边界

14项新RL行为回归通过，包含完整发布循环的前缀/端点一致与物理子步中途HIL；Web相关46通过/1既有可选集成跳过。初次Web worktree测试缺dagger sibling失败，按真实依赖根/scratch链接修正后通过，未改生产环境。

Cobot不可变bundle37文件SHA256全部匹配。隔离GPU运行原生step2000、boot/RTC两条路径；四次(0/2/2/0前缀)输出均有限8x7，RPC约105.5/97.2/96.2/89.5ms。没有ROS发布者或机器人动作；隔离模型已退出。原生只读Orbax缺可选metrics提示保留，不伪造metrics。

初始/loader审计、CPU日志位于 `scratch/rl-platform/expo-web-20261013/`；现场隔离证据 `rl-platform/runtime/verification/expo-web-20261013/`。模型成功率、真实HIL操纵与现场deadline尚未验收，不以CPU或保存样本推理替代。

## 现场交付完成

运行代码先RL b684395/Web630fa2f提交push后同步9文件，逐SHA匹配，旧登记列表逐条未变。快速HIL交还补丁106e63f增加pending RTC起始时钟及实循环回归；合并时保留其他对话e6db138 teacher研究提交，源合并4aeea97。只在EXPO暂停、无Episode/录制时卸载并替换一个runtime模块后重新暂停加载。

现场14项CPU行为测试通过。首轮测试副本按相对路径寻找runtime失败，改为已导入core位置后通过；现场Web生产环境没有pytest、RLT测试环境没有fastapi，未安装或混改环境，直接在实际Web环境用标准库patch执行3项相同接口断言通过；所有初次失败日志保留。

Web正式8015重载（旧PID795010退出），EXPO登记available，输入/输出路径、capabilities及10Hz原生RTC显示正确。真实POST加载原生step2000至ready/paused；无错误，manual_pause=true，published=0。实际/api/collection/model的Session开始/结束均只暂停，session_active正确切换，策略命令保持0，录制idle。最终修订版加载PID2057530，无活动Episode/Session，policy fault为空。保留模型等用户手动开始；没有真机动作、归位、硬件重启、录制、EXPO在线更新或RLT资产修改。

6个硬件PID/命令身份及4份RLT权重/norm SHA一致；最终完整回执A6000 `scratch/rl-platform/expo-web-20261013/field-evidence/`，现场 `runtime/verification/expo-web-20261013/`。最终paused回执 `web-loaded-paused-final.json`；旧首轮paused/Session回执保留不覆盖。

网页使用：刷新正式8015，在场景plug_left_2cam → EXPO → rtc-sft → 2000选择模型。模型已加载且暂停，可按现有部署/采集/HIL/标签/归位按钮操作；本批只有左臂和两腕，默认动作时基10Hz。归位按用户选定的左臂/位姿，不自动套用旧右臂位置。可用HIL控制和采集不代表在线学习，train=false。

唯一下一步：现场用户短轮验证Start/Pause/HIL交还/终止/归位，再验收成功率与实际deadline；不自动开始动作。

## 2026-10-13 原生10Hz选项兼容补充

用户询问Hz/RTC/滤波选择是否可用。发现旧前端仅提供20/30/40/50，下拉框无法显示EXPO默认10；单独切RTC/滤波时payload保持10，被Web通用校验拒绝。修复RL afaed39（完整commit以Git为准）及Web a8d7a8c：EXPO显示10/20/30/40/50并接受原生10独立开关；其他模型仍保留原频率契约。训练时RTC权重保持，部署RTC=true调用原生前缀/异步路径，false走delay0同步路径；滤波是可选EMA，Hz只改发布插值，不改训练10Hz。

15执行行为、45Web相关（1既有跳过）、9前端用例通过。先commit/push再3文件现场SHA同步，实际Web环境10Hz开关解析通过；仅重载idle网页，原EXPO PID2057530与执行10Hz/RTCtrue/滤波false保留。用户此前已运行并暂停，发布累计765；本次前后计数同为765，未产生新动作、录制或模型加载。JS正式HTTP SHA匹配。证据scratch/rl-platform/expo-web-20261013/{rl-native10,web-native10,ui-native10}-tests.txt及field-evidence/native10-field-delivery.json。

EXPO界面开关变更需释放后重新加载，不热切换当前运行模型；默认Hz不覆盖/RTC开/滤波关作为基线。真机执行/成功率不由该UI修复认证。
