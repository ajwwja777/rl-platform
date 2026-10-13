# EXPO 左臂双腕 RTC-SFT 网页接入

日期：2026-10-13。负责人：/root。状态：部署验证进行中。

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

13项新RL行为回归通过，包含完整发布循环的前缀/端点一致与物理子步中途HIL；Web相关46通过/1既有可选集成跳过。初次Web worktree测试缺dagger sibling失败，按真实依赖根/scratch链接修正后通过，未改生产环境。

Cobot不可变bundle37文件SHA256全部匹配。隔离GPU运行原生step2000、boot/RTC两条路径；四次(0/2/2/0前缀)输出均有限8x7，RPC约105.5/97.2/96.2/89.5ms。没有ROS发布者或机器人动作；隔离模型已退出。原生只读Orbax缺可选metrics提示保留，不伪造metrics。

初始/loader审计、CPU日志位于 `scratch/rl-platform/expo-web-20261013/`；现场隔离证据 `rl-platform/runtime/verification/expo-web-20261013/`。模型成功率、真实HIL操纵与现场deadline尚未验收，不以CPU或保存样本推理替代。

唯一下一步：串行提交/push，同步校验并在确认现场仍idle后重载Web、加载EXPO为暂停状态，回写实际结果。
