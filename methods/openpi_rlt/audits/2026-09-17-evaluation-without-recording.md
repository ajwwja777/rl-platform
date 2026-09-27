# 2026-09-17：不保存数据的固定actor检测入口

## 需求与语义

用户这次移动排插检查目标跟随，只需现场观察，不保存rollout数据。旧--frozen-actor只冻结参数，仍录制成功/失败并写replay，之后可能作为训练输入。保留旧语义，新增--no-record，必须同时指定--frozen-actor，避免“未采集数据却继续更新”的混合模式。

检测命令：在既有plug部署目录执行 ./rlt_up.sh 4000 --frozen-actor --no-record。

- 保留ROS/相机实时门禁、现场Arm、Start/Pause/HIL、操作员终局、all --pose plug及复位后下一轮。
- 不调用Task5 start/finish/status，不产生新HDF5、labels或动作trace，不写replay，不请求在线更新。
- 成功/失败只用于当前内存Session终局与既有归位流程；结束Session使活动检测轮放弃，等driver完成后退出，不额外归位。
- 普通服务日志与轻量状态仍保留用于故障诊断；这不代表保存检测样本。没有删除此前已录制的历史数据。
- Session status明确recording_enabled=false/replay_eligible=false。原先--frozen-actor单独使用仍为固定actor采集模式。

## 实现与回归

RltSessionApplication在不录制模式直接进入recording_ready的控制状态，不申请录制writer lease；终局不调用Task5固化，但保留driver/Session收尾同步。CobotOnlineEnv明确禁止raw-step写出和replay提交，drop_transition=true，不触发request_cycle。RosTask2IO不创建AtomicEpisodeTraceWriter，即使被调用record_raw_step也不会写文件。

backend_lifecycle转发/核验--no-record并在现有backend配置不一致时拒绝复用；online及continue5416两个实际入口均检查--frozen-actor约束并设置COBOT_RLT_NO_RECORD。顺带把上轮角色退出错误码补修同步到实际continue5416入口（原先仅online入口带该错误码），不是调整训练或控制参数。

A6000和Cobot各52项相关测试通过，覆盖成功与下一轮、活动检测轮结束、无Task5调用、无trace writer、无raw/replay/更新请求、正常采集及启停回归；shell语法与缺少frozen参数的早期拒绝通过。尚未执行真机Start或动作，不能把无动作测试称为现场验收。

## 安全切换与动态快照

只在旧Session已暂停、waiting_scene且本轮aborted时，调用既有stop（旧episode3/generation25），没有动作或归位；Machine B退出、writer为空，保留Machine A PID2220940。

正式文件备份与部署hash：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/deploy-backup-no-record-20260917/deployment.json

同步10个main项目源码/测试文件到formal overlay；未改Task5平台源码，不删任何历史数据，不提交/push。

2026-09-17T07:49:14Z新后端ready_disarmed：supervisor2456065、replay2456222、learner2456503、actor2457102、env2457384；internal Session phase=disarmed/policy_paused=true/chunk0/recording_enabled=false；learner training_enabled=false/0步，实际服务actor7816验证通过，current global15632/transactions207不变。

日志：/home/agilex/cobot_magic/task3/jiaan/runtime/cobot-rlt-backend-v1/supervisor-20260917T074858Z.log
及当前run logs/interface_20260917_154858.log。

这些PID/端口是检查快照，使用前重新核验。下一步用户刷新统一8015，确认检测模式与现场准备，再明确Arm/Start执行位置探针。结束Session保留模型；切回在线采集需等backend offline后执行./rlt_up.sh 4000（可加--explore）。需要完全释放模型仍用rlt_down.sh 4000。
