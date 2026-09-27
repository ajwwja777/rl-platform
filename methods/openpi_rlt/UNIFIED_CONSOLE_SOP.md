> 2026-09-18 plug_v2请优先用 [新流程](PLUG_V2_ROLLOUT_SOP.md)；下文旧5416/7592/旧相机仅历史。

> 2026-09-17当前：所有现场命令统一从外接盘cobot-platform/scripts执行。默认left-camera-v2只采集；继续旧模型需明确legacy-camera-v1 profile。456旧完整HDF5及5incomplete已按用户要求删除，旧raw回放不可用；本SOP较早数量/端口/版本均为历史快照。

# Cobot 统一采集 / RLT 操作流程（2026-09-16）

状态：已部署，接口与无动作启停检查通过；实际模型加载、真实ROS采样和真机闭环尚待现场验收。本次未启动CAN、Task2、相机、模型、训练或机器人动作。

## 入口

笔记本VS Code仅转发远端8015，打开 http://127.0.0.1:8015/ 。同一网页选择普通采集或RLT，共享三相机与单writer。普通模式保留原节点/历史/视频；RLT模式显示加载阶段、actor、暂停/HIL/终局，数据审核入口 /rlt-review/ 支持历史补标/回放且禁止另开录制。模式切换只允许writer空闲。

8016/9101/9102为内部RLT服务，8000为内部Machine A，不再作为浏览器入口。8017旧独立录制不在默认流程。纯π0.5与原始固定warmup demo独立历史入口尚未纳入本次生命周期，不混用。

## 1. 现场准备

操作员按原平台流程配置CAN、启动Task2和三相机、准备plug初态；本次未执行。后端启动要求ROS master、订阅注册、三路新鲜图像和示教状态检查通过。handover_mode是锁存状态，收到有效值并核验发布节点在线即可，不要求每秒重复发布；teach_active仍实时检查。

2026-09-16较早只读快照为226条完成HDF5、latest episode_000227.hdf5、UUID 73f67e72-a020-4fe8-be76-bcbf679eab0f，标签未完成；日终标签状态未复查。启动前以当前prepare结果为准：若label_blocked=true，在RLT审核页按实际结果补标，不自动猜成功/失败，无法确定时保留证据。

## 2. 开网页

Cobot终端：

~~~bash
conda activate aloha
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_up.sh --profile legacy-camera-v1
./scripts/ui_status.sh
~~~

网页可先启动，ROS不就绪会显示原因，不自行启动机器人服务。旧start/check/stop_task5_segmented_teach_v1.sh转发统一入口。普通采集直接使用普通模式；在线RLT先在网页选RLT。

## 3. 加载在线RLT

~~~bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/rlt_up.sh 4000
~~~

默认立即返回，后台加载，网页显示loading_machine_a→replay→learner→actor→env→ready_disarmed。loading不表示成功，必须等待就绪。冷加载历史约9分钟，本次未重测。可加 --wait-seconds 900 等待终端结果。

默认调用continue5416，续接 /media/agilex/Getea1/jiaan/projects/rlt/runs/plug-online-from5416-20260914/current.json。交付时actor7160/global14320，seed5416；不重新导入replay，不回到旧plug-online-warmup-r1/actor5000。

相同配置重复up复用；更改explore/frozen/shadow需先down：
~~~bash
./scripts/rlt_up.sh 4000 --frozen-actor
./scripts/rlt_up.sh 4000 --explore
~~~

--frozen-actor冻结当前续接actor，并非原始warmup5000对照。默认eval无噪声但使用已学习actor并在线更新；--explore增加原有受控chunk级探索。未更改探索std、平滑、右臂契约、训练参数。

## 4. 开始与逐轮

ready_disarmed后，确认急停、CAN、相机、夹爪、工作区，再点击“现场Arm”，仍暂停；点击“开始Session”才rollout。

HIL释放后保持暂停，选择成功/失败/放弃或明确继续。暂停不是终局；成功/失败固化并提交正式replay，放弃不进入正式replay。暂停/HIL采样掩码规则保留，本次未改变训练语义。

有效轮次按既有逻辑触发32 learner updates，固定验证决定接受/拒绝；accepted=false没有发布新actor。接受actor只在下一轮边界生效，轮内固定。固化/特征/更新/归位耗时本轮未重测。参数更新不等于成功率提高。

正常终局按既有LIVE流程all --pose plug，等归位后操作员复位物体再点下一轮。后臂CAN/示教或command/tracking bounds故障不自动重试、不强制恢复。

## 5. 完整退出

先网页结束Session，等待数据固化与停止，然后：
~~~bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/rlt_down.sh 4000

cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/ui_down.sh
~~~

down请求活动轮放弃/收尾，依次停受管角色并释放注册Machine A，确认相关端口关闭；不执行机械臂归位。UI独立保留，最后stop。重复down/stop可重入。新后台入口不需手动Ctrl-C/查PID kill。

writer活动/提交未完、未知端口/PID身份不符会报错并保留证据，不扫端口kill。固化中不要强制关机；CAN/Task2/相机和物理关机另按现场流程处理。

## 日志与验收

- UI runtime：/media/agilex/Getea1/jiaan/projects/cobot-platform/runtime/data-console
- lifecycle state/supervisor日志：/media/agilex/Getea1/jiaan/projects/cobot-platform/runtime/rlt
- 在线run：/media/agilex/Getea1/jiaan/projects/rlt/runs/plug-online-from5416-20260914
- 原子state、generation、PID启动ticks/子进程身份防止历史状态误认就绪。

代理固定loopback/允许接口，阻塞HTTP在worker线程执行以允许Session回调同一8015录制API；JSON/HTML错误可读并关闭请求连接；VS Code转发下旧HTTP400/501真实现场消失尚未验证，不宣称网络问题全部根治。

现场依次验收：加载无动作→Arm仍暂停→Start→Pause/HIL→终局固化→all plug→下一轮actor→Session stop→down→UI stop→再开。异常保留页面和日志，停止动作。


## 2026-09-17：固定模型且不保存样本的检测

在部署目录使用 ./scripts/rlt_up.sh 4000 --frozen-actor --no-record；统一8015 backend ready后现场Arm/Start。--no-record必须配frozen。成功/失败/放弃仅结束当前检测轮，不保存HDF5/labels/动作trace或replay，也不更新参数；原有all --pose plug及手动下一轮流程保持。Session recording_enabled=false，服务诊断日志仍保留。结束Session后等待backend offline；再次检测重复本命令，切回训练采集用./scripts/rlt_up.sh 4000。不自动删除之前检测已保存的数据。
