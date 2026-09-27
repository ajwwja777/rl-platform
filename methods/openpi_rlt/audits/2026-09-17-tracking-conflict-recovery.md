# 2026-09-17：command/tracking冲突的暂停收尾修复

## 现场证据与数据收尾

用户报告unknown_fault/backend_unavailable: URLError。8015正常而internal8016拒绝连接；interface_20260917_142544.log在实际actor7816的chunk495/496后抛出command/tracking bounds conflict，env异常退出。仍无法仅凭异常确定实测偏差来自接触、手动位移或状态时序。

EP303原始HDF5由max_timesteps于120秒停止，3600/3600帧完整、committed。文件UUID为5bb58def-96ea-407d-a4a9-00561e19a3ee。经已有PUT labels接口按基础设施故障放弃，episode_outcome=aborted、episode_quality=bad、keep_for_training=false、termination_reason=safety_stop并保留中文说明；HDF5未删除或改写。不将本轮伪装成有效成功/失败训练数据。PUT响应15秒超时，未盲目重试；随后读取实际sidecar确认已完成，console writer lease释放。较早全库list请求12秒超时，不作为数据结果。

mounted recorder路径为/api/rlt-recorder；POST storage/prepare返回latest303/labels_complete=true/label_blocked=false/next304。console active_mode=null、writer_token=null、ROS readiness ok。current.json仍actor7816/global15632/transactions207，未训练新参数。

## 修复

- action_conditioning新增CommandTrackingConflict(ValueError)，保留原命令连续性/跟踪限制及冲突reset行为。
- execute_chunk只捕获该类型：chunk_ready=false、不发布失败动作、不生成该失败动作样本，清除平滑历史，Session进入terminal_pending等待操作员成功/失败/放弃；不会自动reanchor后恢复动作。
- 原有暂停采样分支继续排除结果等待时间；真实操作员终局仍由既有记录/固化流程处理。
- terminal_pending支持与网页Pause同时发生；若已有终局/停止决策则保持该决策，避免回滚状态或异常退出。
- supervisor在角色退出时记录machine_b_<role>_exited_status_<status>错误码，供后续故障页展示，不只写phase=fault/error_code=null。

## 验证与部署

新增可复现测量突变的execute_chunk回归：修复前在同一位置抛异常；修复后只发布1条有效命令、排除冲突尝试和等待间隔，保留操作员failure终局。新增暂停与终局并发回归。

A6000相关55项无动作测试通过；Cobot formal overlay环境31项env/session HTTP相关测试通过；bash -n通过。测试过程中两次pytest临时目录父目录缺失，补建登记的scratch目录后重跑通过；不是代码测试失败。

正式部署与逐文件hash记录：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/deploy-backup-tracking-conflict-20260917/deployment.json

备份保留原文件，部署前核验Machine B端口关闭及未发生并发文件修改；main项目canonical与Cobot formal overlay同步action_conditioning.py、cobot_online_env.py、session.py、session_http.py、interface_task2_teach_rlt_online.sh及两项测试文件。部署备份/incoming保存轻量传输内容，不含凭据。

## 交还状态与下一步

本轮未启动模型、后端、训练、Arm、rollout、ROS publisher或归位；网页无需重启，Machine A 8000仍预加载，8016/9101/9102关闭。backend旧fault快照会保留到下次up，不能说后端已ready。未修改模型权重、损失系数、探索参数或平滑限制，未提交/push；其他项目修改不动。

用户下一步可在既有plug部署目录执行./rlt_up.sh 4000 --frozen-actor，加载现存actor7816，统一8015就绪后由现场明确Arm并Start，做排插位置探针。冻结只停更新，不自动隔离未来训练数据；这几次应标位置探针，不能当正式留出评估。正式20轮A/B计划仍需专门评估隔离。

若再出现该约束冲突，应看到terminal_pending及原因command_tracking_bounds_conflict；不要强行继续动作，按真实情况选择失败/放弃后依既有流程收尾。动作/归位安全限制不变，实体复验仍待现场。
