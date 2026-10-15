# 本任务 HPC 位置与资产边界

2026-10-15 用户明确授权申请4或8卡并训练，选择单节点4卡、acd_u/formal-user、48CPU/384GB、最多12小时。实际GPU型号由分配节点核对，不沿用历史硬件结论。

- 登录：jhe724@hpc3login.hpc.hkust-gz.edu.cn，实测ACD-Login-1；旧HPC3_jhe724地址10.120.48.26返回无活动作业拒绝，未修改共享SSH配置。
- 新项目根：/data/user/jhe724/jiaan/research-workspace/projects/rl-platform。
- 本任务快照、数据和输出：上述根/outputs/realtime-expo-rlt-20261015/；source为独立提交快照，inputs为已校验5922帧派生数据，stage1为训练输出，smoke为独立小步恢复检查。
- 只读复用既有环境：/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl/envs/rlt-stage1-py311-relay-v1；已实测Python3.11.15/Torch2.7.1/JAX0.5.3/Flax0.10.2/pytest8.3.5。
- 只读核验并复用base：/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl/cache/openpi-assets/pi05_base/params，须与A6000 base-identity.json逐文件匹配。
- 不改旧项目数据、环境、checkpoint，不在登录节点训练。新代码采用已提交源码、独立固定上游Git快照；传输校验后才提交训练。
- batch作业先GPU小步训练/独立进程恢复，通过后正式5000更新；固定4卡、batch32、H50/10Hz。batch32为恢复原RLT常用批大小，相比初始CPU预备batch8须重新核对真实输入；不改网络/loss/precision。
- 作业结束自动释放，失败即停止后续步骤并保存日志。正式每1000步保留checkpoint；DEV只用于离线评估，不作为独立TEST。
- 状态/命令/作业ID保存在本任务STATUS及A6000 outputs/hpc-status.json，未记录作业ID前不能声称已申请成功。
