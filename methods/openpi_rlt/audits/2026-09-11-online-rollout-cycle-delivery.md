# 2026-09-11 在线 rollout 交付与现场验收

已部署软件闭环；尚未由本次工作验证真实机器人效果。旧 interface_task2_teach_rlt_live.sh 保持 Stage 1 采集用途，新入口为 interface_task2_teach_rlt_online.sh。

## 使用流程

在 Cobot 执行 conda activate aloha，source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash，进入 /home/agilex/cobot_magic/task3/jiaan/realworld_rl/openpi_rlt_plug。LIVE 前由现场确认 CAN、Task2、三相机和 all --pose plug 初态。

1. 无动作检查：./interface_task2_teach_rlt_online.sh 4000 --shadow。独立 shadow run，不发布动作、不归位、不进入正式 replay、不更新。
2. 建议首次真机固定候选验收：./interface_task2_teach_rlt_online.sh 4000 --frozen-actor。使用 warmup actor，零更新；有效数据可供之后采样，但不会在取消冻结后按这些旧轮次补发更新。
3. 完整自动更新流程：./interface_task2_teach_rlt_online.sh 4000。默认确定性；确认动作后可加 --explore，启用 std 0.001 的 chunk 探索。
4. Enter 仅 arm，网页 http://127.0.0.1:8016/ 点击开始 Session 才开始。HIL 释放后暂停；成功/失败后固化有效数据、32 learner updates、固定验证集回归检查、发布通过的 actor，随后 all --pose plug 归位并暂停。现场复位物体后点击下一轮。放弃与 shadow 不触发正式更新。
5. 网页结束 Session 后 interface 及其创建的 Machine B 进程退出，Machine A 常驻。全部结束可执行 ./stop_rlt_model_server.sh 4000。
6. 回退：先结束 Session 并等终端返回，再 ./rollback_rlt_online.sh。回退最近一次接受的 actor/checkpoint，保留已处理 episode 标记；运行时拒绝回退。

## 模型与更新契约

初始 seed 是离线 learner step10000 / actor5000，不是最终 step20000。主运行根 /media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/plug-online-warmup-r1；shadow 根同前缀加 -shadow。重启恢复最近接受版本，不重新 warmup，不重复消费已处理 episode。

每个新有效完整 episode 触发32 learner updates，actor 每2步更新，因此通过检查时版本通常增加16。一轮内固定版本，下一轮才切换。训练按专家成功/当前成功/当前失败 38/51/39 (batch128，约30/40/30) 分组并按 episode 采样，验证集不参与训练。BC10/Q0.1/delta10，沿用运行时动作连续性限制和 HIL 暂停。

验证要求 score、velocity、acceleration、first、boundary 全部 finite，且同时不超过上一次接受版本与固定 seed 指标的 1.10 倍加 1e-4。不通过保留旧 actor，网页明确提示。此门槛限制相对退化，不证明候选优于 Stage 1：warmup 候选模仿误差及原始加速度尚未优于 reference，真机效果待现场验收。

启动时预编译两次私有更新后恢复完整状态和 RNG，不计正式更新。旧自动 warmup 未重新开启，新 worker 仅在轮次边界按预算更新。

## 验证证据

53项相关测试通过；UI纯逻辑检查通过（无浏览器视觉端到端验收）。Cobot真实 checkpoint + GPU learner + CPU Actor HTTP 的隔离数据集成验证两轮 actor5000→5016→5032，轮内 pin、重启去重、回退5032→5016全部通过；正式旧 replay 哈希不变。固定 actor 模式验证0更新。测试未创建 ROS publisher、未操作机器人。

预编译后两次更新全流程1.616/1.721秒，包含训练、验证、保存和actor确认；不含录像固化、特征补齐、机械臂归位，也未与真实 Machine A 并发压测。首次 Machine A 已无预加载进程，仍需冷加载。

部署备份 /media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/deploy-backup-online-cycle-20260911，含原文件及新文件hash。runtime-overlay.sha256与deployment.json同步，旧入口SHA保持0b8cb505e1087a82a206d0bfd3547cb8d85b11cbcadc24a71ad67ab135e351e9。resolve_online_config.py以已部署版本反向对齐canonical，保留既有action-stats参数和默认值，未覆盖生产resolver。

机器证据见 online-cycle-20260911/。测试进程已退出，未启动正式新session，未commit/push。其他项目修改未纳入。
