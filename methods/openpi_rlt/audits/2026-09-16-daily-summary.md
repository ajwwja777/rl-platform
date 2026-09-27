# 2026-09-16 Cobot统一网页与RLT工作总结

## 今天完成的工作

- 普通节点采集、RLT操作与历史审核统一到8015，共享三相机和单writer。普通历史节点/视频保留，RLT历史审核使用/rlt-review/，不再并开独立录制。
- 新增rlt_up/rlt_down：网页与后端独立启停、后台加载进度可见、显式Arm后Start；相同配置复用，结束Session保留Machine A，完整down释放模型/GPU，最后stop网页。
- 存储准备/UUID/标注按精确系列与当前episode处理；历史补标不能释放另一条writer占用。新增ROS master/订阅/新鲜相机门禁与PID/generation原子生命周期记录。
- 修复真实现场recorder_not_ready和handover_stale：锁存handover_mode是状态而非心跳，只要求已收到并核验当前发布节点在线；相机/teach_active仍实时检查。
- 修复相机MJPEG长连接拖住UI退出：设置有界graceful shutdown；真实预览连接打开时3.35秒正常stop，重复stop/restart通过。
- 修复Arm后Start无响应/task5_start_failed：async代理直接阻塞等待8016，而8016回调同一8015录制healthz，形成回环阻塞；改worker线程，真实HTTP回环回归先失败后通过。
- 明确explore与续接规则：默认eval仍在线学习；--explore只增加原有std0.001、右臂chunk平滑探索；平滑alpha0.35/step0.01、夹爪hold、32 updates/有效轮次、验证拒绝及下一轮actor切换保留。不重新warmup，不回到旧5000。

## 验证

正式部署相关RLT52项、Task5基线范围327项、JS6+2+1项通过。补修锁存/录制/脚本71项、HTTP回环相关11项通过；这些套件部分重叠，不累加成独立测试总数。真实ROS ready、正式up preflight、相机预览下stop/restart通过；recorder healthz约0.008秒。

扩大套件仍有环境/既有失败：Stage1缺依赖与固定路径，converter缺pyarrow，旧gate异步断言与队列排空语义不符。限制见审计，不声称全套测试通过。git diff --check和doctor验证记录见当日审计；doctor0 error、7条既有legacy attachment warning。

## 在线运行最新检查点

修复时保存actor7160/global14320/processed175；今日发布前只读核对同一run已保存actor7592/global15184/processed218。这是参数与去重列表推进的证据，不等同于43条新自主rollout，也不证明成功率提升。未在本次总结任务启动训练、服务或机器人动作。

运行根：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/plug-online-from5416-20260914

发布前8015 status/session无法连接，服务终态与GPU需下次重新检查，不自动重启。较早故障轮chunk0/no UUID/replay不合格，没有进入训练。EP227曾label_blocked，后续标签状态未重新核验，不把旧待办当作当前已确认阻塞。

## 下一步

1. 按[SOP](../UNIFIED_CONSOLE_SOP.md)现场验收新的加载→Arm→Start→Pause/HIL→终局→all plug→下一轮actor→结束→完整退出/再开。
2. 固定条件对比reference、固定warmup与持续更新actor，分别统计自主成功、HIL成功、失败/放弃；参数更新不能代替效果评估。
3. 右夹爪高度停止尚待标定/去抖/验证；其他demo入口统一与自动转换删除HDF5仍独立待办。

[Todo](../UNIFIED_CONSOLE_TODO.md)；
[统一生命周期审计](2026-09-16-unified-console-lifecycle-delivery.md)；
[锁存门禁修复](../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-16-latched-mode-readiness-fix.md)；
[HTTP回环修复](../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-16-session-proxy-reentry-fix.md)。

## 发布范围

本次GitHub发布为轻量框架文档：本日总结、SOP/Todo/审计/阶段交接/设计计划、两项目状态/决策及Latest News。完整Task5代码仍在Cobot独立工作树，RLT代码未提交内容保留；不在框架提交中顺带纳入旧代码、FluxVLA、Franka或其他交接文档。
