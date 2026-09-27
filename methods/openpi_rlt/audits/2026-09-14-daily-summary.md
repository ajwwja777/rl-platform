# 2026-09-14 Cobot RLT 工作总结

## 今日交付与证据边界

- 共用执行层增加指令 EMA alpha=0.35、每发布周期关节步长上限0.01 rad，保留测量跟踪边界及暂停/HIL重锚定。reference 与固定 warmup 两组已由用户现场试用，反馈“平滑了一些”；这是定性反馈，不是统一协议的成功率提升证明。reference 为 Stage1 联合训练模型的 reference 路径，并非独立纯 SFT 基线。
- 修复同轮旧 generation 导致网页暂停请求409，以及可选 update_request.json 的 exists/read 竞态导致 learner 退出。VS Code 转发出现 NUL/HTTP400/501 的问题仍待处理；JAX 预编译 native SIGSEGV 的根因仍未确认，不能称已修复。
- 复现在线候选连续拒绝的原因：被动左夹爪归一化目标约2399.53，污染训练损失。learner私有动作副本只保留右臂实际关节动作，被动臂和两夹爪投影为reference；原始HIL/replay数据不变，学习率、损失权重及验证门槛不放宽。
- 对37轮历史数据按顺序隔离重放：26次候选接受、11次拒绝；总尝试1184 learner步，保留832步，得到global_step10832 / actor5416。更新与验证中位1.3559秒，不含固化、特征补齐和归位；验证集参与诊断和选择，不是独立泛化评估。
- 固定5416验收入口已经部署，用户已进行该版本rollout；其成功率提升未建立。新增从5416继续在线训练的独立入口，初始化一次性导入完整review数据、先补更新再就绪，后续每有效episode更新32步，只有通过验证的actor用于下一轮。新入口完整现场闭环仍待验收。
- 普通采集8015的真实segmented_frontend已修复历史episode节点/截图切换，增加H.264视频回放及HTTP206读取。已部署；19项Python测试、6项UI测试和2项选择竞态检查、两条真实episode视频接口通过。没有可用浏览器，视觉验收仍待用户完成，原始HDF5没有删除。

## 当日已审核 rollout 数据

原warmup actor5000加平滑的有效episode24至61（排除30），总37轮：自主成功17、HIL成功11、失败9。自主成功率17/37=45.95%；(失败+HIL)/总轮次=20/37=54.05%，这是非自主成功比例，不称成功率。放弃和未完成轮次不计入。该统计不包含后续5416或新续训run，不可跨版本合并归因。

## 当前入口

Cobot目录：/home/agilex/cobot_magic/task3/jiaan/realworld_rl/openpi_rlt_plug

| 用途 | 命令 |
|---|---|
| reference，无warmup，平滑 | ./interface_task2_teach_rlt_smooth_reference.sh |
| 固定原warmup actor5000，平滑 | ./interface_task2_teach_rlt_smooth_warmup.sh |
| 固定actor5416，不更新 | ./interface_task2_teach_rlt_contract_review.sh |
| 从5416继续在线更新 | ./interface_task2_teach_rlt_continue5416.sh 4000 |

4000指基础checkpoint，不是RL actor版本。结束旧Session并等待终端退出后切换；前两组和固定5416不参与在线更新。普通online.sh仍对应旧run，不用它代替continue5416入口。
普通分段采集：在 /media/agilex/Getea1/jiaan/projects/cobot-realworld-vla/task5/segmented-teach-v1/code 执行 bash scripts/start_task5_segmented_teach_v1.sh。
端口用途：8015普通分段采集，8017为RLT自动管理的连续录制，8016为RLT Session操作。笔记本使用VS Code转发给出的实际本地地址。

## 明日：右夹爪高度停止

用户确认采用右夹爪高度作为停止触发来源，用一次正确插入位置标定；标定安排2026-09-15讨论与执行。今天未实现或启用此功能。
候选设计：读取基坐标系下实际TCP Z，必要时限制插孔XY区域和持续帧数；仅策略rollout阶段触发，排除HIL和归位。触发后先暂停发布、使旧chunk和在途推理失效，保持位置；初期成功/失败仍由人工确认，不将到达高度直接当成功标签。坐标来源、工具偏移、阈值、去抖、越界容差及异常处理明天确认。此决定不授权今天操作机器人或标定。

## 详细记录

- [平滑](2026-09-14-command-smoothing-demo.md)
- [XR-1核对](2026-09-14-xr1-smoothness-reference.md)
- [预编译崩溃诊断](2026-09-14-online-precompile-segfault.md)
- [请求轮询修复](2026-09-14-request-poll-race.md)
- [训练动作契约与5416产物](2026-09-14-training-action-contract.md)
- [从5416继续在线更新](2026-09-14-continue-from5416.md)
- [Task5网页修复](../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-14-segmented-history-video.md)

本次收尾仅更新与发布Markdown。代码修改仍留在原工作树；数据、模型和视频不提交。进程、端口、GPU状态本次未重新核验，旧PID不是当前在线证明。旧A100路径已失效，allocation607356已释放；未访问HPC、未启动服务或机器人。
