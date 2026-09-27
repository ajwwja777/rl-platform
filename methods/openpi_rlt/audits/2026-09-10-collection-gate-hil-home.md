### 2026-09-10：采集审核闸门、示教释放暂停与归位异常

- 用户要求先审核数据质量和数量。插孔启动入口固定 `COBOT_RLT_COLLECTION_ONLY=1`；达到 128 条也不锁定 warmup、不执行 warmup 或在线更新。重启保持采集模式，训练须后续明确启用。已有 replay 98 条、global_step=0，未改动数据。
- 实际 Cobot LearnerService 无动作测试：模拟 1000 条 replay、200 次轮询，零更新、无 warmup latch；相关回归 45 项通过。
- 退出示教后改为暂停，由操作员选择成功/失败或点击继续；不自动恢复策略。最后一条辅助成功记录在释放后仍有约 25 个策略步，第一步最大受限关节增量约 0.002616 rad；尚不能据此确定现场前冲的完整原因，需现场复验。
- 终端退出已定位为 replay 提交后右后臂拒绝 CAN 归位，异常传播。现在归位失败保留 fault/暂停页面，不让该异常终止 env；不自动重试或清除电机故障。
- 核验启动参数及包装脚本已经传递 `all --pose plug`；CLI 先前臂、后两条后臂，原日志的 front 并不表示只执行 front。增加完整归位命令日志，无动作参数测试通过。
- Cobot 已部署并校验 overlay SHA-256；未启动 rollout、未执行任何归位动作，真机复验待操作员进行。无 commit/push，其他项目修改未触碰。

### 2026-09-10：修复采集模式达到阈值后仍等待训练的遗漏

现场 replay=151、global_step=0、training_enabled=false。Task5 episode_000012.hdf5 已 complete/committed，431/431 帧，当前 Session 第 3 轮 outcome=aborted 卡 replay_committing。定位 CobotRolloutPhaseController.begin_episode 在下一轮仍按128阈值等待 learner ready；关闭训练时永不满足。现显式传入 collection_only，绕过轮次间训练就绪等待，保留 warmup 采集标签而不执行训练。35项回归通过，Cobot已部署并验证连续5次轮次边界不等待。现有进程需用户结束Session、Ctrl-C后重启入口才生效。无机器人动作、无训练、未改动正式数据；未commit/push。
