# 2026-09-10 Rollout 审核与训练参数提案

## 已核验数据快照

Cobot 路径：/media/agilex/Getea1/jiaan/data/cobot-realworld-vla/task5/plug-insertion-rlt-v1/raw。
37 个完整 HDF5（index 3–39）、25,942 帧；index 1–2 为历史 incomplete，保留隔离。完整数据中成功 22（自主 2 / HIL 辅助 20）、失败 9、放弃 6。31 个成功/失败轮次贡献636条 replay；不可将辅助成功率当成自主成功率。此为操作员结果标签，不是本次独立真机验收。

所有37轮三相机与qpos valid mask全有效、qpos有限、时间戳单调。原始 action 有无效值，符合暂停/无最新命令的valid_mask语义，不能直接用作无掩码SFT数据。636条 replay 的数值数组均无NaN/Inf；每个有效轮次有1条done，成功轮次累计reward=1、失败reward=0。放弃未进入replay。当前global_step=0、training_enabled=false、collection_only=true。

人工辅助的采样间隔中位数约0.02秒，纯策略段明显更慢；部分间隔p99约0.53–0.59秒，原始30Hz录制亦存在约0.43秒间隔尖峰。需要把执行时序问题与学习器动作抖动分开，不以训练loss证明机械平滑。旧代码HIL分支没有显式控制周期sleep；本次未擅自修改已有数据或时间轴。训练审核应按episode划分验证集，不随机拆相邻chunk以免泄漏。

六轮终局三相机图像抽查可读取、场景可见，但手爪遮挡使单帧不足以独立判定完全插入，保留人工终局标签。两轮真实回放(episode6、38)生成通过，HTTP206 video/mp4通过；对应节点数为0和1，UUID正确。

机器证据：
- /media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/scratch/rollout-review-20260910/hdf5-audit.json
- /media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/scratch/rollout-review-20260910/replay-trace-audit.json

## 文献及实现依据

原始论文 https://arxiv.org/html/2604.23073v1 （PI；RL Token）。
论文使用冻结VLA、轻量actor/critic、reference pass-through与BC锚定；附录Ethernet等为2层256宽MLP，chunk10、reference dropout50%，训练窗口每2控制步取样。控制50Hz，与当前20Hz配置不同。论文未提供可直接照搬的“128条/200步”或“每k轮32步”的通用处方。

我们固定上游为 https://github.com/Yyshadow/openpi-RLT ，commit c1e40ac360185778c98cf20da2820e22d2d415e7，是第三方开源复现，不能称为PI官方实现。公开runtime README的Ethernet默认600条、warmup20000更新、每新增transition5更新、20Hz、chunk10；样本stride0，不同于原论文dense stride2。
参考 https://github.com/Yyshadow/openpi-RLT/blob/main/rlt_online_rl/README.md 。

## 提案（尚未启用、未训练）

- 现有数据够做初次受控离线warmup检查，尚不足以证明自主稳定。按episode分层保留验证轮次，失败样本保留给critic；成功/HIL样本用于适用的BC目标。不要沿用HDF5 keep_for_training=false将所有失败从RL丢弃。
- warmup先100步评估，再决定是否达到200步；不自动延长到1000/20000。batch128，actor/critic lr均1e-4，MLP 2x256、tau0.005保持。
- 候选warmup BC=10、Q=0（先锚定/模仿，critic仍训练），online BC=10、Q=0.1；delta_weight=10保持，需离线比较后接受。此为本任务保守提案，不是论文定值。
- 验证看有限值、actor与reference偏移、右臂一阶/二阶变化、chunk边界变化和HIL释放后首动作；先检查未经执行限幅的actor输出，再检查限幅后输出，避免过滤器掩盖坏actor。损失合格不等于真机合格。
- 部署先确定性actor验证；探索保留chunk级linear_endpoints std0.001，不能改成逐控制步独立噪声。右臂启用、左臂/夹爪固定，限步0.02rad、增量变化限0.005，HIL释放暂停、恢复fresh replan、episode内固定actor版本。
- 在线建议k=1个有效成功/失败episode后做一批32次optimizer更新，完成后发布新actor，下轮才切换；放弃不计数。应有明确批次/版本日志、重启幂等和独立checkpoint。该episode批次调度尚未实现；目前上游仍按新增transition预算更新，但collection_only门禁完全关闭训练。待参数确认后实现并验证，再启用。

## 本次交互修复

RLT Session stop：活动轮次先aborted并完成trace/replay收尾；已提交success/failure保持原结果；完成后写operator-shutdown marker，supervisor正常cleanup并退出，Machine A保留。活动时按钮显示“结束并放弃本轮”。不触发停止后的归位。
专家网页：修复旧labels请求覆盖新选episode的异步竞态、preview请求同类竞态、常规目录刷新保留选中UUID；列表增加连续显示序号，物理episode_index/UUID不重排。视频回放已有且本次两例API验证通过。
转换：已有LeRobot0.4.2转换器保留mask、时间戳、labels、UUID与manifest，但没有“录完即转并删源”的完整自动管线。用户所说lebero格式确认仍待回复。HDF5仍是当前历史查询/回放及特征重建依赖，不能直接删除后宣称网页仍可用；需先接通LeRobot查询/回放并做转换核验。未删除任何真实episode。

## 验证与限制

Session HTTP/shutdown14项通过；smoothness、actor pinning、action conditioning、Task2、collection gate、phase及Session综合35项通过（CPU合成测试，无正式训练）。专家网页旧请求覆盖测试先失败后通过，既有8项workflow测试通过。当前浏览器控制工具无可用浏览器，因此未声称完成浏览器点击端到端验收。
代码/manifest与状态写回远端，无commit/push。GPU训练未启动，机器人动作未执行。
