# 2026-09-21 在线学习现状诊断与真机RL框架复盘

## 范围

本轮仅做只读诊断、概念复盘和文档整理。没有启动或停止RLT训练、没有修改生产代码或权重、没有执行机器人动作，也没有访问HPC。

## 当前动态快照

2026-09-21 现场只读核验：

- Cobot RLT根：`/media/agilex/Getea1/jiaan/projects/rlt`。
- 上游仓库：`https://github.com/Yyshadow/openpi-RLT.git`，固定commit `c1e40ac360185778c98cf20da2820e22d2d415e7`，`main...origin/main`且工作树干净。
- backend为`ready_disarmed`，Session已停止；上一批真机episode固定使用actor13765。
- 最新接受release为`rtc-online-20260921T175500-164578`，global step/actor指针14135；下一条新Session才会加载14135。
- 最新批次含5个新UUID、74条transition和370次更新，发布门接受；另有1条pending，尚不足下一批5条。
- 当前在线目录共240条RLT sidecar：104 success、87 failure、49 aborted。success包含HIL成功，不等同纯自主成功。

上述PID、端口和进程状态是动态快照，后续使用前重新核验。

## 主要诊断结论

在线链路确实在采集、转换、训练和发布，问题不是“没有更新”。actor从8445推进到14135，但真机数据没有显示单调改善。在未严格固定场景的历史统计中：

| actor范围 | 纯自主成功 | 自主失败 | 纯自主成功率（排除HIL与放弃） |
|---|---:|---:|---:|
| 5000–8445 | 32 | 29 | 52.5% |
| 8445–11000 | 3 | 30 | 9.1% |
| 11000–13000 | 7 | 16 | 30.4% |
| 13125–13765 | 0 | 12 | 0% |

该表受场景漂移影响，不能解释为actor的无偏因果效果，但足以否定“版本号上升即性能持续提高”。

固定15条旧留出的early-policy Q separation从step5420的0.1292降至step14135的0.0581，约缩小55%；AUC仍约0.86–0.98，表示排序大致保留，但成功与失败的价值幅度正在收缩。当前发布门只限制每个候选不比父版本下降超过20%，可能允许逐批小幅下降累积成明显退化；门也不检查当前场景的纯自主成功率、终点空间偏差或反复偏向同一侧。

最新训练使用stratified replay，实际batch约50% human、75% success。最新五条是3条HIL成功与2条自主失败，比例本身符合既定计划，但成功监督主要位于HIL后段，未必能直接纠正更早发生的系统性偏孔。

当前actor目标保持上游形式`BC - Q + delta`，配置为online BC 5、Q 0.1、delta 300；固定上游AgileX配置的delta为10。高delta有助于稳定和平滑，也可能抑制相对Stage1参考轨迹的空间纠偏。现有metrics摘要恰好落在critic-only step，`did_actor_update=0`，没有保存actor更新时的weighted BC/Q/delta，尚不能直接量化三种梯度的相对主导程度。

场景初始分布也发生明显变化。最近22轮与actor8185自主成功批次的最近邻像素MAE约为camera_high 9.5、camera_left 4.0、camera_right 8.5；各批内部最近邻约为1.7、2.2、2.4。初始关节状态最大差约0.0117 rad。相机、插头夹持、排插位置或光照变化会混淆不同actor间的成功率对比。

最近13765有效轮次没有deadline miss；最后一次aborted来自handover fault。持续偏孔的主要证据指向学习目标、评估门和输入分布，而不是RTC deadline或RPC连接。

## 框架认识

DAgger和真机RL可以共享相同的rollout、HIL、录制和部署框架。DAgger在策略访问到的状态上拟合专家动作；RL通过critic最大化累计奖励。当前RLT同时包含HIL BC、Q最大化和Stage1参考约束，因此critic或Q梯度失效时会退化成“带失败数据和参考约束的DAgger”。

episode终局失败标签是任务结果，不是每个action chunk的局部质量标签。当前实现使用中间reward 0、成功终局reward 1和SMDP TD传播，没有把failure复制成每一步的分类标签。仅有稀疏终局信号仍不能直接识别哪一个chunk导致失败，需要成功/失败轨迹重叠、及时HIL和可靠Q传播。

## 下一步建议

在继续自动发布前，固定相机、home plug、中臂位姿、插头夹持和排插位置，对causal step5000、历史actor8185及当前14135各做同场景冻结纯自主评测。每版建议10轮，不写训练数据、不HIL，记录成功、左右/上下终点偏差、保护暂停和轨迹平滑度。

同时离线完成三项单变量诊断：

1. 在最近场景留出上画Q随episode位置变化，检查价值是否只在HIL或终局附近抬升。
2. 记录actor-update step的weighted BC/Q/delta和human mask，确认actor实际由哪一项主导。
3. 对delta 300/100/30/10做固定replay消融，并将发布门改为相对固定基线检查，避免父版本棘轮退化。

这些是诊断建议，尚未成为生产配置变更或真机放行结论。