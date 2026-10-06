# Supported 7k 候选交付与停止条件（2026-10-06）

本流程已在执行对话先展示拟写内容。它是可选候选的操作边界，不替换原5k默认。主代码、Git、完整证据在A6000；现场由当班负责人操作。本任务没有启动/停止现场服务、加载模型、占用现场GPU或运动。

## 交付身份

- 冻结模型 ID：`plug-v3-supported-7k`；Online ID：`plug-v3-supported-online`。
- 不可变交付目录：`models/rlt/plug_v3_yyshadow/history/candidates/supported_online_20261006_v3/`。
- 原型 `supported_online_20261006` 和 `_v2` 存在恢复/专家phase打包缺陷，不使用。
- Learner7000 / Actor3500；初始完整5k状态继续2000 Critic /1000 Actor，预先指定seed42。
- MC=.3、Actor Q=.1、BC=5、delta=10、ActorLR=1e-5、CriticLR=1e-4、batch128、C10、gamma=.99、dropout=.5、Actor周期2、目标EMA tau=.005。
- 固定初始Actor作为teacher，权重50，仅约束旧专家/自主成功TRAIN的前6关节。夹爪固定，Critic各备选动作投影到相同测量夹爪，保留原生7D wire格式。
- 执行固定 `async_rtc50`：发布50Hz、逻辑20Hz、RTC、物理时间滤波；目标网络EMA不等同执行滤波。

## 第一步：无动作加载与一致性验证

现场旧模型必须由当前负责人释放。核对现场coordination、实际deployment/Session/recorder状态；不得在运行中覆盖Python源码或切换模型。源码先在A6000提交推送，按交付manifest逐文件校验同步，保护现场已有改动。Cobot无Git副本不被当作主仓库。

只同步新候选资产，不替换旧5k/生产Replay。候选包含原生Actor、完整Adam/target/RNG checkpoint、teacher、归一化、derived TRAIN journal及profile。该私有journal是2496窗口/201Episode；不是生产Replay备份。

先在独立目录建立候选runtime（不存在才允许创建）：

```bash
python scripts/fork_supported_runtime.py --source <不可变候选目录> --target <RLT_MODELS>/history/candidates/supported_online_20261006_runtime
```

此命令只复制/校验候选并重定位YAML，不启动服务、不加载Stage1、不发布Actor。冻结ID的Actor路径使用不可变候选；Online ID读取这个单独runtime。`profile.json`与teacher/norm/source SHA必须一致；现场asset root须与注册的RLT_MODELS一致，不改生产root。

选冻结ID并加载，保持Session未开始、机械臂暂停，核对Actor3500、Learner禁用、所选checkpoint/归一化SHA、右夹爪hold、50Hz/RTC/滤波实际设置与Stage1 RTC协议。加载失败、回退Reference、版本-1、身份不符即停止，不开始运动。旧绑定5k的诊断设置需在释放时禁用，或重新绑定新冻结ID：`configure_evaluation_diagnostics.py --enable --model-id plug-v3-supported-7k --trace-root <独立目录>`。设置只在下一次构建读取。

## 第二步：受控冻结验收

验收对象是这份已经改变的7k，不重复把旧5k评测当作新算法验证。先使用评测用途，禁止Replay写入/学习。固定Reference、执行profile、初始/目标布局和任务定义；记录自主成功、辅助成功、失败、放弃分别的完整Episode。预先登记布局顺序、样本预算和停止条件。若用于选型，标DEV；独立TEST保持封存，不参与选择。

首次可用预先随机排列的40Episode（原布局/小幅移动布局各20），新7k与旧5k在同布局做匹配交替；如资源不足按实际完成数报告，不补数据。人工介入后成功只能算辅助。轨迹/窗口不当独立样本，区间按完整Episode配对重采样。该预算是新评测设计，不是已有结果或成功率保证。

分开统计实际发布间隔、推理时间、deadline miss、队列耗尽、跟踪误差和任务结果。逻辑20Hz、C10对应0.5s；重规划/RTC预算按实际队列与时间戳判断，不拿高Hz或小命令步长证明插入更好。先无Learner测量，再开启独立Learner检查争用。碰撞、持续偏移、输入/动作身份错误、异常固定通道动作或明显自主退化立即停止并回退旧5k。

## 第三步：分批Online学习，候选更新不直接覆盖执行Actor

只有第二步达到预先登记的接受条件，才能选择Online ID。解除冻结和运动由现场负责人执行。输入保存完整state/RGB/prompt/RTC身份；新经验必须保持source/HIL边界、实际收到命令、terminal/reward/next-state身份。未知/冲突/不完整Episode不得进入本候选MC训练。不要删除HIL标签把辅助成功冒充自主成功。

- 新经验预算UTD=1，计量单位是Replay入库transition，并非独立Episode或物理动作。基准步7000、历史2496条不产生继承训练欠账。
- 原生分层采样 `.4 recent + .3 warmup + .2 HIL + .1 uniform`，池可能重叠；每batch写身份审计，实际互斥比例以审计为准，不宣称8:2。新数据比例随Replay改变。
- 完整checkpoint/teacher/profile恢复合同不能用原生默认入口替代。所有资源路径在独立candidates目录。
- Learner写`pending_actor/actor_snapshot.pkl`；服务继续读取`actor_snapshot/actor_snapshot.pkl`。训练更新不会自动改变正在执行的Actor。不要直接点历史5k seed分支来代替该入口。
- 每批先看输入身份、实际batch/预算、数值有限性、六关节保持/纠正拟合和同状态Q1/minQ诊断，再决定独立冻结验收/版本发布。发布必须在Episode之间、有明确版本和回退副本；本交付不含基于代理指标自动发布的许可或保证。

候选只通过软件、恢复和离线代理验证；新独立自主成功/学习曲线证据仍为空。已经重复使用的20DEV和旧6DEV不能改名TEST。历史缓存不自动因RTC修复而变正确。全部阶段结论、图、数据/代码/配置SHA、命令和边界见A6000 `outputs/supported-online-20261006/REPORT.md`及`delivery.json`。
