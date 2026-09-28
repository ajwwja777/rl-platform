# Cobot RLT 操作与恢复

本手册针对新目录的 plug_v3_yyshadow。现场入口：`http://10.7.165.64:8015`。迁移是否已切换、验证证据见 [MIGRATION](MIGRATION.md)；不要仅以目录存在判断可用。

## 日常启动

在 Cobot 终端执行。机械臂上电、工作空间和急停由现场操作员检查。机械臂启动沿用已有 auto-enable 行为。

```bash
cd /home/agilex/jiaan/project/cobot-control
./scripts/can_up.sh
./scripts/roscore_up.sh
./scripts/arms_up.sh
```

arms_up 在当前终端前台运行，另开终端运行相机：

```bash
cd /home/agilex/jiaan/project/cobot-control
./scripts/cameras_up.sh
```

另开终端启动网页。网页进程与 ROS launch 分开；重启网页不会自动重启硬件或解决学习进程故障。

```bash
cd /home/agilex/jiaan/project/cobot-web
./scripts/ui_up.sh
./scripts/ui_status.sh
python3 scripts/console.py recovery status
```

## 在线学习

1. 在采集页选择数据目录 `/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/online/three_camera_v3`，检查并使用；选择目录不需要先加载模型。
2. 选择在线模型 `plug_v3-online-latest`，核对路径为 `/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/online/actor_snapshot/actor_snapshot.pkl`。固定 warmup 5k／Reference 用于对照，不开启在线 learner。
3. 点击加载，等待成功提示。加载保持策略暂停；初次编译耗时数分钟，以 ready 状态为准。
4. 点击开始 Session，然后开始一轮。HIL 沿用后臂示教按钮接管／释放后的新动作规划逻辑。
5. 开启成功／失败标注：↑ 成功、↓ 失败；空格暂停／继续；→ 开始，暂停时结束保存为未标注；← 放弃。复位按所选机械臂、位姿和复位勾选执行。
6. 一轮完成后等待 Replay 提交及归位结束。检查 learner_step、actor_version、replay_adds_total；完成 warmup 后每新增 transition 使用现有 5 次更新预算。没有新 Replay 时停在当前 step 是正常现象。
7. 结束工作时结束 Session，再释放模型；按需要停止相机／机械臂／ROS。

命令行等价入口（每条状态变更只执行一次，超时先查状态）：

```bash
cd /home/agilex/jiaan/project/cobot-web
python3 scripts/console.py model list
python3 scripts/console.py model load --id plug_v3-online-latest
python3 scripts/console.py model wait --seconds 600
python3 scripts/console.py model session-start
./scripts/rlt_v3_status.sh
python3 scripts/console.py model session-stop
python3 scripts/console.py model unload
```

开始／暂停／继续／成功／失败／放弃和目录选择的完整参数：

```bash
python3 scripts/console.py --help
python3 scripts/console.py capture --help
python3 scripts/console.py model --help
less docs/COMMAND_LINE.md
```

## 固定模型评测

评测选择 `plug-v3-reference`、`plug-v3-warmup-5k` 或 `plug_v3-frozen-latest`。保存目录使用 `/media/agilex/Getea1/jiaan/data/evaluations`。最新在线模型在评测中被禁止，避免边测试边改变权重或把评测样本加入训练。

固定 warmup 位于 `/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/warmup_5000`，独立于持续更新的 `/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/online`。不要将最新 Actor 的后续成绩归到固定 warmup 5000。

## HTTP 报错／网页无响应

HTTP 503 不等于“失败结果已经保存”。可能是 Session 尚未就绪、后端已退出、录制目录不可访问或终止请求仍在处理。刷新只修复页面状态，不能修复底层进程、磁盘 I/O 或 ROS 故障；不要反复按成功／失败。

```bash
cd /home/agilex/jiaan/project/cobot-web
python3 scripts/console.py recovery pause
python3 scripts/console.py recovery status
python3 scripts/console.py recovery snapshot
./scripts/rlt_v3_status.sh
```

保存诊断后，通过输出栏选择出错任务。网页失效时恢复工具直接访问进程与 Session，不依赖 8015。先看身份和中断范围：

```bash
python3 scripts/console.py recovery interrupt model
python3 scripts/console.py recovery interrupt rlt
```

上述默认仅预览；执行参数见 `recovery interrupt --help`。工具检查 PID、启动时间、进程组和子进程，优先 SIGINT；不要用大范围 pkill 或只杀 htop 的某一个线程。无法确认进程归属时工具拒绝处理，应回到其原始终端 Ctrl-C。

只有网页本身故障且底层策略已经暂停时，才重启网页：

```bash
./scripts/ui_down.sh
./scripts/ui_up.sh
python3 scripts/console.py recovery status
```

Input/output error 不是普通 HTTP 故障：先暂停并确保硬件安全，停止写入，检查 `df -h /`、内核磁盘错误和挂载状态；重启 ui 不能修复磁盘。完成本批切换后，新在线模型、数据和日志位于 Cobot 内置文件系统。

## 路径和服务

| 内容 | 位置 |
|---|---|
| Stage 1 | `/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/reference_4999` |
| 固定 warmup | `/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/warmup_5000` |
| 在线权重／优化器 | `/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/online` |
| Replay 和学习指标 | `rl-platform/outputs/rlt/plug_v3_yyshadow/online` |
| Stage 1 进程登记 | `rl-platform/outputs/rlt/plug_v3_yyshadow/model-server/process.json` |
| 网页任务／输出 | `cobot-web/runtime/console-jobs`、`runtime/deployment` |
| 硬件 launch 日志 | `cobot-control/runtime` |
| 网页／Session／Stage 1 | 8015／8026／8030 |
| Actor／Replay | 9131／9132 |

项目路径均以 `/home/agilex/jiaan/project/` 开头。A6000 同名项目位于 `/data/LFT-W02_data/jiaan/jiaan/projects/`。

## 模型和配置核验

```bash
cd /home/agilex/jiaan/project/rl-platform
./envs/online/bin/python scripts/preflight.py
./scripts/rlt_status.sh
```

preflight 只读权重和配置，不发布机器人指令。通过不代表真机成功率已验收。迁移后现场仍需从暂停状态进行一次短轮次，确认相机、中臂位置、控制权和结果保存后再连续采集。

## 2026-09-28 运行验收与现场下一步

固定 Warmup 5k / 最新在线模型均实测加载到 ready，Session disarmed、策略暂停。Stage 1 本次启动约44秒，首次编译后固定输入推理约76ms，仅作加载参考，不是闭环真机时延或成功率。

本会话的断电被动验收任务已停止；随后网页启动了新的臂／相机任务，最终只读复查为5臂反馈和3相机可用。实际操作前先查当前状态，缺少的节点才按“日常启动”启动，避免重复launch；检查相机和关节反馈，再选择实际所需臂／位姿归位。先做一次短轮次，确认暂停、HIL、成功／失败保存和归位，再连续采集。目录以操作者选择为准，本批不把历史 warmup 重命名为 online。

迁移基线 learner5000/actor2500/replay2567。历史归档与旧RLT清理已完成；模型已释放。已校验数据约158GB，Cobot系统盘最终复查剩余约77.6GiB；采集前检查工控机空间。本轮不删除其他项目共享资产腾空间。
