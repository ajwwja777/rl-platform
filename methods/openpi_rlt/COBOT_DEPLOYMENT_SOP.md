# Cobot openpi-RLT 现场部署 SOP

## 当前状态

R1 joint `step_4999`、Machine A/B、真实三相机/14D observation、Task5 自动录制、网页人工终局和连续 episode 已完成 zero-publisher shadow。验收期间 `/task2/policy/joint_left` 与 `/task2/policy/joint_right` 始终无 publisher，shadow replay 写入为 0；尚未执行 RLT 真机动作。

统一入口：

```bash
/home/agilex/cobot_magic/task3/jiaan/realworld_rl/openpi_rlt/interface_task2_teach_rlt_live.sh
```

日常操作页：`http://127.0.0.1:8016/`。页面只绑定 loopback，需要在 Cobot 桌面浏览器打开。

## 一次现场实验的最短流程

### 1. 按既有流程准备底层

由现场操作员完成并核验：

1. 六路 CAN、急停、夹爪和工作区；
2. Task2 五臂/button-handover ROS launch；
3. 三相机 launch 与中相机姿态；
4. 没有其他 policy server/client 占用本模型端口或 `/task2/policy/joint_*`。

这些步骤可能 enable 或移动机械臂，不由 RLT 脚本隐式执行。`task2_home_cli.py front` 不是启动前置条件；但按用户选定的现场流程，LIVE episode 正常终局并完成数据固化后会自动调用它。该 home 姿态与 `legacy40-v2.1` 采集初态存在分布差异，必须作为实验条件记录，不能与历史基准直接混比。

### 2. 启动 Task5

RLT 自动录制 Session 明确依赖 Task5；普通 VLA deployment 是否使用 Task5仍是独立选择。

```bash
conda activate aloha
cd /home/agilex/cobot_magic/task5/jiaan/hil_realworld_rl
bash v1/scripts/start_task5_v1.sh
bash v1/scripts/check_task5_v1.sh
```

RLT 会自动调用 Task5 开始/结束每条 episode，不再需要操作员在旧 Task5 页面逐条点击录制。在线数据写入：

```text
/home/agilex/cobot_magic/task3/jiaan/realworld_rl/data/task5-rlt-r1
```

入口要求该数据盘至少有 30 GiB 可用空间。历史实测 600 帧录制约 1.66 GB，长 session 必须持续关注磁盘。RLT 不再把 600 步当作 episode 终局；Task5 仍保留现有 3,600 帧/约 120 秒的 recorder 安全上限，达到该上限时策略会暂停并要求操作员选择终局，禁止脱离录制继续运行。

### 3. 启动 RLT

只有在当次现场动作授权和安全检查完成后才运行 LIVE：

```bash
conda activate aloha
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
cd /home/agilex/cobot_magic/task3/jiaan/realworld_rl/openpi_rlt
./interface_task2_teach_rlt_live.sh 4999
```

需要重复无动作诊断时使用：

```bash
./interface_task2_teach_rlt_live.sh 4999 --shadow
```

`--shadow` 会读取真实图像/关节反馈并输出每个推理 chunk 的编号、latency 和 actor version，但不会创建关节命令 publisher，也不会把 shadow transition 写进 replay。

模型首次冷加载实测约 9 分钟。Machine A 现在由项目生命周期脚本常驻：精确匹配 PID、checkpoint、端口与 metadata 时直接复用，不再重复恢复 7.8 GiB checkpoint；Machine B 和 ROS/UI 仍需约几十秒启动。出现“已就绪，当前暂停”后，按 Enter 只完成 operator arm，仍不会开始 rollout。

### 4. 在网页操作 episode

在 Cobot 浏览器打开：

```text
http://127.0.0.1:8016/
```

第一条点击“开始 Session”。系统顺序为：

```text
创建 session/episode
  → Task5 recording ready
  → fresh observation/replan
  → rollout
```

运行中只需按实际结果点击一个：

- `成功`：reward 1，允许进入 replay；
- `失败`：reward 0，允许进入 replay；
- `放弃本轮`：保留 Task5/trace，明确不进入 replay。

RLT 不设置固定 600-step 终局：不同任务可有不同长度，模型在完成位姿等待时由操作员判断并点击成功、失败或放弃。LIVE 下终局顺序固定为：暂停策略 → 固化 Task5 与 compact trace/replay → 通过项目包装器执行原 `/home/agilex/miniconda3/envs/aloha/bin/python .../task2_home_cli.py front` → 页面进入 `waiting_scene`。包装器只解决原 CLI 服务成功后 ROS 辅助线程不退出的问题，不改变 home 位姿、速度、参数或服务。回位时操作员不得进入工作区；回位完成后人工恢复物体和场景，再点击“开始下一轮”。Task5 会自动创建新 episode，策略从最新 observation fresh replan。正常流程不再需要另开终端调用 outcome 或 object-reset ROS service。

SHADOW 终局、`Ctrl-C` 收尾和 fault 路径明确不调用 home。若 home 命令失败，Session 停留在 `fault`，不会允许开始下一轮；不得为了继续采集而绕过故障门禁。

## HIL 与控制权

- 任一后臂示教按钮按下：policy 立即暂停，Task2 把对应前臂交给示教，Task5 与 RLT trace 继续记录。
- 单臂接管记录逐臂 `expert_mask`，双臂接管记录双臂 expert；HIL 本身不结束 episode，也不自动判断结果。
- 最后一个接管释放：接管前的请求、chunk 和 generation 全部失效；从最新 observation fresh replan，禁止续播旧轨迹。
- terminal、Task2 fault、相机/服务错误期间释放按钮不能误恢复策略。

## 数据、replay 与 online actor

- 正式 run 为 `/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/openpi-rlt/online-r1-legacy40-v2.1-session-v2`。
- 旧 run `online-r1-legacy40-v2.1` 含 60 条早期 shadow 虚构 transition，永久隔离，禁止恢复或训练。
- Task5 HDF5/sidecar 与去图像的原子 compact trace 是 Cobot 原始证据；不再额外保存每轮重复的 upstream raw image pickle。
- success/failure 才能进入 replay；aborted、pending、fault 和 shadow 均不进入。
- 正式 run 已用 605 条 replay 完成一次性 20,000 warmup update，当前 learner step 20000、actor version 10000、`ready_for_online=true`；后续每新增 1 条 replay 按配置追加 5 次 update。online actor 只允许在 episode 边界切换。

## reset 边界

当前 LIVE 版本只实现用户明确指定的 `in_the_pot` 前臂回位：人工 success/failure/aborted 后，必须先完成数据固化，再调用既有 `task2_home_cli.py front`；策略全程保持暂停，回位成功后才进入 `waiting_scene`。SHADOW、`Ctrl-C`、服务故障和数据固化失败不会触发回位。

这不等同于无人值守的 `--auto-reset`：脚本仍拒绝该参数，也不会自动恢复物体、自动开始下一 episode 或猜测成功。未来插接等局部任务必须单独定义并现场验证命名 `reset_profile`、目标姿态、速度、工作区、超时、退避路径和失败处理，不能复用本任务的 `front` home 假定。

## 停止与故障

- 活动 rollout 中按 `Ctrl-C` 会先按 `aborted` 固化 Task5 标签与 compact trace、暂停 policy，再停止本入口创建的 replay、learner、actor、env driver 和 UI；不会 kill 未知进程，也不会触发 home。
- Machine A 默认常驻以缩短下一次启动，`Ctrl-C` 不释放它。当天全部 RLT 工作结束后显式运行 `./stop_rlt_model_server.sh 4999`，脚本只会停止登记且精确匹配的服务。
- Task5 是否停止由操作员决定；停止前先确认没有 active recording。
- 端口 `8000/8016/9101/9102` 已占用、Task5 不可用、磁盘不足、状态不一致或 stale generation 时，入口/网页 fail closed，不静默跳过录制。入口每秒检查 learner、actor 和 replay；任一角色异常退出即暂停策略并清理本次 Machine B，不能继续留在无 learner 的假在线状态。
- 脚本退出后仍须现场确认 policy publisher、Task2 coordinator、CAN 和机械臂实际状态，不能把终端退出等同于物理停止。

## 本轮已验证与仍待验证

已验证：

- Cobot 正式环境相关测试、Bash 与网页脚本；
- 真实 `step_4999` 加载和 shadow chunk 遥测；
- success → waiting_scene → next episode → aborted 的连续网页流程；
- Task5 50/50、600/600 committed 与 UUID/sidecar；
- 无 600-step RLT 上限的 105-chunk shadow、zero publisher 与 shadow replay=0；
- 活动 episode 的 Ctrl-C 自动 aborted 收尾、Task5/trace 原子提交、无 traceback；Machine B 端口退出，Machine A 可复用。
- 自动 home 的调用顺序、精确无 shell CLI、SHADOW/Ctrl-C 禁止回位和失败转 `fault` 已通过自动测试；现场原 home 服务已返回成功，CLI 退出包装器尚待下一次终局验证。
- 605 条真实 replay 的 20,000-step warmup、最终 checkpoint/snapshot、step 20000 恢复与 actor version 10000 的 finite `(10,14)` 离线推理。

仍待现场动作授权验证：

- LIVE 首个小工作区动作；
- 后臂按钮真实接管、释放与 fresh replan；
- success/failure LIVE replay 内容、reward 和 `expert_mask` 对账；
- 最终 actor version 10000 的 LIVE episode 边界切换及后续 5× online update；
- 新 CLI 退出包装器下 LIVE 正常终局的 front home → `waiting_scene`；
- 任意无人值守自动 reset/自动下一轮；
- R1/R2 真机成功率和 B1 公平对照。
