# EXPO rollout 监控（2026-10-20）

用户批准录像/Q 联动、实际 Edit 选择/幅度、推理与等待、回合结果趋势，并明确禁止动当前 HPC 进程。范围为 CPU 源码/静态页面/本地只读观察器，不启动/停止/替换 Actor、Learner、Session，不触碰 Replay、权重、动作或现场硬件。

## 已实现

- 复用原 _jitted_fast_select 已计算的候选数组，保留原结果/RNG，无新增模型 forward。Q 为 target-critic ensemble reduction 对执行滤波前动作块的预测。
- 每次选择保存 best_unedited_q、edit_selected、edit_norm（选中 Edit 与其对应基础候选整个 normalized chunk 的 L2 范数，无量纲）。未选择 Edit 的0是实际选择语义；旧/异常数据缺字段不补0。
- inference_ms = 当前/select compute_seconds；precache_ms = 对应/precache compute_seconds；wait_ms = 同一预计算结果返回就绪到调用/select前的间隔，包含正常RTC调度等待，不表示机器人卡顿/网络延迟。私有timing不发送给模型。
- 新点同时记录单调时钟与系统时钟，供视频 frame index/fps 与 frame_timestamps 对齐；旧wall-only记录不伪造映射。
- 普通/异步和continuous覆盖方法均支持。Continuous保留每点Actor/precache版本和整轮版本列表，不假设其回合内权重固定。

## 验证与生效边界

103项CPU相关测试通过，覆盖选择数组无更改/RNG不参与、RPC字段保持、时间配对、缺失字段、归档、多版本continuous。较宽旧回归有1项基线fixture失败：test_expo_execution_migration.py 的 previous/device_layout.py缺失；在未改7ef725f复现，未扩大修复。未启动GPU或机器人测试。

生产端新字段需要未来正常加载Actor/runtime；本次只同步源码，不重载或热注入。当前HPC不安装文件、不发控制请求。Web独立export_rollout_review.py仅在Cobot读取本地登记目录、现有录像/JSON/SQLite，并输出静态review JSON；缺失数据保持缺失。Web实施记录见cobot-web/docs/MIGRATION.md。

集成worktree：scratch/rl-platform/expo-rollout-integration-20261020；基线7ef725f。现场SHA与静态验收回执计划保存在Cobot cobot-web/runtime/verification/rollout-monitoring-20261020/，实际交付结果后补。
