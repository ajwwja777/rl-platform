# 2026-09-26 普通 / RL 统一采集控制

来源：Cobot `/media/agilex/Getea1/jiaan/projects/cobot-platform` 和相邻 `rlt/methods/openpi_rlt`。发布于 18:26，只重启空闲的 8015 网页；未运行模型加载、推理、归位或训练。

## 操作

- 普通采集：开始采集（勾选启用模型时为开始推理）、暂停并打节点、继续并打节点、只打节点、结束保存、结束放弃。方向右键依次开始 / 暂停 / 继续，左键放弃。归位按既有开关、机械臂、位姿选择执行。
- RL：相同六个 Episode 操作，另有成功、失败、开始 Session、结束 Session。移除可见现场 Arm、场景已复位、重复停止 Session 按钮。开始推理自动准备 Session，或从 waiting_scene 开始下一条；Session 停止后可重开，保持加载的模型。
- 用户明确确认：RL 结束保存不标成功失败。HDF5 保留，标签为 unknown / uncertain / operator_save / keep_for_training=false；存储检查认定为明确保留待审，不阻止下一条。会话 outcome=saved、replay_eligible=false；驱动收到 aborted 以排除在线 replay 并清除学习 trace。只有明确成功/失败进入原有 RL 链路。训练参数未改。
- 暂停 / 继续 / 只打节点写入 labels 的 operator_nodes（frame_index、node_kind），读图沿用首尾预加载及中间节点插入；相同帧的暂停/继续保留为独立节点。
- 普通采集模型选择接入真实 π0.5 / DAgger 加载，含原始 step2000 与 step2000 初始化后再训3000的 DAgger。使用已有 ManagedRuntime，不创建 deployment eval 记录。开始 Episode 后启动推理；暂停/保存/放弃先暂停推理，恢复时先恢复采集再推理。示教录制沿用 SegmentedCaptureService 的既有 HIL 门控。无模型仍为原纯示教。
- 普通模型的开始 Session 只准备并保持暂停；Episode 可自动准备。结束 Session 保留权重，释放模型才退出进程；活动采集中禁止加载/释放/结束 Session，先明确保存或放弃。
- 放弃继续沿用上一版无文件、无历史、无删除账本行为。

## 接口与实现

新增 `/api/collection/model` GET/POST（load/unload/session_start/session_stop）；普通 StartRequest 新增 use_model、collection_model_id。可选 CollectionModel 钩子在录制开始/暂停/恢复/终结与推理门控之间排序，复用原 UUID/generation 与 writer lease。

RL 新增 `/api/session/prepare`、`/api/episode/marker`、`/api/episode/save`，8015 对应 `/api/rlt/...` 转发。所有会话写操作保留 episode_id/generation 校验；加载成功不代表自动运动，只有开始推理实际执行。

## 验证 / 回退

- 平台 65 项测试通过：labels、collection_model（含 HTTP 联动）、segmented_capture_http、deployment_evaluation、discard_without_records、console_api。
- RLT 32 项测试通过：session、session_http、task5_client、session_e2e、session_shutdown、discard_trace。覆盖自动准备、未标注排除 replay、标记、重复/过期请求、停止重开及迟到的完成回调。
- 浏览器隔离写请求验证：DAgger 身份传递、活动轮次释放禁用、RL 直接开始、未标注保存及下一条。
- 线上只读验收：1600 / 920 / 520 px 两列按钮等宽对齐，加载实际新脚本，无 JS 异常、无机器人写请求。
- 21 个文件原子发布前逐项 SHA256 校验，无并发覆盖。备份：`runtime/backups/collection-controls-20260926T182608/`，诊断：`runtime/diagnostics/collection-controls-20260926/`。
- 当前现场的旧 Reference 部署注册仍显示 error（17:59 旧加载进程已退出），RLT Session 离线。本次没有处理/重载该模型，不把界面与模拟验证当成真机效果验收。用户需刷新后选择并加载采集模型再试用。
- A6000 同步本次 RLT 修改时保留其 session.py 允许从 PAUSED 进入 terminal_pending 的既有差异；旧 test_session_http.py 含与当前模块签名不符的历史用例，不覆盖，新增独立 test_collection_controls.py 保存本次会话测试。Cobot 为本次经过测试的运行版本。
