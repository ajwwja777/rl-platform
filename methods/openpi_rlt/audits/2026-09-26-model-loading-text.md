# 2026-09-26 模型加载改为文字提示

源项目：Cobot `/media/agilex/Getea1/jiaan/projects/cobot-platform`，8015。

采集与部署统一静态加载文字：插孔模型“正在加载 · 预计约 8–10 分钟”，π0.5 “正在加载 · 预计约 1–4 分钟”；其他模型显示预计数分钟。移除部署 indeterminate progress 元素及加载按钮 spinner，采集加载按钮也禁用 spinner。保留真实状态查询、错误/断连/过期状态处理及真实就绪后的一次成功提示。新模型开始加载时清除上一模型的旧成功提示，避免误认为当前已经就绪。无根据时间流逝推断成功或假进度。

耗时依据（保守范围，非统计平均）：

- 插孔 frozen 成功启动：`runtime/deployment/model-20260926T142844.log` 的启动时间 14:28:44；supervisor 14:37:11 启动，`runtime/deployment/logs/actor_service.log` 14:37:16 加载 actor2500，就绪链路约 8 分 32 秒。
- π0.5 DAgger：`runtime/deployment/model-20260926T175428.log` 17:54:28 启动；其 `pi05/logs/rtc_policy_server_20260926_175428.log` 权重恢复 5.78 秒，ROS 客户端 17:54:41 已启动，模型日志有 prewarming、ready and PAUSED，17:57:44 已开始保留的成功评估。没有完整 ready 时刻，因此用 1–4 分钟作保守提示，不声称测得准确加载时长。实际可能更快，等待相机/编译时也可能更慢。

修改仅在 `app/backend/segmented_frontend/`：`collection_model_ui.js`（共享估计及待处理动作）、`unified_collection.js`、`deployment_ui.js`、`deployment.css`、`locale_catalog.js`、`index.html`。修复 π0.5 异步 phase=loading 且 operation=null 时未显示加载状态的前端分支；Session 操作/释放显示处理中，不冒充模型加载。

验证：JS 语法，本地和已发布页面的隔离状态注入覆盖两个模型系列、加载/就绪/过期就绪/错误/重复轮询、采集与部署、中英切换。加载前不报成功，真实 ready 单次提示，过期状态不启用开始按钮，错误不被预计时间覆盖。1600/820px 无横向溢出，JS 异常 0。所有控制请求被拦截，未实际加载或操作机器人。

19:49 静态发布完成，无服务重启。完整修改前备份 `runtime/backups/loading-text-20260926T194833/`；通知收尾前增量备份 `runtime/backups/loading-text-20260926T194916/`。回退整项使用前一个备份。证据：`runtime/diagnostics/loading-text-20260926/`。
