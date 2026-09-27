# 2026-09-26 单页采集与公共快捷键

来源：Cobot `/media/agilex/Getea1/jiaan/projects/cobot-platform`。18:55 发布至 8015，仅在 writer 空闲、采集空闲、RLT 后端离线且部署无活动轮次/操作时重启网页。没有加载/释放模型、启动推理、移动机械臂或训练。

本记录按用户最新补充覆盖此前采集快捷键描述（包括右键暂停/继续及快捷键遵循自动复位开关的旧描述）。

## 当前操作契约

只有「数据采集」一页，取消普通/RL 页面切换。模型选项选择 π0.5/DAgger 或 RLT 模型服务；两个服务保留各自录制/推理实现。结果标注是独立复选框，不因服务为 RLT 就强制出现成功/失败。

- 不启用模型：普通示教采集，公共操作。
- 已加载模型、未勾成功/失败：公共操作，保留 HIL。
- 已加载模型、勾选成功/失败：增加成功并复位、失败并复位按钮，以及上下结果快捷键，保留 HIL。
- 活动轮次中不能更换模型类别、禁用模型或改变结果标注模式；模型未就绪不能开始推理。
- 普通/DAgger 与 RLT 数据、控制卡片等宽同排；目录在上，窄屏单列。

| 按键 | 行为 |
| --- | --- |
| → | 空闲时开始；暂停状态结束保存并复位。推理/录制运行中不结束。 |
| 空格 | 暂停 / 继续，沿用暂停/继续打节点。 |
| ← | 放弃本轮并复位；清除本轮数据、缓存、历史，不写删除账本。 |
| ↑ / ↓ | 仅已加载模型且勾选结果标注时，成功 / 失败并复位。 |

上述终局快捷键以及成功/失败按钮明确要求复位，使用当前选中的机械臂和位姿；普通保存/放弃按钮仍遵循自动复位复选框。归位前必须成功保存/清除本轮并停止推理，保留既有新鲜设备检查与 home 任务反馈。HIL 进行中不把「暂停推理」当成「已暂停采集」来让 → 结束。

RL「结束保存」仍是 unknown / uncertain / operator_save / keep_for_training=false，不进入 replay 或训练；π0.5 采集启用结果标注后，普通 stop 请求可附 success/failure/unknown，写入本轮 labels 和历史摘要。π0.5 标注数据不会跨服务自动送进 RLT replay。

HIL 门控未改变：π0.5 沿用 `ConsolePauseGate` 的人工暂停与 HIL 双门控（未手动暂停时退出示教恢复推理，已手动暂停则保持暂停）及 SegmentedCaptureService 的示教录制段；RLT 沿用既有 HIL/暂停规则。模型加载保持暂停，Episode 开始才发推理命令。

## 实现及验证

- 新增 `segmented_frontend/unified_collection.js` 统一布局、选项、快捷键判定；app.js 接入实际操作、所选归位及单页轮询。切换后台失败回退选择；RLT 操作结束立刻刷新按钮，避免等待轮询才解锁。
- 普通 stop 的可选 outcome 字段与 `SegmentedCaptureService.label_outcome`；写标签失败明确返回「数据已保存，结果标注失败」，同 UUID 终结重试幂等。
- 56 项后端测试通过：collection_model、segmented_capture_http、segmented_capture_service、discard_without_records、labels。
- 11 项纯 JS 快捷键判定通过；浏览器真实事件处理函数模拟验证模型身份、结果开关、活动轮次禁用、Space/四方向、HIL、忽略按住重复与表单输入、保存路径及选定复位。所有写请求由本地模拟拦截。
- 线上浏览器只读验收：1600 / 920 / 520 px，普通与 RLT 共用 operation 页面、卡片/按钮对齐，无横向溢出、无 JS 异常、无机器人写请求。现场冷加载/运动效果不属于本次验收。
- 10 文件发布前 SHA256 比对；备份 `runtime/backups/collection-controls-20260926T185505/`。证据 `runtime/diagnostics/unified-collection-20260926/`，含清单、浏览器结果、截图与用例。
- 前一批会话功能已在 18:26 发布并同步 A6000（自动 Session、未标注保存、节点等），见 `2026-09-26-unified-collection-controls.md`；本批没有再次修改 RLT 算法或训练参数。

下一步：用户刷新 8015，选择并加载需要的模型，在现场逐项验收暂停/继续、HIL、终局标签及所选归位。旧 Reference 加载失败状态未擅自重载。
