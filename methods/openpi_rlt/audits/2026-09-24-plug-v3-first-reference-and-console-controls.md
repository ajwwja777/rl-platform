# plug_v3 首轮 Reference 诊断、replay 隔离与统一控制台操作闭环（2026-09-24）

## 结论

- 首次且唯一一次剧烈抖动不是 `step_4999` Stage 1 Reference 的表现。补丁前的 EnvDriver 子进程没有安装右臂/Reference runtime patch，171 个控制步全部误走未训练的 RL actor（`actor_param_version=0`、`source=RL`）。之后正常 Reference 均为 `actor=-1`、`source=BASE`。
- 其余 rollout 在插孔前近似停住不是 tracking bound 或 safety pause。失败/放弃轨迹的尾段 90%–100% 为近保持动作，requested-step RMS 约 `0.0011–0.00124 rad`；成功轨迹尾段为 `0.002163 rad`，HDF5 实测尾段移动也更大。策略本身在这些观测上收敛为保持。
- 运行时还发现 plug_v3 错用了历史锅盖任务 prompt：`Open the pot lid...`；训练数据 canonical prompt 是 `Insert the held plug into the socket.`。v3 启动入口现显式固定正确 prompt。

## 抖动证据与修复

错误轨迹：
`/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v3_yyshadow/online/traces/reference/episode_1790235377576533708_aborted.jsonl`

量化结果：

- action 相对 reference RMS：`0.015686 rad`
- requested step RMS：`0.040631 rad`
- 相邻 action delta RMS：`0.015331 rad`
- 方向反转比例：`0.564`

后续正常 Reference 的 action-reference RMS 仅约 `1e-5–4e-5 rad`。修复包括：

1. EnvDriver spawn factory 内显式执行 `install_bimanual_runtime_patch()`，不再依赖父进程 monkey patch 被 spawn 继承。
2. Cobot env 接受上游 keyword-only `control_hz` 调用契约。
3. Reference 继续以 actor version `-1` 强制使用 Stage 1 `ref_chunk`。

## Replay 隔离

补丁前 journal 含 46 条 transition：

- 错误 episode 0：17 条，全部 `source=RL`，来自未训练 actor；
- 正常成功 episode 3 与 7：29 条，全部 `source=BASE`，两个终局正奖励。

未删除旧 journal；它继续保存在 `online/replay/replay_journal.pkl` 作为证据。新配置改读：

`online/replay_clean_v1/replay_journal.pkl`

该 clean lineage 经官方 `inspect_replay_journal.py` 验证为 29 条 BASE transition、2 个成功 episode、无 RL/HUMAN/MIXED 污染。

## 统一控制台

8015 的真机 RL 页面现把三相机和 rollout 控制放在同一首屏：宽屏左右并排，窄屏自动单列。新增键盘操作：

- `→`：开始；rollout 中再次按为暂停；暂停时为继续；`waiting_scene` 时开始下一轮；
- `↑`：成功；
- `↓`：失败；
- `←`：结束并放弃。

输入框、下拉框、视频或可编辑区域获得焦点时快捷键自动禁用；按键重复、Shift/Ctrl/Alt/Meta 组合也不会触发动作。成功/失败/放弃按钮下方新增“终局后自动归位”勾选框。plug_v3 下次启动会把该选择传给 Session，并使用平台登记的 `all --pose plug2 --yes`；复位在数据固化/replay 提交后执行，完成后进入 `waiting_scene`，再次按 `→` 开始下一轮。

## 验证

- RLT runtime/ROS/factory：29 passed。
- 平台 RLT/API 定向：24 passed。
- Node 控制台快捷键与相关组件测试通过；live 8015 已实际返回新版 HTML、app.js 与 console_ui.js。
- clean replay 官方检查通过。
- Python、Node 与 Bash 语法检查通过。

未启动或重启机械臂、相机或 RLT Session；未发布机器人动作。操作员暂停本轮工作后，RLT EnvDriver 已退出；Stage 1 模型服务和 8015 网页服务仍为动态现场进程，后续使用前重新核验。

## 下一次现场验收

1. 刷新 8015 真机 RL 页面，确认三相机与控制同屏、快捷键说明可见。
2. 选择 Reference 并启动；确认新 trace 中 prompt 为 `Insert the held plug into the socket.`，actor/source 为 `-1/BASE`。
3. 固定场景先做 5–10 条 Reference；分别统计自主成功、近孔保持和偏差方向。
4. 只有 prompt 修复后的有效成功/失败数据进入新的 clean replay；结束放弃不进入训练。
5. 达到 600 个有效 transition 后再开始忠实 warmup。


## 17:35 Task5 ??????

Reference ??????? task5_start_failed??????? RLT ????? actor=-1?clean replay 29 transitions ???ROS readiness ? ok??????? warmup/episode_000008.hdf5 ???????? label sidecar?Task5 storage/prepare ?? label_blocked=true?????????????????????? aborted / bad / keep_for_training=false??????? label_blocked=false?

??????????Task5 ???????????????????????????????????? episode ??? aborted/non-training??????? fail-closed?Session ??????????????? HTTP detail???????? task5_start_failed?Cobot ???? 15 ????Reference ???????? ready_disarmed???? idle?ROS readiness ok?? arm???? rollout??????????
