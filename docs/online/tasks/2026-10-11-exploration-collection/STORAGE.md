# 2026-10-11 网页保存目录与探索录制不一致

状态：已定位，A6000修复/15项CPU验证完成；现场模型仍加载，未同步到活动运行实例，未重启服务或搬数据。

## 已验证事实

用户网页选择 `/media/agilex/Getea1/jiaan/data/datasets/test`；现场 `runtime/storage-selection.json` 的 selected 与 current.online 均是此目录。该目录47条历史录制，最新修改截至10月8日，没有本次3轮。

本次3轮真实落盘：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/exploration-hil8-7281-v1/`。

| 文件 | UUID | HDF5帧数 | 标签 | 完成 |
|---|---|---:|---|---|
| episode_000001.hdf5 | 164b2bd2-4cee-47ca-89e8-a8b3c5a4e99a | 82 | failure，无HIL | complete |
| episode_000002.hdf5 | bcaef048-d842-47ce-8b55-ab11fc95f397 | 98 | failure，无HIL | complete |
| episode_000003.hdf5 | 01187a85-44df-4b52-9ec8-b8380d3d34a3 | 126 | failure，无HIL | complete |

HDF5以只读模式打开检查属性及所有dataset形状，未复制图像或整份数据。三相机/动作帧数对应一致；这是结构/完成检查，不是图像质量或动作语义审核。实际网页 GET `/api/rlt-recorder/api/episodes?data_root=<上述实际路径>` 返回3条 complete 且 labels_complete=true。直接选择该目录可查看已有数据，不需要搬文件。

数值trace分别155/187/241逻辑步，探索元数据覆盖全部行，Actor始终3640，无HIL。rollout日志显示三轮分别写入30/37/48，独立Replay统计size=115；本次没有写Replay或加载pickle推断能力。另有三条dropped轮次无Replay提交，不把它们算作这三个完整HDF5。

标签存在 `episode_quality=bad, keep_for_training=false`，与Replay独立记账不可混同。此处只报告原标签，不擅自改成成功、质量通过或训练可用；后续使用需按各入口的数据准入规则复核。

## 失败原因与修复

确定的代码错误：交付121f0f8在 `shared_model_env.create_shared_env` 对探索入口固定 `root_for_phase=lambda: locations()[recording]`，覆盖了web的selected_root。此前CPU验证没有覆盖「真实用户选择→Episode录制目录」组合，导致界面与落盘不一致。这是本次执行接入的疏漏。

修复：`exploration_collection.recording_root(phase)` 读取网页显式selected或phase选择；没有显式选择时才使用专用目录作为默认。Episode开始锁定路径，下个Episode重读选择；非法路径显式失败。私有Replay、trace、Learner禁用与权重身份约束不变。

验证：`tests/test_exploration_storage.py tests/test_exploration_collection.py`，15 passed。包含显式选择优先、phase兼容、无选择fallback、拒绝非法路径、当前Episode不变而下一Episode采用新目录。A6000隔离分支 `fix/exploration-storage-selection-20261011`，基线57b3bde。

现场状态：核对时ready、session_active=true、recorder stopped/complete；没有停止、释放、重新加载或运动，没有改写原始录制/Replay/网页选择。修复不能声称当前进程已生效；等待用户结束Session并释放模型后，限范围同步两个模块，再由用户重新加载验证下一轮实际路径。

证据：A6000 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/exploration-collection-20261011/storage-audit/`：field-storage.json、episode-completion.json、hdf5-metadata.json、history-api.json、storage-tests.txt。
