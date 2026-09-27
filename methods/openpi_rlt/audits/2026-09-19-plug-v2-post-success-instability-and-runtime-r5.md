# plug_v2 成功后轨迹不稳定诊断与 Runtime r5

日期：2026-09-19

## 结论

固定候选没有在成功后发生参数更新。`episode_47` 自主成功后，48–51均因轨迹异常由操作员放弃；actor始终为 `rtc-corrective-r1@2000`、SHA-256 `a197ab49…`，没有online cycle，Machine A使用固定seed42。当前该候选有效终局累计2成功/11条（18.2%），但后续异常说明不能把它称为稳定改进模型。

成功47与后续四轮起始14D state最大差仅0.001343rad，排除明显home/关节初态漂移。后续首个reference plan相对成功轮次最大差分别约0.0494/0.0605/0.0453/0.0461rad，差异在corrective actor介入前已经出现，说明模型对观测场景/物体复位细节敏感。

Runtime r4累计发生6次RTC deadline恢复；自动恢复会重新进入未由corrective actor覆盖的d0 reference规划，可能进一步改变轨迹。但episode51没有新的恢复仍异常，因此它不是唯一原因。

历史放弃轮次按既定策略删除HDF/video，只保留compact state/action，无法重建48–51的首帧。读取当前三相机并与成功47首帧做同一成功state确定性推理，首10步右臂最大差仅0.001330rad、RMSE 0.000492rad，表明当前画面接近成功画面；也证明后续必须按episode保留小型首帧证据。

## Runtime r5

源码：`/media/agilex/Getea1/jiaan/projects/rlt/methods/openpi_rlt/plug_v2/runtime.py`。

- 首次 `rtc_deadline_missed` 即丢弃迟到计划并进入 `terminal_pending`，不再自动d0重锚和继续动作。
- 每轮第一次真实观测保存14D state与三相机首帧压缩NPZ；no-record评测也保存，用于追溯场景差异，不保存整段视频。
- 30Hz、delay6、0.04rad tracking bound、Piper关节限位、固定actor均不变。
- `py_compile`、8项focused regression与首帧保存smoke通过；没有启动新Session或机器人动作。证据：`runs/plug_v2/learning/candidates/rtc-corrective-r1/runtime-validation-r5.json`。

## 下一步

当前旧r4 Session在waiting_scene且policy paused。先结束Session；下一次普通 `./scripts/rlt_demo.sh --no-record` 会替换runtime并加载r5，不需要restart。只做1条canary：若轨迹从一开始不接近目标立即暂停/放弃；若发生deadline会自动安全暂停。核验首帧、第一计划与现场画面后，才恢复reference/warmup/corrective三组对比。
