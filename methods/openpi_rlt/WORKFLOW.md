> 2026-09-18 plug_v2请优先用 [新流程](PLUG_V2_ROLLOUT_SOP.md)；下文旧5416/7592/旧相机仅历史。

# 插孔 RLT 简单流程（2026-09-17）

执行根/media/agilex/Getea1/jiaan/projects/rlt；数据根/data/rlt/plug；统一现场入口/media/agilex/Getea1/jiaan/projects/cobot-platform/scripts。旧入口兼容，不从旧trainer/HPC路径恢复。

顺序：相机与任务契约→专家示教→完整归档和数据QA→Stage1基础策略/RL token→特征缓存和warmup→冻结评估→在线rollout/HIL/replay更新→隔离复测。

当前left-camera-v2只完成目录与采集profile；现场视角标定/新数据/训练尚未完成，不能部署旧warmup冒充新视觉成果。首次采集前确认左视角同时看清插头、孔、接触与插入。新旧cohort及z_rl/ref缓存不得默认混用。

平台入口：

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/status.sh
./scripts/ui_up.sh --profile left-camera-v2
# 结束录制后
./scripts/ui_down.sh
```

新视角RLT模式受阻止，当前仅普通节点专家采集。旧模型继续实验时选legacy-camera-v1，再使用scripts/rlt_up.sh 4000；不因此证明其效果。方法参数暂未改变，20轮试验未由agent执行。

目标持久格式为LeRobot+完整metadata。旧456个完整HDF5已按用户明确放弃要求删除，历史依赖它们的图像/回放不可用；新批次自动完整转换后删除仍未实现。禁止把Stage1精简训练视图或compact replay当成完整图像/节点归档。

本轮又清理旧test与5incomplete共16.55GB；旧sidecar保留。平台目录与命令详见关联平台 WORKSPACE_LAYOUT.md。

## Todo

- [x] 盘点456个完整插孔HDF5、标签及相机通路。
- [x] 清理仅诊断/测试342个HDF5并保存证据。
- [x] 平台/方法/数据物理隔离、兼容入口、轻量profile及软件回归。
- [ ] 左相机现场视角标定、可辨识性审核和camera manifest。
- [ ] 少量新专家轨迹试采及完整归档QA。
- [ ] 生产LeRobot完整转换、节点/历史回放读取适配，验证后原件清理。
- [ ] 新cohort Stage1、特征提取与warmup训练。
- [ ] 冻结评估、有限在线RL及对照验证。

完整目录和证据见[关联平台布局](../../../proj-20260829-cobot-realworld-vla/platform/WORKSPACE_LAYOUT.md)。
