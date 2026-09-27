# plug_v3_yyshadow 忠实复现入口

本目录定义从零开始的 Yyshadow/openpi-RLT 忠实复现实验。它与历史 `plug_v2` 完全隔离；旧数据、旧 norm stats、旧 feature cache、旧 replay、旧 Stage 1 checkpoint、旧 actor/critic 和 release pointer 均不得作为本实验输入。

- 专家采集采用隐式成功语义：完整提交且未被拒绝/故障的专家 episode 直接记 success=true，无需现场再点一次 success；原始14D只作证据，转换时固定取索引7–13形成右臂7D训练样本。

## 复现目标

- 上游固定为 `https://github.com/Yyshadow/openpi-RLT.git` 的 `c1e40ac360185778c98cf20da2820e22d2d415e7`。
- Stage 1 使用上游 `rlt_pi05_agilexbag_image_delta_joint`：Pi0.5、H=50、RL token 1×2048、2-layer encoder/decoder、`rlt_alpha=1.0`、5000 steps。
- online RL 首轮使用上游 Ethernet 默认算法、loss、replay 和更新参数，不引入 MC-success critic、stratified sampler、五 episode 发布门、30 Hz RTC 或自定义 actor gate。
- Cobot 只保留不可避免的硬件适配：ROS1、三相机键名、Cobot 关节映射、后臂 HIL、急停和硬件限位。

## Cobot 动作合同

- 平台可保留完整双臂14D原始证据，但训练、proprio、actor和critic只使用右臂7D：`[right_j1..j6, right_gripper]`，对应物理14D索引7–13。
- 左臂关节、左夹爪和中臂状态不进入训练；左臂固定在`all/plug2`，中臂固定在`mid/plug2`。
- 视觉输入使用三路真实画面：中臂固定相机、左臂相机和右臂相机；仅右臂7D参与动作与proprio。
- Stage 1 的Pi0.5内部动作维度保持32，物理有效维由adapter显式标注。
- 7D→14D 写回只替换右臂槽位；禁止把历史 `plug_v2` 的被动动作/prefix修补直接带入。

## 数据与运行根

已在 Cobot 建立并与历史 cohort 隔离：

```text
/home/agilex/jiaan/data/rlt/plug_v3_yyshadow/
  demonstrations/
  warmup/
  online/
  evaluation/
  manifests/

/home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/
  stage1/
  machine_a/
  replay/
  warmup/
  online/
  evaluation/
```

完整代码、数据、权重和日志留在登记执行服务器；本目录只保存小型合同、配置和审计。

## 当前阶段

- Stage 0：上游身份、源码哈希、原始参数与隔离边界已锁定。
- Stage 1 数据已冻结：141条原始记录中134条进入训练，7条拒绝，1条在已完成插入后裁剪无效尾部；共19,022帧、三路视频402个。
- LeRobot v2.1转换逐帧向量核验和视频解码通过；训练物理state/action固定为右臂7D，视觉为mid/left/right。
- HPC job `646566` 已在4×H100完成5000步忠实Stage 1训练（`COMPLETED`, `0:0`, `01:01:03`），最终checkpoint为`step_4999`。
- `step_4999`已完整同步到Cobot并通过零publisher的checkpoint恢复与三次固定输入推理；稳态推理约76.8 ms，发布状态为`offline_validated`。
- 已建立隔离的7D online配置、右臂ROS1适配、动作归一化和真机warmup入口；尚未执行本cohort真机rollout或warmup。

## 数据质量与转换工具

- `tools/audit_expert_hdf5.py`：检查14D原始证据、右臂7D有效性、三相机同步、固定左臂、示教覆盖及action/qpos异常，输出可重复的训练选择清单。
- `tools/analyze_expert_trajectories.py`：汇总右臂起终点、路径长度、步进、跟踪误差和轨迹离群项。
- `tools/convert_experts.py`：只消费冻结的质量清单，写出右臂7D state/action和三相机LeRobot v2.1；逐帧校验Parquet向量、完整解码视频并复核源文件哈希。两条合成episode烟雾测试已通过，正式数据尚未运行。

## 验证

```bash
python3 validate_contract.py
python3 validate_contract.py --require-scene-reference
```

第一个命令验证 Stage 0 和场景合同结构；第二个命令只有现场参考证据完整后才通过。

## 真机入口

操作员完成CAN、机械臂、三相机和`plug2`位姿后，在Cobot执行：

```bash
cd /home/agilex/jiaan/project/cobot-web
./scripts/ui_up.sh
./scripts/rlt_v3_up.sh warmup
```

进度与停止：

```bash
./scripts/rlt_v3_status.sh
# rlt_v3_up终端中Ctrl-C：停止本次Session/在线组件，保留Stage 1模型
./scripts/rlt_v3_down.sh  # 最终释放Stage 1显存
```

完整数据门槛、标签策略和warmup后检查见`WARMUP_HANDOFF.md`。
