# plug_v3_yyshadow 真机 warmup 交接

## 已固定的训练合同

- Stage 1：Yyshadow/openpi-RLT commit `c1e40ac360185778c98cf20da2820e22d2d415e7`，Pi0.5，右臂7D，H50，1×2048 RL token，5000步，seed 42。
- online：右臂7D、C10、20 Hz、delta chunk、actor/critic均为2层×256、fixed std 0.002、reference dropout 0.5。
- warmup门槛：600 replay transitions；达到门槛后执行20,000次warmup更新，每500步发布actor。
- 视觉：mid、left wrist、right wrist；只向右前臂发布策略命令。
- 本轮不继承`plug_v2`的模型、replay、norm stats、actor/critic或自定义loss。
- Cobot发布已完成无动作恢复验证：`step_4999`权重树哈希`e73ef3f30c72bc946e7f3e9b5f73d5e97c99021716c0004f8d90aefe40e31270`，稳态固定输入推理约76.8 ms，验证时`robot_publishers=0`。

## 现场启动

完成CAN、机械臂节点、三相机、`all --pose plug2`与`mid --pose plug2`后：

```bash
cd /home/agilex/jiaan/project/cobot-web
./scripts/ui_up.sh
./scripts/rlt_v3_up.sh warmup
```

`rlt_v3_up.sh`首次加载Stage 1可能需要数分钟；加载完成后在`http://127.0.0.1:8015/`开始每个Session。warmup阶段执行的是冻结Stage 1 reference，actor尚不控制机器人；右后臂HIL会按真实执行动作写入replay。

另开终端查看进度：

```bash
cd /home/agilex/jiaan/project/cobot-web
./scripts/rlt_v3_status.sh
```

以`replay_transitions >= 600`为停止采集标准，不以episode编号为标准；按当前短插入轨迹通常约40–70个episode。保留真实分布，不为了凑比例伪造标签。建议至少覆盖：

- 15条以上自主失败/near miss；
- 15条以上在真实偏差点介入并最终成功的HIL；
- 所有自然发生的自主成功；
- abort、坏帧、错误复位不进入正式replay。

达到600后完成当前episode，停在“等待场景复位”，不要开始下一轮，让learner继续完成20,000次warmup更新。`rlt_v3_status.sh`显示`warmup_ready: true`后按Ctrl+C：这只停止本次Session、actor/critic/replay进程，已加载的Stage 1模型保留。

最终关机或需要释放GPU0时：

```bash
./scripts/rlt_v3_down.sh
```

## warmup 后的门槛

不要直接凭loss进入在线RL。先检查：

1. learner checkpoint和actor snapshot可恢复；
2. 成功episode的Q在精细插入末段高于失败episode；
3. HIL动作拟合优于或至少不劣于reference；
4. 固定actor真机小样本无明显抖动、发散或持续偏孔；
5. replay中7D state/action、source、terminal reward和episode标签一致。

通过后才运行：

```bash
./scripts/rlt_v3_up.sh online
```

该入口从同一replay与learner checkpoint继续，不重复导入数据。
