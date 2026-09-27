# plug_v2 warmup replay 修复与离线验收（2026-09-18）

状态：用户授权修复完成；实际数据CPU时间轴验证与隔离合成CPU模型保存/恢复通过。正式warmup、正式teacher特征提取、GPU训练及真机效果均未执行。本轮不访问HPC、不启停机械臂或预加载模型、不commit/push。

## 修复

执行源 `/media/agilex/Getea1/jiaan/projects/rlt/methods/openpi_rlt/plug_v2/{replay,replay_contract,learning,training_flow}.py`。旧源备份与全部证据在 `runs/plug_v2/audits/warmup-repair-20260918`，部署manifest新增独立warmup_replay_contract说明与四文件hash，原Stage1交付快照保留。

1. 后继按同一stream/segment/generation及精确控制时间匹配，不再全局searchsorted拼暂停/HIL间隙。真实终局done、控制截断truncated、可bootstrap分开记录；截断无奖励、不引入另一片段的next_z。expert坏帧切分的前片段不伪造成功终局，原UUID最后片段才允许成功奖励。非有限状态/action切断，不丢弃整条剩余有效数据。
2. 自主数据复用实际accepted plan的z/ref/context及prefix，无未来示教prefix、无额外RPC。Runtime.tick是发送后的next_tick，标签按tick-1还原；同一tick多次accepted请求取最后有效计划，其prefix必须逐位等于已发命令。一个decision承诺10个post-prefix动作，后6个可进入下个decision的已锁prefix，仍归当前动作；动作标签涵盖同片段真实执行的10个tail，未执行padding不入Q/BC。奖励与Bellman持续时间按request tick起的实际10控制步区间，不能把奖励平移6步或把控制步换成stride2采样序号。回归验证前一tail后6步等于下一context锁定prefix。
3. 专家/右侧HIL以实际state/images提取cold context，prefix_length=0，不用未来教师动作作为输入。HIL类型按原ROS时间与同generation实际handover mode/phase匹配，避免source snapshot标记在退出边缘晚一帧而遗漏成功尾帧；仍要求数值/相机mask有效。
4. 正向BC只用成功专家、成功右侧HIL纠正动作、或整轮无HIL的自主成功；HIL成功前后自主片段不当正确动作模仿。失败/非监督动作仍用于Q与reference约束。平滑与BC均排除未执行padding，critic显式bootstrap布尔决定是否使用next Q。
5. 正式特征缓存换成 `runs/plug_v2/replay/v2`，learner拒绝旧时间轴版本；按Stage1/源文件签名缓存，来源变化明确拒绝静默覆盖。旧.source.npz事实不变。新rollout按完整UUID初始20%留出，第一次prepare前固化；后续不移动已有train/val。专家维持74/8原UUID划分（75/8片段）。验证batch512覆盖各组heldout episode，候选检查写validation.jsonl，actor结构/部署接口不变。
6. online仍每5条新episode轮间触发，UTD5只计v2新train transitions，排除heldout与历史预算；重启去重行为保留。本轮未开启自动warmup或online服务。

## 验证

- 全部75条真实warmup（55 HIL成功、20自主失败）和82原始专家的83派生片段离线布局通过，288个旧候选跨段连接修复后为0；当前policy正向BC目标0；55 HIL成功全部保留终局奖励。
- 专家4578 replay行：train4139/val439；success2912：train2300/val612；failure162：train129/val33。全部7652行，其中train6568/val1084；这些是修正后的C10采样行，不是原图像帧数，存在stride2重叠。失败policy有10处截断，只有11条尾端观测满足真实终局匹配，不给未匹配终局伪造done；截断不bootstrap亦不伪造正奖励。
- 新成功55原UUID train44/val11、失败20原UUID train16/val4；实际production split_registry尚未写入，提议仅在audit/proposed_split_registry.json保存，首次warmup prepare会按同一规则固化。
- 19项unittest通过：精确十步后继、同generation暂停、切换、committed prefix核验与credit、cold teacher无未来输入、padding、坏缓存拒绝、源变更缓存保留、BC mask、explicit bootstrap、actor每两步更新等。
- 既有在线cycle六项隔离检查通过，无优化器subprocess；合成CPU100 warmup+10 online测试通过，包括新Sampler、accepted artifact、actor实际加载/推理与优化器/target恢复。测试权重已删除，不是正式warmup模型。
- 真实数据元数据/源facts/mask/trace前后hash一致。预加载模型PID455734/start2358420保持，当前Session stopped；无正式v2缓存/正式actor/新特征提取，无model RPC/机器人动作/节点重启。

证据：actual_data_validation.json、actual-data.log、unit-tests.log、cpu-publish-test.log、validation.json及before/。teacher在CPU实际布局审计中使用合成占位特征，只证明时间轴/监督mask；首次正式图片特征提取及完整20k训练耗时仍待执行，不把此验收称为训练或真机效果。

## 下一步：手动训练

维持原批准参数：专家/成功/失败采样30/40/30，batch128，actor/critic LR1e-4，gamma每控制步.99，tau.005，critic:actor2:1，BC10/Q.1/smooth10，最多20k critic/10k actor，检查100/500/2k/5k/10k/20k选优。保留同一Stage13999和原RTC/执行条件化。

当前Session已结束，模型仍在；不需先rlt_down或重启机械臂。由用户或新明确训练指令启动：

```bash
cd /media/agilex/Getea1/jiaan/projects/cobot-platform
./scripts/warmup.sh
```

先准备/缓存真实特征再训练，可能需数分钟，不能用旧optimizer-only基准承诺完整时间。正式产物仍在projects/rlt/runs/plug_v2/learning/warmup/{actor.pt,ready.json,validation.jsonl}。仅accepted后使用 `./scripts/rlt_up.sh --frozen-actor` 做独立自主验收；通过后再默认rlt_up在线更新。改善仍需固定场景真机成功率/误差验证，HIL成功分开统计。
