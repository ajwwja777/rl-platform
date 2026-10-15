# realtime_expo：先采集 Reference，再做 Warmup

## 当前问题与用户纠正
2026-10-15：用户指出 Warmup 应先采集新 Reference rollout，并要求提供 Reference baseline 与现有 Actor 两版本。
此前仅用44条TRAIN成功示范、551条expert transitions完成Warmup7000，未包含新Reference rollout。该执行跳过了用户要求的顺序；保留为纯专家对照，不能冒充rollout-informed Warmup。Stage1输入49示范为44TRAIN/5反复选型DEV，无独立TEST。

## 范围与验收
负责人/root；分支fix/realtime-expo-reference-20261015；RL/Web隔离worktree在scratch/rl-platform/realtime-expo-reference-20261015/。
本次提供纯Stage1 Reference4999与原7kActor3500两个不学习入口；同双腕相机、左臂10Hz逻辑/默认50Hz发布RTC/滤波。先采集结果再决定Warmup；不开展新训练、不运动、不改既有Replay或权重。
Reference使用既有warmup阶段的精确ref_chunk旁路，source=BASE、Actor版本-1，不调用RL Actor推理；禁用phase controller与Learner。采集journal从独立空库开始，不把成功专家Replay复制进去。
旧Online登记撤下，launcher拒绝Online以阻止旧客户端绕过；既有Online资产原样保留。
Web显式展示Reference，保留其他项目默认隐藏重复Reference的行为；7k标注纯专家对照。

## 版本与数据位置
- Web：plug_left_2cam → RLT → realtime-expo-rlt。
- Reference：realtime-expo-rlt-reference；checkpoint /home/agilex/jiaan/data/rlt/realtime_expo/stage1_4999；记录 /media/agilex/Getea1/jiaan/data/datasets/realtime_expo/recordings/rlt/reference；Replay /home/agilex/jiaan/data/rlt/realtime_expo/reference_rollouts_20261015/replay_journal.pkl。
- 对照：realtime-expo-rlt-frozen；原release_20261015/frozen/actor_snapshot/actor_snapshot.pkl；原frozen记录和Replay保留。
- 新入口启用独立evaluation-traces；只有重新加载时生效，不声称已有评测补齐trace。

## 已验证
- 45项RL CPU测试：Reference配置不加载Actor pickle、不复用expert journal、拒绝错误checkpoint/被污染的未初始化Actor位置；原生phase旁路精确相等、左右臂/相机/阶段/评测trace回归。
- 36项Web Python测试通过、1跳过；9项前端DOM测试通过。Reference可见、无虚构Actor/Learner版本、评测/采集用途不启用学习。
- bash语法及git diff --check通过。初次Web测试误用训练环境缺fastapi、Node未在默认PATH；换项目现有Web环境与已安装Node后通过，无修改生产环境。
命令与产物：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/realtime-expo-reference-20261015/{rl-tests,web-tests,ui-tests}.log。

## 证据不足与唯一下一步
现场安装/激活结果待追加。单元测试不证明自主成功或平顺性。
先使用Reference采集完整Episode并标注自主/辅助成功、失败；与原7k在相同位置/初始姿态/执行设置下比较，记录移动目标的分组。数据完整性、实际执行动作和标签核对后再设计Warmup，不自动触发训练。
