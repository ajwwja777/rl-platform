# 低频探索候选交付（2026-10-12）

## 已验证的改变

不叠加两级EMA，不改Stage1/Actor权重、探索幅度或共用VLA。旧plan_ar模式保留，新可选episode_constant每个完整Episode只抽一次六活动关节有界偏移，1秒五次曲线平滑加上；跨计划在绝对时间上连续。暂停/HIL再进入时重新平滑加上同一Episode偏移，新Episode重抽；夹爪不变。RTC已提交前缀及探索逐行来源保留，Replay仍记录逻辑步实际发布动作。

代码0b9dd4a（A6000已提交push）。新入口 `plug-v3-hil8-explore-constant`，网页名称**插孔 · HIL8 低频探索候选（固定权重／不训练）**。固定既有HIL8 Learner7281/Actor3640；Stage1、归一化、RTC、逻辑20Hz、发布50Hz、原单级EMA tau0.08均不变；Learner关闭，不会边采边更新。原入口 `plug-v3-hil8-explore` 保留。

## 离线证据与限制

主产物 `/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/matched-learning-20261012/temporal-exploration-v2/`；输入29历史Episode为重复DEV，不是独立TEST。源 `outputs/chunk-continuity-20261011/commands.json` SHA f43140f322b2140161e40c5f1d4c81d3df79295f5e473daba4ba6b27e28bc039。

固定历史观测/Reference/Actor目标、EMA与50Hz发布插值，仅替换探索偏移时间序列；每集采用首个已采偏移，未按成功挑选。旧模式重构实际发布命令的Episode RMSE中位数4.14e-6 mrad。与旧探索相比，常量候选命令加速度RMS中位−9.86%，命令路程−13.48%，末3秒命令路程−19.40%。不同模式同Episode配对，未把重叠窗口当独立样本。

![29集开环命令复放](../../../../outputs/matched-learning-20261012/temporal-exploration-v2/temporal-comparison.png)

**已验证**：减少这批记录中探索变化带来的命令高频成分。**证据不足**：闭环TCP、接触往返、成功率是否改善。原计划本身的变化仍存在，不能称已消除抖动或大幅抽插；关探索的开环结果也有残余。历史固定观测复放不能模拟策略随新轨迹改变输入。运行实现按绝对时间算ramp，复放按历史逻辑时间近似。

**动作契约边界**：现有20Hz逻辑Replay存每段最后50Hz发布；中间发布仍在trace。本次测试保证新模式没有破坏这一契约，不把它升级为“全部50Hz动作逐条作为训练动作”。训练期RTC、最终Actor前缀条件训练没有实施。详见[ACTION_CONTRACT](ACTION_CONTRACT.md)。

## 安装与验证

A6000 86项相关CPU回归通过（与学习任务共用一次统计，不累加），含时间重叠、边界重置、RTC前缀、逐行偏移来源、夹爪不变和Replay最后发布一致性。4个文件同步到Cobot：

- configs/deployment_models.json
- integrations/cobot_runtime/exploration_collection.py
- methods/openpi_rlt/cobot_adapter/exploration.py
- methods/openpi_rlt/cobot_adapter/async_execution.py

安装前只读确认无模型进程/无录制，使用模型操作互斥锁并核对旧SHA；安装后CPU启动配置校验、SHA、权重/norm/旧Replay/共用执行代码与监测进程身份均不变。无服务启停、模型加载、GPU计算或机器人运动。正式API经过其30秒后台目录刷新后已返回新入口available=true、training_enabled=false；不是只看JSON文件存在。现场原模型状态error是此前退出状态，本次没有恢复或切换。

收据：`outputs/matched-learning-20261012/field-install.json`、`catalog-verification-v2.json`（首次目录刷新前查不到新项的catalog-verification.json保留）。

Cobot备份：`/home/agilex/jiaan/project/rl-platform/outputs/matched-learning-20261012/field-install/before/`。
新运行输出：`/home/agilex/jiaan/project/rl-platform/outputs/exploration-collection/hil8-7281-constant-v1/`。
新默认录制：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/exploration-hil8-7281-constant-v1`。页面若显式选自定义目录，按已实现的网页选择契约记录实际目录，不凭默认路径判断保存位置。

## 用户验收与停止条件

1. 刷新网页，在数据采集选择上述**低频探索候选（固定权重／不训练）**，由现场负责人加载；先核对模型/Actor/RTC50Hz/单级EMA/学习关闭及专用保存目录。不要选旧“两级平滑试验”混入这次对照。
2. 同一插孔位置、同初始姿态，用原探索与低频探索交替做少量完整Episode；记录位置、实际动作、HIL起止、自主成功/辅助成功/失败，不能将人为救回计为自主成功。第一轮先关注动作连续性，安全接管保持可用。
3. 若出现更大往返、跟踪异常、接触恶化、超时或动作来源不一致，即停止候选并回原探索入口。未完成新数据身份/实际发布与Replay核对前不继续学习；本候选本来就不启用Learner。

源码回退只在释放模型且取得现场互斥后逐文件核对备份恢复，不在运行中热替换。通常只需选择旧入口，无需撤销安装。新训练权重未上机；本次采集验收不等于Online训练放行。


## 2026-10-13 网页分组修复

用户截图揭示此前只核验API可见遗漏了前端去重：新旧探索共用checkpoint、training_method、frozen模式，model_picker.weightKey将候选合并。已将constant候选方法身份独立为 `frozen-hil8-exploration-constant`；同一真实model_picker.js在JSDOM中通过“方法切换→版本选项→选中ID/提示”验证。未修改web代码、模型、训练参数或执行参数。之前的“网页可见”应限为API登记可见，不能作为此前下拉框可选证据。

实际方法下拉名称：`frozen-hil8-exploration-constant`；版本／模式：`7281 · 冻结 · Actor 3640`；选中提示含 `plug-v3-hil8-explore-constant` 和“低频探索候选”。旧 `frozen-hil8-exploration-collection` 保持。产物 `outputs/exploration-picker-20261013/`。


## 2026-10-13 现场反馈的验证边界纠正

用户反馈抖动可接受、移动插孔约12/15成功。但只读核查session-use=evaluation、38底层记录均Replay写入0、private trace为空；该路由按设计不施加探索。找到两条完整网页成功记录，无法还原口述全部轮次。因此不能把这次反馈当作episode_constant探索已生效或改善抖动的证据；也不能把其余底层success0记成任务失败。详见[续训任务的现场核对](../../../online/tasks/2026-10-11-online-learning-evidence/TEACHER_CONTINUATION.md)。不改标签、不回填Replay、不自动重启或采集。
