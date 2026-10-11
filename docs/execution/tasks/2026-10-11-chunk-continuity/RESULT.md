# 命令平滑候选：实现、对照与交付

## 实测与边界

已验证：2026-10-11本轮GET `/api/rlt/session` 与 `/api/collection/model` 确认当前plug-v3-hil8-explore、Actor3640、Learner关闭、RTC开启、50Hz发布、EMA开启；29条完整Episode的实际trace均记录逻辑20Hz、发布50Hz、tau.08、replan5、delay上限4。源基线5960622，Actor SHA020a84d83a2d9211689e6756a834109fd515a145effe7f8cbe1a21b62fc0ebf5。快照时waiting_scene/paused，未据此取得运动权。

已验证：共享VLA CausalJointFilter是一阶EMA加命令速度限制，没有硬加速度/jerk约束。RLT按逻辑目标插值到50Hz，滤波状态跨chunk保留，RTC固定前缀不会被新Actor改写。检查未发现EMA未启用或每chunk被重置；当前参考目标确实发生来回变化，详见Online FINISH_AND_MOTION与BATCH3。不能把参考往返唯一归因于纯Stage1、RTC或探索。

## 最小候选与比较

RLT本地新增`cascade_filter.py`，仅新入口`plug-v3-hil8-explore-smooth`采用两级因果低通，等效tau.12。不是强制加速度/jerk限制，也不是修复模型决定反复换方向的算法。两级系数按alpha_p=2alpha/(1+alpha)选择，使总离散低频延迟与该tau一阶EMA一致。50Hz下旧tau.08低频延迟约70.4ms，新tau.12约110.3ms，增加约40ms。六关节滤波，夹爪沿原有固定/限速路径。

固定相同历史逻辑目标、时序、初始化/HIL边界，使用实际生产类做CPU开环数值复放；没有重跑模型，也未模拟新命令影响的视觉/接触反馈。旧EMA重算与现场命令各Episode RMSE中位数4.14e-6mrad、最差1.03e-5mrad，证明该滤波对照正确重构基线。加速度由50Hz名义命令差分计算，非电机实际加速度。29Episode为重复选型DEV，未做新模型训练，没有独立TEST；统计单位完整Episode，图示IQR不是置信区间。

| 方案 | 加速度RMS比值中位数 | 加速度绝对p99比值中位数 | 命令相对原目标RMSE中位数 |
|---|---:|---:|---:|
| 当前EMA .08 | 1.000 | 1.000 | 1.486mrad |
| EMA .12 | .702 | .709 | 2.051mrad |
| 两级等效 .08 | .838 | .848 | 1.621mrad |
| 两级等效 .12（候选） | .530 | .538 | 2.208mrad |

已验证：候选29/29轮均降低加速度RMS（比值.501–.556），中位下降47.0%；全程六关节命令路程中位下降12.4%，末3秒下降15.0%。末3秒仅时间窗口，非语义阶段；路程下降不是插入成功提升，也不必然全是消除无效动作。跟随原目标RMSE上升48.4%，这是滤波代价，非TCP毫米误差。A6000纯filter单次p99约.024ms，最慢.290ms，不代表现场总控制耗时。

![同目标对照](../../../../outputs/chunk-continuity-20261011/comparison.png)

选择候选的理由：相同等效tau.12下，两级较单级进一步降低高频命令加速度，CPU开销小，无新依赖，能保留独立回退。没有盲目继续加大tau。图中较大幅度慢往返依然存在，尚未解决其闭环根因；不能交付为“已经不抽插”。

调研：RTC处理推理期间固定前缀和异步补全，不保证所有预测方向合理（[原论文](https://arxiv.org/abs/2506.07339)）；Ruckig可以约束速度、加速度和jerk，但任意变化目标的直接跟踪仍需处理滞后与可行性（[官方教程](https://docs.ruckig.com/tutorial.html)）。本批不额外引入Ruckig依赖或声称两级EMA具有其硬约束保证。

## 代码与验证

- execution_profiles/async_execution：新入口选用两级filter，执行状态/逐步trace带smoothing_order=2与tau=.12；旧入口阶数1、tau.08保持原路径。
- exploration_collection：新入口沿用相同Actor/归一化、确定性基础Actor、相关有界探索、RTC与replan参数，关闭Learner，使用独立smooth-v1录制/Replay/trace；不是新训练模型。
- replay_feature_cache：新入口同属固定Stage1/key42，保留精确输入缓存及原有关闭开关，避免新入口退回重复前向。
- 暂停/HIL/终止清空两级状态，恢复重新锚定；HIL不发布策略动作。保留原命令速度/反馈跟踪安全检查，超时保护没有放宽。
- Actor的未滤波提议、Reference和真正发布命令仍分开；原生C10 Replay回归逐数组匹配实际执行，并完成CPU小网络两步schema兼容检查（非现场训练）。

已验证：首轮132项中131通过、1失败，失败来自新增测试给了不存在的profile别名；修正测试后相关52项通过。追加缓存/候选入口合同26项通过（包括新增2项）；总覆盖134个唯一测试。原测试通过项未因新功能而跳过。日志tests.txt/tests-recheck.txt/tests-cache.txt保留失败与复测。

## 使用与回退（本候选验收建议）

网页候选标签：**插孔 · HIL8 平滑试验（固定权重／不训练）**，ID `plug-v3-hil8-explore-smooth`。用户结束当前轮并释放旧运行后选择该入口、保持50Hz/RTC/平滑开启，再加载并手动开始采集；源码安装不等于正在运行的旧进程已切换。Actor仍3640，Learner仍关闭。基础Stage1可沿用网页已有保留/复用能力，不手动重启现场服务。

只比较执行：先固定同一插孔位置和初始位姿，旧/新各少量交替轮次，分别标注自主成功、辅助成功、失败；重点看末端反复抽插、chunk衔接、纠偏响应和插入耗时，不能只看动作变慢了。出现明显迟钝、推压/跟踪异常或超时就停止当前轮并回退原`plug-v3-hil8-explore`。不自动启动训练，不向其他模型推广。

新原始录制：`/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/exploration-hil8-7281-smooth-v1`。
新Cobot运行输出：`/home/agilex/jiaan/project/rl-platform/outputs/exploration-collection/hil8-7281-smooth-v1`。
旧路径/旧数据均保持。新旧入口探索随机序列不保证相同，因此真机对照需多轮并记录seed；历史开环比较才严格使用相同输入序列。

## 产物与复现

A6000产物根：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/chunk-continuity-20261011/`。
`runtime.json`当前只读快照；`extract_commands.py`通过A6000转发Cobot Python读取29条完整trace；`commands.json`数值证据及原trace SHA（无图像/权重）；`compare_filters.py`重放；`filter-comparison.json`每Episode统计；`comparison-summary.json`聚合；`filter-curves.json`曲线；`plot_comparison.py`生成comparison.png/jpg。

命令：A6000 `python3 outputs/chunk-continuity-20261011/compare_filters.py` 和 `plot_comparison.py`；CPU测试在本任务worktree用`PYTHONPATH=.:third_party/openpi-rlt/rlt_online_rl/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/runtime-restore-verification/envs/online/bin/python -m pytest -q` 后接tests日志列出的测试文件。共享VLA源码只读，现有docs/MIGRATION.md未提交改动保持。
