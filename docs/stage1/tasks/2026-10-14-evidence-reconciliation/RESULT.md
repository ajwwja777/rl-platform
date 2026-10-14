# 2026-10-14 复核结果：保留 Teacher5 输入合同，否决仅切换 Reference 时基

本轮已完成缓存复核和冻结候选比较，没有训练、发布新权重、占用GPU或操作Cobot。Teacher5仍是已有受控候选，不是“已证明Online持续改善”的交付认证。

## 结论

| 项目 | 状态 | 直接证据与边界 |
|---|---|---|
| 历史全量Stage1前向是否存在且可复用 | 已验证 | 134完整Episode/19022帧；140份缓存（全量134+两个服务提示词各3）逐SHA、Parquet SHA、逐集回执与主要MAE/RMSE/bias/p95重算一致。134条属于分析计算图，全被Stage1训练见过 |
| 全量真实服务路径是否通过 | 证据不足 | 服务路径各3Episode/446帧；跨进程精确恢复另有3输入历史收据。不能把这三类范围混合成134条服务认证 |
| Teacher5在旧示范上是否整体遗忘 | 已验证（仅本批拟合） | 六关节均值1.612mrad，HIL8为1.640，原5k为1.651；Teacher5较HIL8有124/134集改善。不能替代位置OOD、接触控制或新HIL验收 |
| 只将旧Actor的Reference改成20Hz | 失败（该离线代理） | 三个Actor均变差，Teacher5全量误差+20.75%；3条服务路径同方向。不能推广为所有一致时基重训方案失败 |
| 完整50Hz命令轨迹与训练动作是否同一序列 | 证据不足/尚未满足完整轨迹要求 | 历史29Ep是20Hz每段末命令进入训练；中间50Hz命令不逐条进入C10。不能将逻辑步严格匹配宣传为逐发布轨迹匹配，也不能仅凭不同频率断言算法必错 |
| 自主Online增益/独立泛化 | 证据不足 | 本轮没有独立TEST或新真机结果，不能按训练误差、Q或bootstrap区间放行 |

## 同输入冻结对照

固定Stage1缓存、相同Episode和20Hz物理目标插值、C10、确定性Actor；使用各候选配套归一化，夹爪按现有HeldGripperCritic投影，主指标只看六活动关节。下面是完整Episode等权MAE，单位mrad；134条全部Stage1 TRAIN见过，反复分析过，不是独立TEST。转换14条不是Stage1测试；`warmup_expert_training_seen`只表示原Warmup专家物化，不能概括后续所有候选暴露历史。

| 冻结权重 | 原Reference合同 | 仅改Reference为20Hz | 误差变化 |
|---|---:|---:|---:|
| 原5k | 1.65074 | 1.93451 | +17.19% |
| HIL8 7281 | 1.63964 | 1.93036 | +17.73% |
| Teacher5 8127 | 1.61160 | 1.94599 | +20.75% |

Teacher5−HIL8配对差为−0.02804mrad；完整Episode bootstrap95%区间[-0.03183,-0.02449]，124集改善/10集变差。Teacher5−原5k为−0.03914mrad，130集改善/4集变差。这是训练拟合变化区间，不是自主成功率区间；不能将不同训练历程的候选差异全归因teacher权重。单因素teacher50/5证据仍以Online任务TEACHER_CONTINUATION.md为准。

实际服务路径缓存的3集、训练提示词条件下，原合同分别1.85368/1.84451/1.80578mrad；换20Hz分别1.96516/1.95919/1.95238。当前Teacher5配置用短训练提示词，本轮也保留另一长提示词分支，不混为同一输入。

Reference自己重采样后，全134集误差2.01045→1.35819mrad；已训练Actor却相反。解释限于：更换Reference构成已训练Actor的输入分布改变，Reference自身更贴合不保证Actor输出更贴合。没有证明该差异独自解释历史Online不学习。

## 实际修改、验证与异常记录

- 新增`scripts/audit_stage1_cached_evidence.py`，无需模型/GPU，核对缓存、源Parquet、回执、覆盖、有限性和重算指标，输出逐集证据与图。
- 扩展`scripts/compare_reference_timebase.py`：显式checkpoint/norm成对传入、单缓存比较、固定夹爪Critic投影、缓存SHA与工具版本。默认原5k路径保留。未修改训练/执行算法或默认配置。
- 7项既有Stage1动作指标测试通过；6组实际CPU冻结比较完成、产物exit0；3组权重/归一化运行前后SHA不变。原5k旧报告六关节MAE最大差1.632e-9rad；初设1e-10严格阈值失败，保留日志后按FP32数值回归1e-8判定通过，不能称位级相等。
- 首次独立worktree比较缺openpi_client依赖，退出在模型前；补PYTHONPATH后重跑独立-v2输出。失败日志保留，不算模型失败。
- 已保存首次/加保护后的脚本快照，逐run的tool_sha256区分执行版本。未重跑已完成训练，未把固定夹爪误差叫物理退化。

## 数据、代码、配置、命令和产物

代码基线79b825795d61f61b213acc3a4ea66ed89dc08922，修改分支audit/stage1-evidence-reconciliation-20261014。固定上游未改。

示范manifest SHA c1c3fd4ad2ca0d1602d611c99f9ad97c4792a17882aca22224c7d375e76f1171；历史Stage1/4999参数文件SHA在旧report.identity，缓存使用seed42、10去噪步、H50；本轮不重新读取整份Stage1权重或加载Stage1。新的Actor checkpoint/norm SHA、实际rl_config和逐Episode值见各report及SUMMARY.json。

本轮产物根：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/stage1-evidence-reconciliation-20261014/`。

- `cache-verification/report.json`：140缓存复核；134/19022与两组3/446分别记录。
- `candidate-commands-v2.json`：6组完整命令，CPU4–5、JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES空；`candidate-exits-v2.json`记录退出码。
- `{original5k,hil8,teacher5}-{v2,full134}/report.json`：配置/资产SHA、Q1/Q2/min-Q、每集动作误差；Q不作成功认证。
- `SUMMARY.json`、`candidate_time_contract.png`、`cache-verification/reference_time_contract.png`：汇总与图，已目视检查。
- `post-source-checks.json`、`metric-tests.log`、`argument-guard.json`：验证；初次失败和过严容差失败单独保留。
- 复核命令见`launch.json`；汇总`python3 outputs/stage1-evidence-reconciliation-20261014/summarize.py`。

历史全量与实际服务证据原位置：`outputs/offline-model-selection-20261005/`。10月13日Teacher5真实加载回执：`outputs/teacher5-delivery-20261013/paused-load-verification.json`，其中Reference30Hz/逻辑20Hz/发布50Hz；不是当前现场状态认证。

## 决策、回退和下一步

保留Teacher5 8127及其原Reference/归一化/执行合同，原5k、HIL8继续作为回退；本轮无权重切换。否决“给现有Actor直接重采样Reference”的快捷改法。

本次证据复核子任务完成，整个Online能力目标未完成。下一步主线回到Online任务：对Teacher5真实首轮新增经验核对录制→Replay→实际batch→更新→pending版本；维持分批/不自动发布，比较新旧DEV与独立冻结自主结果。现场当前由EXPO对话协调，本对话不接管。没有新Teacher5数据时不重复旧参数扫；一致时基候选需要所有来源完整H50/输入身份及训练执行共同变更，旧C10不能凭插值补造缺失尾部。完整50Hz解码/RTC状态进入训练需独立设计，不静默改已选候选。
