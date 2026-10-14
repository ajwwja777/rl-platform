# Stage1 已有证据复核与候选时间合同

负责人：RLT 模型执行对话。2026-10-14，用户恢复离线诊断；状态：本次证据复核子任务完成；整个Online能力目标未完成。

## 问题与验收

接续摘要遗漏了10月5日全量模型诊断，不能据此重复GPU前向。核验已有数组/源数据/指标，区分分析图与真实服务路径；用同一输入核对原5k、HIL8和Teacher5是否能直接切换Reference时基。输出可恢复证据、逐Episode比较、明确保留/否决与缺口。

## 范围和资源

基线79b825795d61f61b213acc3a4ea66ed89dc08922；独立分支audit/stage1-evidence-reconciliation-20261014，worktree位于scratch/rl-platform/stage1-evidence-reconciliation-20261014/rl-platform。仅CPU4–5只读缓存/权重诊断，独立outputs/stage1-evidence-reconciliation-20261014。A6000 GPU按EXPO协调登记保留，不占用；不连接或更改Cobot服务、模型、机器人、生产Replay/默认/固定上游。

## 已验证

- 历史stage1_full有134完整Episode/19022帧，全部Stage1训练见过；本轮逐缓存SHA、源Parquet SHA、逐集回执和主要误差指标重算通过。
- 实际服务路径有两种提示词各3完整Episode/446帧；这不等于134条全量服务路径认证。历史跨进程一致性只有3真实输入，单独保留其边界。
- Teacher5在10月13日无动作加载回执中仍为Reference30Hz、逻辑20Hz、发布50Hz/RTC/EMA。该历史回执不代表10月14日当前现场状态。

## 经验与唯一下一步

不要把“全量模型前向”“全量实际服务路径”“三个输入跨进程一致性”互相替代。重采样Reference改善本身拟合，不代表已训练Actor换输入也改善。同输入三权重冻结对照已完成，见[RESULT](RESULT.md)及输出；不重跑旧训练扫参，不以训练内误差或Q决定自主收益放行。

历史证据：outputs/offline-model-selection-20261005；当前执行合同见docs/execution/tasks/2026-10-11-chunk-continuity/ACTION_CONTRACT.md。正式流程不作变更，本任务仅记录事实。
