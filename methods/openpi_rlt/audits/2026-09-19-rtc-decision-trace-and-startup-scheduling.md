# 2026-09-19 RTC决策时序与记录补齐

状态：离线修复；未重启服务、未执行机器人动作，尚无新真机交付版本。此前候选仍为 rejected_after_onsite_trial。首轮归档提交82d1f81已推送GitHub。

## 新发现与修复

1. Runtime在初次推理完成、next_tick仍为0时再次检查request_due，造成同一generation/tick=0连续两个计划：先d0再d6，后者覆盖前者的部分未来动作。历史trace实际存在该序列。RTCQueue现记住last_request_tick；正常请求序列固定为0/10/20...，暂停/恢复新generation重新从d0开始。没有改变速度、加速度、tracking或workspace限制。
2. 86条有效rollout、877个已接受计划中，96个post-prefix动作尾段没有完整执行；877个均缺完整raw/conditioned计划记录，11条回合终局位于最后一个计划常规C10区间之外。prefix与实际发出命令核验无不一致。延迟动作本身并非错误；应以prefix扩展状态、实际决策间隔明确TD，不能把终局强行移到第10步。
3. 新decision_trace schema2保存完整reference/raw/conditioned计划、actor key、决策tick、动作开始/结束tick；保留已承诺prefix，暂停后的迟到计划不被接受。该记录补齐未来可审计性，不宣称已经修复历史训练数据。
4. reference-only重建工具先按(generation,tick)选择最新sequence，再比对历史滤波器。75条回合中52条满足唯一profile匹配且各决策至少有一条可核对的已执行命令。其余存在零执行计划或多个profile等价，保守不自动还原。历史无关节clamp配置只在离线诊断对象中重放，不修改生产限制。
5. 发现legacy online_contract.sample把next_action误转成bool；已改为float32并用负值fixture核验保真。该字段仅旧online_learner探针使用，supported_learner不使用它，不能据此认定它造成最近现场失败。旧候选不重新启用。

## 验证与位置

- clock、conditioning、RPC和decision trace/调度：25 passed。
- 单独sampler数值回归通过：next_action=-0.125保留float32，next_hil仍为bool。
- session_phase实际读取为stopped，发布指针仍拒绝。未启动新训练、未重启服务。
- Cobot根：/media/agilex/Getea1/jiaan/projects/rlt。
- 源码：methods/openpi_rlt/plug_v2/{decision_trace,rtc_history_audit,rtc_reference_reconstruction}.py及runtime/rtc_queue/online_contract；备份runs/plug_v2/deploy-backup-trace-contract-20260919。
- 证据：runs/plug_v2/diagnostics/upstream-audit-20260919/{rtc-decision-coverage,rtc-reference-reconstruction,numeric-next-action-test}.json。
- 源码/证据哈希：plug_v2/execution_sources_20260919_continuation.json。

## 未满足的发布条件

- 原仓库同步7维actor还不能直接用于99维RTC接口。
- RTC延迟决策的完整action、终局区间和HIL后继契约尚待形成可训练、可验证的数据适配，不能仅把训练步数加长。
- raw动作优势经执行滤波仍大幅消失；尚未证明完整执行路径比旧warmup更准确、更平滑。
- 固定候选成功率与在线持续改善均未验证。不要执行旧demo命令验收本次源码修复；须完成新不可变候选及完整离线链路后另行交付。
