# 2026-09-17–18 Cobot 平台与 plug_v2 RLT 总结

## 已完成

- 将新公共入口封装在Cobot cobot-platform：home/capture/recover/夹爪与teleop、统一8015录制/RLT网页，模型和方法隔离在rlt，数据统一data/rlt/plug_v2。旧CAN/机械臂/相机launch保留。前后臂并行归位、示教恢复与后臂失能逻辑经历现场调整，历史验证与风险见平台审计，未声称所有机械臂状态永不故障。
- 按用户要求清理旧cohort原始数据/目录；新左相机82条成功插入示教、83片段/9119有效帧转换完成，保留计数/轻量事实，HDF仅在完整转换验证通过后删除。详见数据与平台布局审计。
- 新三视角joint Stage1及training-time RTC完成，选3999（4000updates），Cobot BF16严格恢复及FakeIO闭环通过。只覆盖近插孔已持插头的短插入段，不等于完整抓插任务。
- 75有效warmup（55HIL成功/20自主失败）修复decision/prefix、暂停/HIL分段、reward/mask与后继合同。正式20k critic/10k actor已完成，500–20k对比/现场反馈无明显提升，没有以loss或轮次声称RL有效。
- 按作者human_mask_ratio/bc_human_penalty、Q成功/失败排序及干预时机检查。隔离HIL20/demo30/两类policy25合同实验改善Q分类，但同state动作方向与减速偏好不可靠，未开放新在线优化。
- 最终固定RTC纠偏r1通过原UUID留出、实际CPU输出/过滤器与真实Stage1 FakeIO Session验收。明确为执行相容BC actor纠偏、critic冻结，d0 reference/d6修正。交付 ./scripts/rlt_demo.sh；独立权重/manifest，禁探索/在线cycle，不覆盖旧actor。

## 验证与限制

固定候选相对reference留出动作MSE改善约17%专家/58%HIL/60%真实自主输入BC反事实；仅10个自主反事实留出、非factual RL transition，不能当真机成功率。Q_actor_mean最初3自主窗口AUC1，仅11HIL辅助成功/4失败，无自主成功类。多轮机制诊断复用val，非独立最终科学确认。

实际HTTP wrapper与训练输出误差<2.4e-7rad；prefix不变/被动关节固定/30Hz/.2rad每秒限速/跟踪fail-closed通过，完整FakeIO chunk约96–114ms，无机器人publisher。框架此前未提交修改关联回归72项通过，index重建与doctor无error、7条既有legacy附件warning。

没有由agent启动真实rollout/机械臂动作。收尾时真实Session stopped/actor20000，模型仍预加载，8015网页未运行，下一次由用户ui_up。HPC本轮未访问，旧allocation与服务器路径只能作历史。

## 下一步

使用固定候选做相同场景5次独立自主验收，与reference对照，HIL另计；保持专家左相机视角和插入初态。确认实际纠偏与成功结果后，再修建可靠critic动作排序及在线更新验收。默认latest在线旧入口保留，但不是这次推荐方法。

- [固定候选部署/证据](2026-09-18-plug-v2-fixed-rtc-corrective-deployment.md)
- [现场流程](../PLUG_V2_ROLLOUT_SOP.md)
- [平台单入口与清理](../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-17-platform-single-entry-and-cleanup.md)
- [机械臂恢复与并行归位](../../../../proj-20260829-cobot-realworld-vla/platform/audits/2026-09-17-task2-recovery-and-synchronized-homing.md)
