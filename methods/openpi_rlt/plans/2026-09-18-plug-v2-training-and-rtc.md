> 已确认并实施：三视角joint Stage1 2k→4k+训练时RTC、selected3999部署；reference为joint VLA、独立pureSFT未另训；手动warmup/5条在线UTD5。[最终交付](../audits/2026-09-18-plug-v2-rtc-stage1-and-rollout-delivery.md)。下文为讨论阶段记录，以交付/SOP为准。

# plug_v2 训练方案讨论：Stage 1、warmup与training-time RTC（2026-09-18）

状态：方案待确认；HPC 629492已获4×H100，但未执行GPU probe/训练。本文件在用户确认前不修改已准备参数或部署机械臂。之前准备的入口不含training-time RTC，不得称为RTC模型。

## 模块与目标

Stage 1：微调pi05(VLM+action expert)并训练RL token encoder/decoder；L=L_reconstruction+alpha L_VLA，alpha=1。重建输入/目标stop-gradient，重建损失不回传VLA；VLA通过动作监督训练。decoder重建视觉embedding，不是RL actor，也不负责输出机械臂动作。
warmup及在线RL：冻结pi05与token encoder/decoder，仅更新独立actor/twin critic/target及优化器；decoder通常无需运行。离线warmup包括预灌并提取冻结特征，以及actor/critic初始化优化，两者不要混为同一计数。
同一joint Stage 1的VLA分支可作为无RL reference，固定warmup和在线RL使用同一基础模型，才能隔离RL收益。独立train.py纯pi05训练仍可作为额外监督对照；train_rlt.py alpha0冻结VLA，不能作为纯pi05训练。

## Stage 1建议（尚未开始）

pi05_base固定、三RGB视角、物理14D/pad32、H50、seed42、batch32、FSDP4、workers8；alpha1、1个token、2层encoder/decoder、2048维沿用现有配置。AdamW b1=.9/b2=.95/eps1e-8/weight_decay1e-10、clip1、EMA.99；cosine peak2.5e-5/warmup1000/decay30000/floor2.5e-6。LR warmup1000不是RL warmup。
先2000次更新，审核再续至4000；标签1999/3999，保留中间检查点。论文Stage 1为2000–10000次更新，4000是本项目初始预算，不是论文固定值；验证8条完整原UUID/483 H50窗口，训练74条/4610窗口。检查VLA动作与token重建分别的验证误差及部署恢复，不只查看combined training loss。后续真实成功率需独立现场评估。

## RL warmup建议（本轮尚无新cohort rollout）

沿用复现Ethernet/此前个人稳定配置：2×256 MLP actor/twin critic；C10；batch128；actor/critic LR1e-4；tau.005；critic每步/actor每2步；reference dropout.5；BC10/Q.1/smooth10。最多20000 critic/10000 actor更新；分段100/500/2000/5000/10000/20000检查和选优，不默认最后checkpoint。gamma.99是每控制步折扣，固定源代码TD bootstrap为gamma**chunk_len。探索std.001为本项目执行保守值，不称论文固定std。
经验三组按episode均衡：专家成功30%、当前成功40%（自主/HIL分开统计）、当前失败30%；不是要求采集条数30/40/30。需新视角reference rollout与HIL/失败，不能只用82条成功示教启动Q训练或复用旧camera actor/critic。600 transitions是复现仓库Ethernet预灌默认，20000来自复现配置及旧习惯，不是论文固定warmup梯度步数。数量不是唯一门槛，先确认各组覆盖、质量与留出，再人工启动。
在线方案需重新核验新数据时间轴及采样UTD；论文stride2、UTD5、critic:actor2:1、reference dropout.5、C10。按新有效transitions决定更新预算、从全replay抽样，episode仅为提交/版本切换边界；不能把一episode单独拟合32步称为严格论文复现。此阶段参数尚未定版/实现。

## Training-time RTC接入设计（未实施）

依据训练期RTC论文：随机采样延迟d；前d个动作使用真实无噪声prefix；动作token各自的flow time；仅postfix计算loss；推理denoise每一步锁定prefix。固定openpi-RLT采用x_t=t*noise+(1-t)*action，所以clean prefix time为0（论文另一参数化为1），不得照抄符号。
固定代码无RTC；posemb_sincos与embed_suffix只支持batch级time，Gemma RMSNorm把modulation写成[:,None,:]，需要支持per-action time/条件广播；compute_loss与compute_loss_with_prefix须同时修改。采用独立project-owned实现/overlay，保持原upstream commit与旧部署不变；训练和serving都必须加载同一实现并记录RTC metadata。只有训练loss改变、部署仍普通sample_actions不能称为RTC有效运行。
新数据动作时间轴为30Hz，旧Cobot rollout默认20Hz（本轮只读核验cobot_ros1.py/configs/resolved_online.yaml）。需统一；建议新plug_v2动作/部署30Hz，不能直接套用20Hz。暂定d均匀0–6（最大200ms），H50、执行s10；d_max6<=s10<=H-d_max44。论文实机50Hz时d0–10，同为200ms；不能复制整数而忽略频率。最终延迟范围以新实现端到端p99测量校验。
必须接入异步动作队列与硬prefix推理；暂停/HIL/归位清空队列并重新锚定，旧generation结果不得执行。RL actor/探索也必须保持已承诺prefix不变，否则再次破坏跨chunk连续；replay记录实际执行动作与prefix上下文、锚点/时间，critic不能训练未执行的命令。
所有对照应用相同RTC与执行平滑。原alpha.35/step.01在20Hz下的物理滤波时间约.116s、速度上限.2rad/s；新30Hz可换算alpha约.25/step约.00667，但仅为保持相同物理限制的设计换算，未修改/测试现场参数。速度/加速度与边界指标用同一dt比较，不能靠延迟或过度滤波虚报顺滑。
RTC可降低chunk切换跳变，不保证插入成功，也不替代接触阶段、模型质量或RL actor平滑约束。

## 来源
- RL Token论文：https://arxiv.org/html/2604.23073v1 （IV-A、V及附录B）。
- 固定复现源：https://github.com/Yyshadow/openpi-RLT/tree/c1e40ac360185778c98cf20da2820e22d2d415e7 ；configs/tasks/agilex_ethernet/online_rl.yaml。
- Training-time RTC：https://arxiv.org/html/2512.05964v1 （IV及V-B）。
- LeRobot main RTC文档：https://huggingface.co/docs/lerobot/main/rtc （不是本项目0.1.0环境已有功能，不升级现有依赖来假装接入）。

下一步先确认上述阶段、预算和时间轴，再实现RTC并做零延迟等价、prefix锁定、loss mask、参数shape/恢复和异步队列/HIL清空测试；之后才GPU probe与正式训练。
