# 2026-09-18 plug_v2 RTC Stage 1 与 rollout 交付审计

## 范围与当前状态

本轮授权新三视角训练、部署、统一8015网页及无动作验证；不创建Goal，不执行机械臂动作。HPC训练已完成，最终Cobot部署与无动作闭环验证已完成，尚不声明现场成功率或新warmup效果。旧cohort模型/critic/replay不导入新cohort。

82条原始成功专家插入片段，9120帧/30Hz；排除1坏帧派生83片段/9119有效帧。原UUID train74/val8，H50完整窗口4610/483，不泄漏同源片段。LeRobot0.1.0/v2.1三路480×640 RGB视频，H264 CRF10有损压缩；完整非图像事实、节点、源身份和校验保留。82原始HDF及6重复预览在完整转换/官方读取验证后按授权清理。数据根 data/rlt/plug_v2/demonstrations/lerobot。

## Stage 1 与证据

固定上游 c1e40ac360185778c98cf20da2820e22d2d415e7，upstream-only加执行服务器自有overlay。Pi05及RL-token encoder/decoder联合训练，alpha1；alpha0会冻结VLA，不能称为纯Pi05 SFT。当前reference是联合Stage1模型的VLA分支，无RL warmup；独立纯SFT入口已准备，但没有额外训练第二个纯SFT模型。

seed42/batch32/FSDP4/workers8/H50，峰值LR2.5e-5，LR warmup1000、cosine30000/min2.5e-6，AdamW(.9,.95)/eps1e-8/WD1e-10/clip1/EMA.99。training-time RTC延迟0–6均匀抽样：已执行prefix flow time0，只对postfix计算loss；逐动作时间条件与硬prefix serving，保持模型参数形状/名称。训练及推理均30Hz/C10/最多6步200ms覆盖，10次denoise。不是只在推理端加平滑。

HPC执行根 /data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl：
- allocation629492，ACD1-11 GPU0–3四张H100；01:10:07获配，训练/验证/导出完成后已释放，凭据 runs/plug_v2/allocation_release.json。
- 20步GPU probe通过，约1.7step/s，首次编译约3min。
- 2k于02:11完成checkpoint1999；续到4k于02:47完成checkpoint3999。独立留出8UUID×3anchors×2延迟，48预测，严格实际167参数叶子恢复。
- heldout1999右关节10步RMSE0.0130774rad/postfix0.0239589；3999为0.0125258/postfix0.0233236。按10步RMSE选择3999，不把训练loss或离线误差称为真机效果。heldout flow loss0.0227952、token reconstruction0.1383844。
- runs/plug_v2/checkpoint_selection.json、heldout_1999.json、heldout_3999.json、stage1_execution.json、train_rtc_4000.result.json。
- runs/plug_v2/inference_exports/3999：只导出BF16推理params/assets，不传优化器。与经过验证的BF16恢复逐叶子167形状/类型/位值完全一致，export_validation.json及全文件transfer_checksums.json。原FP32训练checkpoint留HPC。

train-only norm SHA256 373d9a01bbcc0907dbbc27b091768774f98933cb28cdf3404a14f378ee02b49d。新RTC模块及实际部署完整代码在HPC/Cobot自己的 methods/openpi_rlt/plug_v2；框架目录只留摘要、方案和来源指针，旧准备代码不作为RTC生产入口。

## Cobot方法与平台契约

方法根 /media/agilex/Getea1/jiaan/projects/rlt；平台入口 /media/agilex/Getea1/jiaan/projects/cobot-platform/scripts。
统一公开网页8015，内部loopback模型8020、Session8026、CPU actor8021。新模型manifest gate逐请求核验cohort及offline_validated；普通录制仍可用，未验证模型不可现场arm/start/resume/next。原CAN/机械臂/相机launch、home/recover/teleop功能不修改。

30Hz命令条件化只开放右前臂6关节；左臂/夹爪固定测量位置。alpha0.24963、每步0.006667rad等价旧20Hz alpha.35/step.01的时间常数/0.2rad/s限速。固定的6条prefix为已条件化命令，actor/explore不能修改；暂停/HIL使generation失效、丢弃旧chunk，fresh replan才能继续。deadline、stale或tracking冲突暂停而不直接杀Session。左相机固定视角与专家初始位姿一致，终局使用 all --pose plug，不使用origin改变视角。归位仍是现场显式终局动作，agent未调用。

网页暂停后进入HIL须切为HIL状态并采入训练；release后保持暂停，暂停等待帧无训练资格。终局成功reward落在最后有效控制帧，操作员等待不成为训练样本。no-record只保存小型终局统计，不保存图像/正式replay；shadow亦无正式replay、无publisher及归位。

RLT控制Py310不导入Torch/PyArrow，不占GPU；JAX模型Py311/ROS控制/CPU Torch actor服务/GPU learner分进程。loopback设置NO_PROXY，避免Cobot代理截走模型连接。

RLT连续录制暂存平铺HDF，后台官方LeRobot转换后完整核验qpos/action、全视频帧数、RGB采样质量、全部非图像事实及SHA，成功才删该原件；以UUID目录发布LeRobot，不覆盖已有记录。暂停/无效/不同generation片段不拼接。放弃样本归档小型事实再删原图像HDF，不训练。保留rlt/labels元数据用于防止序号复用；网页条数从保留摘要/身份读取，不依赖raw是否存在。rlt_down等待注册转换进程结束，避免中途关机丢数据。

## RL warmup 与在线更新

仅手动 ./scripts/warmup.sh；不按transition阈值自动启动。至少3条新有效成功和3条失败，建议先收20条并人工查看质量；每类留出原UUID，训练/验证分配一次冻结。专家30%、新成功40%、失败30%，先验证样本来源/掩码/动作，再训练，不混旧cohort。

冻结Stage1 VLA/token encoder/decoder，训练Torch actor及twin critic。z2048/context99包含state14+6条prefix相对state84+delay1；C10。actor/critic两层256，batch128/LR1e-4/tau.005/gamma.99，critic:actor2:1/ref embedding dropout.5/梯度裁剪1；最多20000 critic及10000 actor updates，多检查点按固定留出BC/QTD/平滑及限速选优。

Cobot adaptation明确区别于论文全动作actor：零初始化右臂有界残差±.05rad，BC/Q/smooth权重10/.1/10（BC及新增残差平滑按.05rad归一化，gate仍用物理rad），critic/BC均排除终局后padding；训练中使用与实际命令一致的可微条件化。失败不作为成功轨迹模仿，成功/HIL模仿实测动作；失败提供Bellman负目标及reference正则。此选择待真机评测，不宣称论文严格全参数复现或一定提高成功率。

最新在线模式每5条已验证新episode，在暂停/轮间执行UTD5×实际新增eligible训练transition（stride2，单批上限20000），保留optimizer/target网络；学习期间共享fcntl lease拒绝start/resume/next。候选没过留出/平滑门保留旧actor；被拒数据仍在replay。每episode开始才切actor，轮内版本不变；固定warmup/eval/no-record不在线更新。更新快不等于逐步学习，必须固定场景独立评测自主成功、HIL和失败，避免把辅助成功计自主成功。

4090仅优化器基准模型预加载并存时200critic/100actor为1.407s，推算20000优化约141s；该值不包括全部图像特征准备、转换和验证。首次专家特征可花数分钟，缓存绑定具体Stage1 checkpoint，后续只补新增样本。尚无新warmup actor，不用旧版本顶替。

## 验证与剩余验收

Cobot runs/plug_v2/tests 下：
- runtime_learning_validation.json：100实际CPU更新、actor隔步更新、成功失败Q学习、可微/实际条件化一致、heldout指标有限、暂停/HIL掩码；无真机publisher。
- conversion_validation.json：官方reader成功/失败、暂停掩码、raw验证删除/幂等、abort事实归档；真实模型feature replay的HIL/generation/terminal/future masks。
- actor_service_validation.json：fixture actor零初始化、按SHA不可变旧版本/新版本选择、错误JSON；没有生产warmup文件。
- session_e2e_validation.json：真实模型+真实Runtime/HTTP+不可创建ROS publisher的FakeIO，暂停立即停执行、暂停后HIL、release保持暂停、学习lease阻止resume、成功/失败/放弃及EndSession。无正式replay及归位。
- lifecycle_validation.json：注册PID/start ticks进程组准确down、模型显存释放，不触碰机器人服务。
- 平台16项API/mode/proxy/flat native HDF/storage回归与6项Node网页行为通过。

最终模型部署/网页验收已通过，动态服务与限制见下方收尾。现场模型效果、机械臂实测跟踪和新warmup actor验收尚未完成。

论文与固定仓库：[RLT](https://arxiv.org/html/2604.23073v1)、[官方实现](https://github.com/Yyshadow/openpi-RLT)、[训练时RTC](https://arxiv.org/html/2512.05964v1)。


## 最终交付收尾（2026-09-18）

- BF16/3999共34文件、6,508,405,454 bytes（不含checksum清单自身）；HPC→relay→Cobot全文件SHA一致。Cobot唯一manifest已offline_validated，onsite_validated=false，代码SHA与测试来源见execution_sources.json。不能把offline-validated称为现场成功。
- 实际三路480×640 RGB的留出专家观测请求20次：mean94.6ms/P95 100.5ms/max131.9ms，delay6 hard prefix完全一致，固定reference/token逐位可复现。实际Runtime/FakeIO Session用最终3999再次完整通过，包含bounded 3s inference/5s handshake/1s close；失联不无限阻塞退出。
- 最终模型down→restart恢复正确3999，模型仍预加载：supervisor455731/model455734（start ticks以backend/processes.json为准）。只有模型和统一网页常驻；没有Session/actor/cycle/真机publisher，无ROS master。结束条件为用户rlt_down/ui_down，网页在ROS未启动时不开放现场arm。
- 平台16 API/mode/loopback reentry/storage/nativeHDF回归；43 ROS subscriber/cache/reconnect检查；6既有Node检查及native readiness/HIL/learning gates通过。网页公共HTTP显示plug_v2/model4k/82专家，normal↔rlt可切换。当前computer-use没有可用浏览器，未执行浏览器视觉截图；HTTP/DOM/Node验证通过。
- 实际隔离fixture 100warmup→10online优化、accepted发布、optimizer/target恢复及100→110版本通过；无生产warmup文件。online batch fixture验证4不训/5触发/UTD只算new train/重启不重消费/并发开始defer不消费，6项通过。批次拒绝不意味着数据删除，真实学习效果仍需现场新数据和固定评测。
- 转换幂等会重新校验已有全部输出SHA；损坏输出绝不删除原HDF。成功/失败官方读取、实际模型feature/HIL/终局/future mask以及abort清理通过，所有fixture清除，没有伪造新cohort生产rollout。
- 4090与已加载JAX模型并存的RL GPU基准batch128/200critic/100actor为1.407s，max allocated66.24MiB，无OOM；这是优化器基准，不含特征准备/采样/验证。
- 仅清理本轮创建的临时Cobot1999 FP32部署副本15.428GB及relay1999/BF16中转21.937GB，总约37.36GB；HPC原训练checkpoint、选中Cobot3999、校验/日志/Tokenizer保留，不删除其他项目。
- 新用户流程见PLUG_V2_ROLLOUT_SOP.md：ui_up→rlt_up --reference→现场arm/start→新数据→EndSession（保留模型）→人工warmup→frozen比较→latest每5条更新→rlt_down→ui_down。不用旧interface4000或旧actor_7160。没有commit/push。

## 最终框架检查

执行源 SHA 与六个 shell 入口校验通过；git diff --check 通过；索引已重建；workspace doctor 0 error，仅 7 条既有 legacy attachment warning。最后一次学习发布/优化器恢复测试通过，未生成正式 warmup actor，未执行机器人动作。
