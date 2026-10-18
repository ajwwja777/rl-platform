# EXPO HPC 四卡 Online：执行事实

截至 2026-10-18 22:52:20 CST：四卡更新、恢复、Cobot新Actor推理与完整正式HTTP回传验收通过；按用户要求取消作业734617，四卡已释放（Slurm CANCELLED）。可复用恢复入口实际重启生产98并再次通过通信检查。没有运行机器人或真实rollout。

## 资源与生产状态

- 本次Slurm作业734617，acd_u，ACD1-20，4张NVIDIA H100 80GB HBM3、48 CPU、384GB；21:36:13分配，原12小时截止2026-10-19 09:36:13。已于2026-10-18 22:52:20主动取消，实际占用1:16:07；另两个PSI0预约仍运行且未触碰。回执release-receipt.json。
- 生产起点保持Online/Actor/VLA98、Critic1960、完成27回合、在线4523条transition、Replay7824、credit8、批次0/5。25条成功TRAIN示教、优化器、目标网络和RNG完整保留；没有从SFT重新训练。
- 6343文件/18,371,635,039字节跨机迁移逐文件大小与SHA256一致。HPC生产副本与验证副本仅重绑bundle绝对路径及经审查适配代码身份，算法预算不变；源LFT2生产元数据与contract未改。
- 保留5个有效回合触发、global batch64、每30条transition增加一组20 Critic+1 VLA更新、RTC5/Edit0.01、夹爪反馈执行逻辑。四卡使用原data-parallel mesh，不减少原生更新次数。
- 验证Actor100/102来自隔离合成回合，不作为真实生产权重发布。生产仍为98。

## 已完成验证

- EXPO CPU回归246项通过；解释器修复后32项针对性回归通过。真实JAX四H100 GPU backend、4设备分片collective通过。
- 通过真实Learner HTTP接口提交隔离成功/失败回合，包含暂停、HIL切换、重复上传及重复结束。3/5不更新，重启保留3/5；第4回合不更新，第5回合触发2组更新：Critic1960→2000、VLA/Actor98→100。检查点保存与新进程恢复后版本、累计数、Actor manifest一致。
- 更新后四卡检查点在双卡恢复通过；HPC单卡Actor100加载/推理通过。单卡测试没有Replay或ROS发布。
- Actor100完整回传A6000与Cobot，逐文件SHA一致；Cobot实际GPU加载/预热37.51秒，样本推理0.225秒，动作8×7且全部有限。原Actor98此前独立验证69.63秒加载、0.237秒推理。均不代表机器人成功率或持续50Hz验收。
- Cobot→A6000→HPC登录节点→计算节点的实际HTTP状态、manifest、Range内容核对通过。12次状态RPC约6.1–10.7毫秒；断开网关后Learner仍ready，独立生命周期成立。
- 缓存测速另在隔离验证副本运行100→102，Critic2000→2040。稳定一组20 Critic+1 VLA间隔60.497秒；仅一个稳定间隔，不是吞吐分布。原两组冷启动含编译252秒、198秒；未声称四卡线性提速。
- 新的可复用恢复入口scripts/resume_expo_hpc_learner.py已提交8184004，并在HPC逐文件SHA核对。接收新的作业号，核验当前用户/单节点/四GPU分配，重绑Learner与网关job_id，恢复现有生产目录；不自动预约、不开启Web资格、不运行机器人。实际恢复验收见resume-acceptance.json。

## 时间估算

历史27个真实回合：rollout中位17.88秒，P10–P90约14.95–26.39秒；长暂停会拉高均值，不用均值作为正常执行时间。每回合有效transition中位159，最近5回合共942条。

按当前credit8、60.497秒/更新组及历史数据，5回合约26–31组更新。完整正式RemoteSupport.download_actor经全部SSH/HTTP链路下载2,017,658,362字节，含逐文件SHA与落盘实测176.05秒；此前分段rsync/SHA测试197.36秒。保存/发布36.36秒，Cobot独立加载37.51秒。按历史中位回合、平均步数、最近5回合分别估算整批31.87、33.89、37.14分钟，交付按约30–40分钟/5回合、平均6–8分钟/回合预留；前4回合不等待学习，第5回合结束集中等待。正式回执http-return-acceptance.json及timing-estimate.json。

估算不含首次服务加载/编译、人工摆场和机械臂归位；依据一次稳定更新间隔与历史数据，不是SLA或成功率结论。

## Web及保留资产

- 可选入口：plug_left_2cam / EXPO / realtime-expo-ft-edit001-hpc4 / 2000；ID expo-plug-left-2cam-edit001-hpc4-2000。2000是SFT底座步数，真实继续学习版本为Online98。
- 新入口沿用rollout/HIL、暂停、成功/失败标注等已有接口；本次验证协议级生命周期，真实Web加载后的物理rollout/HIL仍由现场操作验收。未自动启动Session、录制、机器人或重载Web。
- 用户要求本次释放，故不创建Cobot runtime/expo-remote/hpc4-ready.json；没有活动预约时入口不可用属于预期。下次新作业需恢复Learner并核对端到端状态，才能生成与该作业对应的资格回执。
- 原LFT2 edit001/batch5 profiles固定旧适配代码快照，签名仍6c8a13201630a2793cc544e117f406d73665e49af971f6b7464003abd9094fd2；原模式和Replay可继续使用。
- HPC独立依赖补充层与编译缓存保留，未修改共享环境。无需重新传输18.37GB输入或重新采集/标注数据，加载时由持久化记录恢复内存Replay。HPC是原98的独立迁移分支；以后若LFT2继续学习，两端不会自动合并，恢复入口读取HPC自身最新committed状态。

## 证据与位置

A6000：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/expo-hpc-online-20261018。

HPC：/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/expo-hpc-online-20261018。

Cobot：/home/agilex/jiaan/project/rl-platform/runtime/verification/expo-hpc-online-20261018。

下次接续的资产/配置/恢复入口索引保存在handoff.json；原orchestrate.py、deliver.py及finish-and-release.py是本作业验收脚本，不作为下一作业的启动入口。

主要回执：asset-transfer-receipt.json、qualification.json、delivery-receipt.json、warm-benchmark-result.json、timing-estimate.json、resume-source-revision.json、resume-acceptance.json、http-return-acceptance.json、release-receipt.json。以上回执已实际生成并核对。HPC生产状态目录online-edit001-hpc4与Cobot真实生产cache独立于verification。

源码9821542/9426151为四卡与Python解释器支持；652771e/8184004为可复用恢复入口；cb9f0e3更新已通过验收的目录说明，已SHA安装Cobot且未重启Web。

## 本次故障与修复边界

21:36首次启动读取计算节点/etc/ssh/ssh_host_ed25519_key.pub权限不足，训练尚未开始。已改为优先使用已有可信known_hosts并保持StrictHostKeyChecking=yes；可复用恢复入口对新节点通过本作业srun在计算节点loopback获取SSH公钥，本次节点已有可信记录，未触发新节点分支；未降低严格SSH校验，未修改系统SSH设置或其他作业。旧HPC/usr/bin/python3为3.6，网关和监控已改用指定Python3.11。
