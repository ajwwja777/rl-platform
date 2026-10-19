# EXPO 每个 Critic minibatch 固定50%示教（2026-10-19）

用户授权将当前 seeded-buffer 改为固定示教配额。新增可选 critic_prior=demo50；64条小批次恰好32条示教、32条在线，1280条外层batch按20个小批次分别满足配额。在线池保留成功/失败；按重建时不可变示教前缀识别来源，不按is_hil/is_success冒充来源。基础π0.5成功样本采样、K30、UTD20、五回合调度保持。示教与在线之间不构造跨来源backup窗口。池缺失时报错，不静默回退为全示教。

实现：prior_sampling.py、online_learner.py、remote_launcher.py；Cobot remote_runtime_support.py校验远端采样身份。启动参数 --critic-prior demo50，运行contract固定策略及源码SHA。fork_expo_batch_run.py支持独立采样接续，保留partial batch/credit/Replay/RNG/优化器与权重，不覆盖旧运行。不可变checkpoint/数组复用，回合JSON独立复制。

验证：142 CPU单元/回归通过；实际原生Replay CPU gather与OpenPI转换、RNG复现通过。准备阶段仅作CPU验证；后续GPU启用结果见下文，未执行真机动作。现场原运行Actor121/Critic2420在首次预检时ready且机器人暂停，实际切换前重新取状态。

状态：已部署启用。用户授权从已保存Actor121/Critic2420重启本批；旧运行保留，新运行后缀--demo50。实际Learner已完成首组新配方训练，Actor122/Critic2440，采样统计每64条32示教+32在线，外层1280条。Replay9225步（示教3301、在线5924）。整批23组仍在执行，未声称新checkpoint已落盘。Cobot已加载并保持paused，Session inactive，无机器人动作。142项CPU测试及原生Replay检查通过；GPU训练已验证开始，训练收益尚未验证。证据：outputs/expo-demo50-20261019/activation-acceptance.json。

用户随后明确授权立即从Actor121重启本批，舍弃未保存梯度、保留所有回合。新增restart-pending接续：完整checkpoint恢复，保留credit/partial batch，未提交回合存pending-recovery备份，待Cobot恢复后通过原上传重试链重新提交，避免Learner初始化期间恢复训练阻塞Actor启动。142 CPU检查通过。
