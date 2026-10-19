# EXPO 每个 Critic minibatch 固定50%示教（2026-10-19）

用户授权将当前 seeded-buffer 改为固定示教配额。新增可选 critic_prior=demo50；64条小批次恰好32条示教、32条在线，1280条外层batch按20个小批次分别满足配额。在线池保留成功/失败；按重建时不可变示教前缀识别来源，不按is_hil/is_success冒充来源。基础π0.5成功样本采样、K30、UTD20、五回合调度保持。示教与在线之间不构造跨来源backup窗口。池缺失时报错，不静默回退为全示教。

实现：prior_sampling.py、online_learner.py、remote_launcher.py；Cobot remote_runtime_support.py校验远端采样身份。启动参数 --critic-prior demo50，运行contract固定策略及源码SHA。fork_expo_batch_run.py支持独立采样接续，保留partial batch/credit/Replay/RNG/优化器与权重，不覆盖旧运行。不可变checkpoint/数组复用，回合JSON独立复制。

验证：141 CPU单元/回归通过；实际原生Replay CPU gather与OpenPI转换、RNG复现通过。非GPU训练收益验证，尚未执行新配方梯度更新/真机动作。现场原运行Actor121/Critic2420在首次预检时ready且机器人暂停，实际切换前重新取状态。

状态：源码准备及同步；当前旧Learner尚未切换。已向用户请求结束空闲Session并暂停重载模型的明确授权，依据项目机器人推理启动边界。待授权后停止旧owner、核验完整checkpoint、建立独立 --demo50 接续、安装profile/运行路径并正式暂停加载验证。保留默认seeded回退，不混写历史实验。
