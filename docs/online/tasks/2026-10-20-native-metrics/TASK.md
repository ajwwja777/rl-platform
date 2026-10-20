# 原生 EXPO 指标网页观察器

用户要求直接复用作者计算逻辑、在网页查看Online/Critic/Actor指标，保留HPC进程。代码事实：online_learner/async_learning直接保留updater返回的标量metrics；原网页只归档q/target/critic_loss。新增scripts/expo_metrics_observer.py仅GET本机deployment/status，原值写SQLite并发布有限JSON历史，不重算loss/Q/候选比例，不import learner、不发控制请求。每2秒观察，可能漏过中间更新，不宣称覆盖组内20个梯度步。未知历史不补造；以model/run目录/learnerPID/d/edit隔离，重启分段，防止回滚混写；SQLite保存所有已观察点，网页每段最近600点。

2个CPU检查覆盖去重、恢复、身份隔离和空指标。Web7项DOM检查覆盖原值/轴/无伪造/参数隔离/GET-only及原四图回归。UI新增基础Actor loss、Edit loss/Q/entropy/temperature、Filter loss/Q、候选Edit采用率、幅度、梯度范数、耗时和全指标表；候选比例明确为训练batch，不是实际执行比例。

W&B：用户团队3082928125-hkust-gz--org，项目暂定expo-online。现有Cobot凭据属于其他账号，没有目标团队；未使用上传、未覆盖。用户随后明确先完成网页，W&B稍后配置；本次不启动W&B上传，不把预留元数据当作接通。

资源：仅Cobot单个CPU观察器、本地SQLite和静态JSON。HPC/Actor/Session/后端不重启，模型和训练源码/配置不改。现场进程1357120与HPC Learner3181876/job740494作为发布前基线，实际验收再核对。

现场验收：RL1f6fee8/Web1442dd4已推送并SHA安装；static HTTP返回新版本及schema1指标。Cobot观察器1427342运行，读取当前异步实验run18eafdadb8ebc3f4e8e7。Runtime1357120/Learner3181876/job740494身份保持，Sessioninactive；Critic400/Actor20/Replay5152。尚无新更新metrics，完整指标点0，未运行训练来造验收数据。证据Cobot rl-platform/runtime/verification/native-metrics-20261020/acceptance.json。W&B按用户要求延期，无上传。
