# 2026-09-10：RLT 录制并发溢出修复与交付

## 问题及证据

现场 episode_000001/000002.hdf5.incomplete 均记录 writer_queue_overflow。第二轮 sampled=369、written=113，256帧队列填满，触发策略暂停；录制未完成，所以点击失败无法完成 Task5 finalize。训练没有据此进入 warmup，不能据此判断模型完整插孔效果。

真实 ROS 订阅、三路预览和 HTTP 单独诊断均没有复现。完整 Machine A + actor + learner + replay + shadow env + 三路 HTTP 预览并发下复现。所有诊断输出位于 Cobot RL scratch/task5-contract-20260910，禁止加入正式数据或 replay。

| 对照 | 结果 |
| --- | --- |
| 原部署完整并发 | 约13秒，384采集/128写入，溢出 |
| 仅16帧批量写入 | 约16秒，448/192，溢出；未部署此候选 |
| 仅移除控制步 trace fsync | 约14秒，411/155，溢出 |
| 仅将 learner 空闲轮询改为1秒 | 约48秒，1441/1185，溢出；未采用该配置改动 |
| 最终：trace终局同步 + learner状态写入节流 | 90秒失败、20秒成功、20秒放弃均正常收尾 |

主要瓶颈为共享 USB/NTFS 上的高频同步落盘竞争：trace 每控制步 fsync，learner 默认0.01秒轮询每次同步写入带新时间戳的状态文件。最终同时处理两处；单独优化 HDF5 不能根治。

## 最终实现

- cobot_adapter/cobot_ros1.py：pending trace 仍逐步追加，在 done=true 时同步整个文件后原子发布。未完成 trace 不是 replay 真值；断电可能丢失未终结 trace 尾部，但不会因此产生完整 episode。
- cobot_adapter/status_io.py 与 scripts/online_role.py：仅限制派生 learner 状态文件写入，通常至多1Hz；ready_for_online/暖启动阶段边界立即写入；正常进程退出刷新最新待写状态。训练更新、原0.01秒轮询、replay journal、checkpoint及actor snapshot持久化逻辑不变。普通展示指标可能延迟约1秒。
- scripts/rollout_recorder_app.py：uvicorn graceful shutdown 上限5秒，避免浏览器MJPEG长连接无限阻塞停止。
- 保留原部署的 all --pose plug home实现。框架源码中原来缺少这段已部署逻辑，此次同步来源以避免覆盖；没有执行或更改真机复位动作。
- 采样30Hz、图像尺寸、队列256、旧的dataset句柄缓存均保留。批量写入候选撤出 canonical source，失败版本保存在诊断目录作证据。
- 固定上游源码未修改。

## 完整无动作验收

目录：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/scratch/task5-contract-20260910/full-shadow-status-trace-v1-20260910

- 真实step_4000推理，explore影子模式，独立端口18016/18017/19101/19102和测试数据根。
- ROS master核验 /task2/policy/joint_* publisher为空；home_after_terminal=false。
- 测试learner warmup门槛设为10亿，replay_size=0、adds_total=0、global_step=0；未进行在线更新。正式门槛与训练配置未改变。
- 第一轮90秒、2714帧，失败；第二轮20秒、605帧，成功；第三轮20秒、604帧，放弃。三个HDF5均complete、有UUID，终局接口正常，进入waiting_scene，下一轮可正常启动。
- 最大队列积压分别25、12、10帧。成功/失败/放弃标签逐一读回匹配；放弃keep_for_training=false。
- 42项RLT状态/trace/session/ROS适配测试通过；93项Task5/API/HDF5/录制测试通过。
- 实际MJPEG长连接保持打开时，SIGTERM后5.41秒退出，证据在 shutdown-real-stream/result.json。
- 本次验证的是部署链路与并发可靠性，不是实体机器人插孔成功率，也没有验证现场home动作或真实在线学习效果。

## 部署及当前状态

部署：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-vla/deployments/openpi-rlt/plug-insertion-stage1-v2/runtime-overlay

精确备份：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/deploy-backup-task5-contract-20260910/status-trace-io-fix

部署清单：
methods/openpi_rlt/manifests/plug-task5-recorder-fix-20260910.json

全部诊断进程已退出。旧故障recorder PID3393383停止采集且publication revoked，SIGTERM后卡在等待浏览器连接，检查后仅对该进程SIGKILL。最终正式8017已重新启动并验证idle、ROS ready，日志service-20260910-200549.log。8000模型和8015保留；8016、9101、9102以及全部诊断端口空闲。未提交、未push，其他项目修改未处理。

用户操作仍为：
```bash
conda activate aloha
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
cd /home/agilex/cobot_magic/task3/jiaan/realworld_rl/openpi_rlt_plug
./interface_task2_teach_rlt_live.sh 4000 --explore
```

按Enter仅arm，打开8016并开始Session。8015查看平台相机；8017是连续录制后端。正常成功/失败由RLT契约决定进入replay，128有效transitions后进入既有warmup流程；本次没有伪造或预灌正式样本。
