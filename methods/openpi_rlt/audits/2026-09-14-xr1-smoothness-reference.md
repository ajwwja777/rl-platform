# 2026-09-14 XR-1 顺滑部署核对

用户要求以曾在Cobot现场感觉顺滑的XR-1部署为参考。仅只读代码和记录，未启动XR-1、未修改RLT部署、未操作机器人。

## 身份

用户给出的 https://github.com/Open-X-Humanoid/XR-1 是另一项目。A6000 proj-20260829-xiaomi-robotics-1/REMOTE.md 与Cobot源码README对应 https://github.com/XiaomiRobotics/Xiaomi-Robotics-1 ，固定commit cfcab04e662514644d62a4f3cfcce97ce83b90b2。不要把两套同名XR-1机制混为一谈。

## 实际Cobot实现

根 /home/agilex/cobot_magic/task3/jiaan/deployments/in_the_pot/xiaomi_robotics_1；dagger_round001入口参数同样核对。
- run_checkpoint.sh：20Hz，replan_remaining10，prefix6，min_async_queue6，starvation_floor2，arm_smoothing_alpha0.35。
- common/robot/inference_xr1_async.py：单后台推理线程与发布循环分离；队列余10步时请求下一chunk；HIL/pause使queue/future/generation失效；余2步仍未完成则暂停。
- common/robot/cobot_xr1/deployment.py：返回结果根据推理期间已消费步数丢弃过期前段，再安装余下动作。
- common/robot/cobot_ros.py:135、462：12关节EMA，q_smooth=q_prev+0.35*(q_target-q_prev)，previous为上一条发布指令；首次用实测初始化。随后limit_action_step，默认各关节0.01rad/发布步，夹爪单独处理。此EMA增加滞后，不等价于严格jerk限制。
- runtime源码README：异步训练默认开启，模型接受N×60的action_prefix；本地client把关节轨迹经FK转为前缀，再将模型输出做IK。不是直接14D关节输出。
- 实际顺滑收益来自上述各项的比例尚无消融证据，用户现场评价保留为定性反馈，不凭代码归因全部改善。

## RLT迁移建议

优先复用命令空间EMA、步长限制和暂停/HIL重锚定，作为reference、固定warmup、在线RLT共用执行层。alpha0.35/20Hz与0.01rad是对照候选，不直接宣布适合插孔。比较延迟、跟踪误差、成功率和抖动。
异步队列/前缀机制第二阶段独立适配：XR-1 horizon30而RLT actor chunk10，不能照抄10/6/2步阈值；也不能给没有prefix条件训练的MLP actor直接塞前缀。需按实测延迟、可用预测长度、actor版本固定及replay实际执行动作对齐重新设计。
当前没有迁移或部署新的平滑控制器。
