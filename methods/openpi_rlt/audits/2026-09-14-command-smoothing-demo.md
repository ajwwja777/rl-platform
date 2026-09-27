# 2026-09-14 两组 demo 指令平滑交付

已部署至 Cobot；44 项 relay 离线回归通过，Cobot 实际运行环境新增 9 项测试通过。未启动机器人动作，真机顺滑程度与成功率待现场验收。

## 两组入口
在 /home/agilex/cobot_magic/task3/jiaan/realworld_rl/openpi_rlt_plug 中：
- reference：./interface_task2_teach_rlt_smooth_reference.sh
- 固定 warmup：./interface_task2_teach_rlt_smooth_warmup.sh

入口已固定 Stage1 step_4000，不再另传 4000。默认 eval；两组均无训练更新。固定组使用独立 runs/plug-fixed-warmup-smooth-demo-v1，原始 step_10000 checkpoint / actor 5000，校验 actor SHA 与已有 pointer；不会跟随正式在线 run 的 accepted actor。
reference 指 Stage1 联合训练模型的 reference 路径，不是另行训练的纯 SFT 基线。

## 算法与边界
保留旧测量跟踪误差限幅，之后以之前最终发布指令为锚点 EMA alpha=0.35，再投影到发布步长 ±0.01 rad 与 measured tracking bounds 的交集。交集为空时拒绝发布，要求暂停重定位。20 Hz 名义控制频率；不是硬实时或严格 jerk 约束。
暂停、HIL、终局清除历史；普通 chunk 切换保持历史。被动臂和夹爪保持旧语义。trace executed action 记录最终处理动作。平滑可能增加跟随延迟，不能保证插孔成功率。
与 XR-1 同类指令滤波，但没有复制异步 RTC/prefix 模块。为了公平，两组使用同样初态、物体位置、参数与无探索设置。

## 验证和回退
44 项：conditioning、runtime、env、session HTTP；新增 9 项覆盖步长、跟踪冲突、暂停重定位、HIL、chunk 历史与参数检查。
Cobot 上验证固定 actor 文件 hash，执行离线测试和 bash -n；没有启动 LIVE/SHADOW 服务，没有生成正式 replay 或学习更新。
备份：/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/deploy-backup-smoothing-20260914。
回退平滑：结束 Session，等待终端退出，再用原 interface_task2_teach_rlt_live.sh 4000 或 online.sh 4000 --frozen-actor。新 wrapper 的环境只作用于子进程；旧入口默认禁用新增滤波。注意旧 online 入口冻结的是正式在线 run 当前 actor。
Machine A port 8000 PID 992802 核验仍常驻；8016/9101/9102 发布前空闲。网页经 VS Code 转发出现 400/501 的问题按用户要求暂缓，未宣称修复。
