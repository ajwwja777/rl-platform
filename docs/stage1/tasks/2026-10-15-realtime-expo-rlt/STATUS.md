# HPC训练状态

2026-10-15 11:33:43 CST 已提交且RUNNING：Slurm 720997，节点ACD1-33，acd_u/formal-user，4GPU/48CPU/384GB，最长12小时（到23:33:43）。真实GPU型号及训练状态由输出复查，不以分配成功表示训练通过。代码0bfb297，固定上游8cef77e。

HPC根：/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/realtime-expo-rlt-20261015

- 入口 hpc-job.sbatch；设备检查device-check.log，状态job-status.json，Slurm日志slurm-720997.log。
- smoke/train.log、smoke/resume.log、smoke/verified.json：2步训练及另一个进程恢复到第3步；不声称数据顺序精确断点等价。
- 正式stage1/train.log；checkpoint在stage1/checkpoints/rlt_realtime_expo_left_2cam_10hz/s42/，5000更新，每1000保存，最终4999。
- 真实输入：TRAIN44/5331帧、DEV5/591帧；H50、10Hz、双腕224RGB、左臂6关节delta+可变绝对夹爪，batch32/FSDP4、seed42。
- 任一阶段失败停止后续步骤，batch退出自动释放。未完成训练/DEV评估/Warmup/部署适配前不可宣称可Online。
- A6000回执：projects/rl-platform/outputs/realtime-expo-rlt-20261015/hpc-status.json、hpc-submission.txt、hpc-verify-assets.log。base20文件SHA全部匹配，未重新传输12.4GB权重；不改旧HPC环境。
- 当前无需重新申请作业；先查720997和job-status.json。A6000/Cobot GPU、现场服务和生产资产不改。
