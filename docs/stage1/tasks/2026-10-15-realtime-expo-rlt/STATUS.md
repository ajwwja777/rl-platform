# HPC训练状态

2026-10-15 恢复核对：本任务仍未交付新Online权重。当前4卡正式重试721065排队；短训评测721070失败，正在修复评测一致性问题。不能把作业分配或单元测试通过视为训练/能力通过。

HPC根：/data/user/jhe724/jiaan/research-workspace/projects/rl-platform/outputs/realtime-expo-rlt-20261015

- 已验证：720997在4×H100上完成2步训练、checkpoint保存及独立进程恢复到第3步。smoke/verified.json；不声称精确数据顺序恢复。
- 失败：720997正式训练初始化遇NCCL/CUDA error999，于11:46退出，训练尚未产生正式checkpoint。stage1/train.log与slurm-720997.log保留。
- 待运行：721065，acd_u，4GPU/48CPU/384GB，最长4小时，排除首次失败节点ACD1-33。hpc-job-r2.sbatch先做三次独立进程collective检查，再从同一base、batch32/FSDP4训练；输出stage1-run2，复用smoke编译缓存。是否启动以squeue/job-status-r2.json为准。
- 失败：721070在独立debug单GPU上恢复短训checkpoint2；单条/批量8推理物理动作一致性门槛未通过，未生成模型选择结论。eval-smoke-721070.log保留。评测修复将保存差值并退回部署一致的单条推理，不放宽门槛。
- 实际输入：TRAIN44/5331帧，重复DEV5/591帧；H50、10Hz、双腕224RGB、左臂6关节delta+可变绝对夹爪，seed42。DEV不是独立TEST。
- 选型：比较1000/2000/3000/4000/4999，完整Episode统计；尚未完成正式训练或选型。
- 左臂双腕/10Hz执行适配仍在独立worktree，相关82项CPU测试通过；RTC模型路径、Warmup、Web/现场交付未完成。
- 用户授权HPC4/8卡；不抢A6000/Cobot GPU、不改变现场服务或机器人。不得重复提交已排队721065。
