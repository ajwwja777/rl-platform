# RLT部署材料与命令

结构见ARCHITECTURE.md，完整流程见RUNBOOK.md。主仓库 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform；现场 /home/agilex/jiaan/project/rl-platform；GitHub https://github.com/ajwwja777/rl-platform。

Git之外需要固定上游、两套环境、tokenizer、Stage1和warmup/online资产。configs/assets及configs/rlt/plug_v3_yyshadow/manifest.json登记位置。A6000本项目outputs/environments/cobot-runtime-20260928.tar.gz是冻结环境：7,485,824,462字节，SHA256 214403a0eeefa5f31bbc7505808fe93ec344d5e656ff4b6d815d92eae7445f2b。它不包含系统ROS/CUDA驱动，解压不代表已验收。

configs/environments/rlt-{online,stage1}.json记录包版本和direct_url；online Python3.10.20，stage1 Python3.11.14。源码使用third_party/openpi-rlt与integrations/stage1-client，不恢复指向旧已删目录的editable依赖。两环境保持独立，不升级。

```bash
cd /home/agilex/jiaan/project/rl-platform
envs/online/bin/python scripts/preflight.py
./scripts/rlt_status.sh
# 现场确认后手动启动，等待操作者开始Episode：
./scripts/rlt_up.sh online
# Ctrl-C停止前台Session；确认Session已停再释放Stage1：
./scripts/rlt_down.sh
```

推荐使用网页或cobot-web/scripts/models.py共享加载，避免同时再起一套rlt_up。硬件顺序CAN→ROS→机械臂→相机；control diagnose通过后选择采集目录、加载online、等Session就绪，再手动开始。成功/失败携带当前episode_id/generation；HTTP503先看Session fault_reason，不盲目重试。

scripts/validate_online_resume.py使用私有临时副本，不连接机器人；已有证据检查5000→5005、Actor2500→2502和源文件哈希不变。本批不往生产Replay注入数据、不更新生产权重。A6000 .venv是轻量测试环境，不等于Stage1完整GPU环境；测试需按环境分别执行。

机器配置见configs/machine.example.json，现场local.json不入Git。数据/权重仍在Getea1/jiaan/{data,model}，代码/环境/日志/PID在本项目。主仓库先提交push、核验，再同步现场；guide Git不由本项目处理。
