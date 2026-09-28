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

## 隔离恢复冻结环境

Git获取代码时使用 --recurse-submodules，确保 third_party/openpi-rlt 固定版本完整。先安装系统GPU驱动及control说明中的ROS依赖，再从A6000取得登记归档；不需要访问原Cobot寻找Python包。

```bash
cd /home/agilex/jiaan/project/rl-platform
# 仅限新机器、envs不存在或为空；不会覆盖现有环境：
python3 scripts/restore_runtime.py --materials outputs/environments --destination envs
# 之后只读复核：
python3 scripts/restore_runtime.py --destination envs --verify-only
```

恢复入口核验SHA256，修正Stage1解释器与editable源码路径，按隔离CPU模式导入在线JAX/Flax和Stage1 Torch/OpenPI。使用 envs/.../bin/python -m 调用包；冻结环境旧console-script shebang不作为换机入口。machine-a-py311-overlay保持独立，不混装两套Python。./scripts/install.sh只安装开发测试.venv，不替换部署环境。

2026-09-29 在A6000 /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/runtime-restore-verification/envs 完成独立恢复：在线Python3.10.20、JAX0.5.3/Flax0.10.2/Optax0.2.8；Stage1 Python3.11.14、Torch2.7.1+cu126、OpenPI模型/训练配置可导入，35项Stage1配置/动作/冻结契约测试通过。另有78项在线Session/录制/收尾契约通过。没有读取原Cobot文件、加载生产权重或启动机器人。

configs/assets/runtime_environment.json区分完整冻结运行材料与包清单。尚需在目标GPU验证权重加载和推理；只监听8030不等于模型就绪，rlt_status分别报告listening、Session phase及model_ready。
