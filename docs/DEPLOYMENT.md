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

## Stage1 加载优化（2026-09-29）

入口和模型目录不变。网页下一次加载RLT自动使用新实现；已有Stage1进程继续复用，不自动重启。in_the_pot π0.5 的代码、环境、权重和进程不在本批范围。

serve_stage1.py只建立VLA和RL-token encoder的抽象结构，不再先生成encoder/decoder随机参数再覆盖。stage1_loading.py通过固定Orbax 0.11.13的选择性恢复，只读取实际执行的VLA/encoder；训练decoder不参与当前Stage1推理，不初始化、不读取。原始checkpoint、归一化、BF16推理精度、seed42、10步去噪、动作映射和在线Learner/Replay均保持。

新日志MODEL_LOAD_TIMING分别记录imports、config_and_transforms、abstract_structure、checkpoint_restore、bind_inference；原MODEL_VALIDATE/READY保留。outputs/rlt/plug_v3_yyshadow/model-server/validation.json增加load_seconds，原inference_ms仍区分首次编译/验证和后续执行，不能把端口监听当作模型就绪。

编译缓存默认 /home/agilex/jiaan/project/rl-platform/runtime/cache/jax/stage1/，可通过COBOT_RLT_STAGE1_CACHE覆盖。保留可复用编译结果；JAX按版本、设备与计算图区分缓存，首次GPU加载仍需编译。缓存不含权重，不移到Getea1数据/模型目录。

Stage1默认XLA_PYTHON_CLIENT_PREALLOCATE=false，取消启动即预占72%显存。实际峰值仍需目标GPU验收，不能承诺显存降至某个数值。如要从终端恢复原分配策略：

~~~bash
cd /home/agilex/jiaan/project/rl-platform
COBOT_RLT_STAGE1_PREALLOCATE=true ./scripts/rlt_up.sh online
~~~

只在没有其他模型/Session、现场允许加载时运行；不同时在网页和终端启动。COBOT_RLT_STAGE1_MEMORY_FRACTION默认.72；它是分配配置，不是模型参数量。

离线复现优先在A6000有足够空闲内存时进行，比较完整checkpoint与推理子集需要约20GB以上额外主机内存。脚本强制CPU，不连接ROS、不更新权重/Replay：

~~~bash
cd /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform
stage1_env=/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/runtime-restore-verification/envs
PYTHONPATH="$stage1_env/machine-a-py311-overlay" \
  "$stage1_env/stage1/bin/python" scripts/validate_stage1_loading.py \
  --checkpoint models/rlt/plug_v3_yyshadow/stage1/4999 \
  --report outputs/startup-optimization-20260929/parameter-equivalence.json
~~~

该env是本批按冻结材料恢复的验证环境，换机依照上文restore_runtime建立自己的环境，不把scratch路径当作生产依赖。验证检查保留参数和真实encoder输出逐值一致；完整固定输入动作对比、缓存命中及时间记录见MIGRATION.md。A6000 CPU结果不能替代Cobot GPU冷启动实测。
