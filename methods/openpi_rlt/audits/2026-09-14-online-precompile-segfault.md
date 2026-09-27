# 2026-09-14 在线 learner 首次预编译崩溃排查

用户 17:37 启动 online 非冻结模式，PID 1293856 在 online_cycle_worker.py:75 -> trainer.train_once:531 -> JAX backend_compile 中 SIGSEGV。尚未启动机器人 env；不是 command smoothing 的执行阶段。没有认定显存不足或具体 CUDA 冲突。

隔离实验（Cobot scratch/startup-20260914）：
- online-compile-repro：独立 GPU worker，预编译 18.27 s 成功。
- online-ros-cache：加入 ROS 和正式 JAX cache，预编译 15.46 s 成功。
- online-aloha：加入 conda aloha，缓存命中预编译 1.85 s 成功。
- full-online-repro：部署环境、replay manager、learner 启动成功，截断于 actor/env 启动之前，预编译 1.85 s。
- online-transaction：复制正式 replay 的 episode 13（12 records）到隔离目录，32 updates + validation + 文件发布通过，耗时 1.3029 s；global_step 10032，actor 5016，accepted=true。没有发布到正式 actor 服务。
- 使用已有 warmup 和原始代码，没有修改 JAX/CUDA 依赖、没有清理共享缓存、没有替换正式模型或回放数据。
- 正式 current.json 仍为 actor 5000/global_step10000/transactions0；4 个历史 processed episodes。
- 测试创建的 worker 均清理；Machine A 保持常驻。

结论：当前缓存和完整启动流程验证可用，可重试原在线命令。原始 SIGSEGV 未复现，底层根因未确定，不宣称修复。1.3 s 是该隔离数据规模的更新/验证耗时，不包含录制固化和机器人归位，不保证未来每轮相同。若再次发生，应保留 PYTHONFAULTHANDLER 日志并捕获 native backtrace，避免盲目重试或禁用训练假装在线模式。
