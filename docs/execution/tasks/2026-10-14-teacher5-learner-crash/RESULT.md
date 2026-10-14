# Teacher5 Online 恢复结果（2026-10-14）

**已验证：正式Teacher5已恢复ready/disarmed/paused，允许操作员继续受控Online采集。** 已完成首轮真实学习、持久保存、重载和Actor服务检查；没有自动发布候选或机器人动作。长期连续rollout仍待真实采集验证。

## 故障与解决

正式恢复时捕获SIGSEGV位于JAX compiler.backend_compile → supported_hil_runtime.train → Learner.train_once，故障后EnvDriver才被supervisor终止。三路相机流持续正常。相同数据的独立GPU训练/完整Learner可运行，但正式多进程环境会崩溃；**底层GPU编译触发条件仍证据不足**，没有声称已修复CUDA或驱动。

提交6867ac6将仅plug-v3-teacher5-online的Learner平台设为cpu，在fresh learner child、初始化JAX后端前生效。其他角色/旧模型继承原后端；Stage1和Actor仍使用GPU。训练数学、学习率、采样、预算、profile身份、执行50Hz/RTC/单级EMA保持；不是冻结或禁用学习。来源9a21f44的错误归因/faulthandler修复也已部署。4个恢复文件逐SHA核验安装，之前2个取证文件亦核验。

## 实测验收

| 项目 | 结果 |
|---|---|
| 代码回归 | 已验证：19项通过，shell语法通过 |
| 隔离完整CPU Learner | 已验证：18更新、保存、独立新进程恢复8145，预算0；生产资产保持 |
| 正式真实学习 | 已验证：8127→8145，9次Actor更新，18条batch回执严格8128…8145且不重复 |
| 正式重载 | 已验证：正常停止保存后重载，checkpoint8145、Learner8145、pending4072、预算0 |
| Replay | 已验证：3641条；原Episode10040的18条失败经验保留，SHA保持 |
| 实际执行权重 | 已验证：served Actor4063及SHA保持，4072仅pending，不自动晋升 |
| Actor真实RPC | 已验证：返回10×7有限动作，version4063，单次约9.2ms；不进入执行队列 |
| 相机 | 已验证：恢复后5秒三路149/150/150帧，29.88–29.91Hz，最大间隔39.1ms |
| 执行状态 | 已验证：ready、disarmed、policy_paused、emitted_commands=0；50Hz/RTC/EMA tau0.08 |
| GPU隔离 | 已验证：GPU compute列表仅Stage1 PID122499及Actor PID176177；Learner PID176074不占GPU |
| 真机任务效果 | 证据不足：本轮未运动，未证明成功率改善或长时间无故障 |

现场最终runtime PID175848、Learner176074、Actor176177是该次回执，不代表未来PID。/proc/PID/environ仅代表exec时环境，不能拿其中初始CUDA_VISIBLE_DEVICES=0否认子进程随后设置的CPU后端；模型配置、fresh-worker设置、运行日志和GPU进程列表联合核对。

## 身份和证据

Cobot runtime：/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/
录制目录：/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/teacher5-8127-online-v1/
当前正式日志：/home/agilex/jiaan/project/cobot-web/runtime/deployment/model-20261014T204634.log
恢复前checkpoint/profile/配置及源文件备份：/home/agilex/jiaan/project/rl-platform/outputs/teacher5-learner-recovery-20261014/before-recovery/、before-cpu-fix/

主要证据（A6000）：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/teacher5-camera-fault-20261014/
- formal-recovery-first.log：正式GPU compiler SIGSEGV完整栈；取证修复实际生效。
- native_gpu_repro_v2.log：独立完整GPU Learner18更新通过；不替代正式多进程环境。首次限2CPU GPU探针65秒超时，另有路径fixture缺文件失败，均保留原记录而不算通过。
- native_cpu_recovery.log / native_cpu_resume.log：完整CPU路径、保存及新进程恢复；非仅train_step指标。
- cpu-backend-regression.txt / cpu-backend-install.json：19回归及4文件同步SHA。
- cpu-formal-recovery-status.log / cpu-persistence-restart.json：实际更新及保存重载。
- recovery_acceptance.py / recovery_acceptance.log：正式状态、完整checkpoint、ActorRPC、保护SHA和GPU占用。最初脚本空字符及exec时环境断言问题已修正，最终执行成功。
- actual-update-batches.json / recovered-camera-5sec.json：真实18批身份及恢复后图像时序。

Replay SHA256：dabf45376c658248415d31ed44cccbcb44d4ab8afb6a5a19b8145766541e4dc1。
served Actor SHA256：8a59565d2ba4538d13df2d49a047b8482707126dd38dacfe34a4e55ce4958f9f。

## 使用与回退边界

沿用网页数据采集的supported-mc30-hil8-teacher5／Online入口，不换权重，不自动晋升4072。下一项是用户首轮恢复后的真实rollout及后续新预算核对。CPU是本次运行故障的已验证规避方案；GPU编译根因独立待查，未经验证不切回。代码回退只针对6867ac6的后端选择，不能回滚已合法推进的checkpoint而重复消费18条数据。未改EXPO入口/资产/进程；另一对话保留其远端训练工作。
