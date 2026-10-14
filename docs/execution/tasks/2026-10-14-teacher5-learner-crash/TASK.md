# Teacher5 首轮后无法继续推理：Learner SIGSEGV

2026-10-14。状态：直接故障已定位，底层崩溃原因及现场恢复未完成。不是已修复／可继续 Online 的交付。

## 当前证据

- **已验证**：首次新增 Replay Episode10040，18 transitions，全部success=0/rewards=0，最后done=true；Replay3623→3641。正式Learner仍8127、served Actor4063，pending budget18，不能称完成学习。
- **失败**：现场Learner在20:11:14、20:11:40、20:12:01、20:12:40、20:14:12连续SIGSEGV；最后一次PID123529，supervisor报告 learner_service exited with code -11，随后终止EnvDriver。ROS shut down before Cobot RLT was armed是该次清理中的下游错误。
- **已验证**：故障后8秒只读订阅三路RGB，各239帧、约29.9Hz，最大间隔38.7ms，640x480；此时不是持续相机断流。
- **证据不足**：相机启动初期front相机确曾等待枚举，20:07:24–26恢复；不能排除用户早先见过短暂相机异常。20:11–14的退出顺序明确为Learner先崩溃，不能以相机启动历史代替本轮直接故障。
- **已验证**：现场相同checkpoint、sampling_state、实际3641条journal、原生网络/训练函数，在隔离CPU内存完成18次更新（8128–8145，Actor9次），所有返回指标有限；生产checkpoint/Replay/served Actor SHA前后保持。首次约5.48s，后续约28–49ms。未导出候选或消耗正式预算。
- **证据不足**：CPU通过只缩小排查范围，不能证明现场GPU/完整服务路径通过。scripts/rlt_up.sh给Online启用CUDA0；现有日志没有native/Python故障堆栈，不能认定CUDA、显存、驱动或数据为SIGSEGV根因。

## 数据标签疑点澄清

原录制UUID 05917918-3065-4b07-9d1d-cc631d3f0eae，episode_000001，53帧、complete，outcome=failure。其quality=bad、keep_for_training=false由cobot-dagger capture_core/labels.py:set_outcome在点击failure时自动写入（兼容字段），不是单独的数据损坏判断。不据此删除有效RL失败样本；没有改Replay/标签。初步把该字段当独立质量判定的疑点已撤回。

## 隔离修改

基线8780b61；分支fix/teacher5-crash-diagnostics-20261014；worktree：
/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/teacher5-camera-fault-20261014/rl-platform

1. runtime_diagnostics保留明确OOM等原因，识别worker致命信号4/6/7/8/11，避免把ROS清理错误作为主因；不把正常SIGTERM视作native crash。
2. process_bootstrap在fresh worker初始化前开启faulthandler，后续native crash可在子进程stderr保留Python线程栈。此为取证，不是SIGSEGV修复；不改训练数学、配置或固定上游。
3. 10项定向/真实spawn回归通过，包括独立子进程SIGSEGV实际产生故障堆栈（禁用core，不启动ROS/GPU）、Replay生产/管理/恢复和旧精度路径。首次旧spawn回归因worktree缺少固定上游Python路径失败/超时；显式使用主仓库只读依赖路径后10项全通过，未改生产环境。详见spawn-regression-resolved-paths.txt。

## 复现与资产身份

证据根：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/teacher5-camera-fault-20261014/
- initial.json / hardware-kernel.json / camera-core.json：Web、服务日志、kernel退出顺序。
- camera-8sec.json：真实RGB订阅计时，非仅节点存在。
- config-recording.txt / new-replay-identity.jsonl：实际配置和新增行身份。
- cpu_repro.py、cpu_pending18.py及对应log/exit.json：现场online Python、CUDA_VISIBLE_DEVICES空、JAX_PLATFORMS=cpu、限2CPU，SSH stdin执行，不写现场资产。
- diagnostic-tests.txt / spawn-regression.txt：隔离worktree测试记录。

运行资产（Cobot）：
/home/agilex/jiaan/data/rlt/plug_insertion/history/candidates/teacher5_8127_runtime_20261013/
录制：
/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/recordings/rl-platform/rlt/teacher5-8127-online-v1/
Web故障日志：
/home/agilex/jiaan/project/cobot-web/runtime/deployment/model-20261014T201339.log

保护SHA256：
- checkpoints/latest.pkl：18b0dfa82e2c2978df524535eb03c0457a4afc3c97b469f89ab57c87a8a825d5
- Replay：dabf45376c658248415d31ed44cccbcb44d4ab8afb6a5a19b8145766541e4dc1
- served Actor：8a59565d2ba4538d13df2d49a047b8482707126dd38dacfe34a4e55ce4958f9f

## 尚缺证据与下一步

缺现场Learner native故障栈和完整服务更新验证。下一步是无动作隔离复现现场Learner路径并捕获栈，区分GPU编译/依赖与服务适配；CPU是可验证的候选运行后端，不静默改生产默认。不能靠反复重启、冻结Learner或删失败数据宣称恢复Online。

本任务未加载模型、启停现场服务、操作机器人或占用现场GPU；保留Stage1及其他对话EXPO激活预约。现场恢复前仍需协调唯一生命周期负责人；其他对话等待器不能被本任务暗中取消。
