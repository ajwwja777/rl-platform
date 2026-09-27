# 2026-09-16 统一网页 / RLT 生命周期交付

## 范围

用户授权普通节点/HIL/RLT整合，网页与模型加载分离、标准启停；本次修改RL方法及明确关联的Task5。没有触碰FluxVLA/Franka/框架旧脏修改，没有提交push。

统一8015、内部8016代理、8017默认停用；单writer lease、历史补标仅释放精确当前episode、ROS/图像门禁、可视加载、显式Arm、受管up/down。最终up默认返回loading，可wait-seconds；Arm与Start分开，相对早期计划同步等待/Start时Arm有明确调整。

当前run plug-online-from5416-20260914，actor7160/global14320，seed5416。current.json SHA256 4543969377df791894a9d12b1023557503962037f7042bec1542798ab5722d7b。本次模型/replay/current/训练配置未改；原32updates/验证拒绝/轮内固定actor/平滑保留。

## 验证

- Task5 327 passed（排除converters与旧异步gate测试），随后最新lease/review37 passed。
- JS统一网页6、历史选择2、continuous迟到请求1项通过。
- 正式Cobot overlay相关52 passed，Session/HTTP/shutdown/client/lifecycle/up-down/managed/ROS IO替身。
- 真实8015 UI两次启动、身份/status、模式切换、HTML/assets、离线JSON错误、stop/repeated stop通过。
- loopback8016 HTTP替身：Arm→Start→Pause→Success→replay finalized→Stop通过；假episode/无publisher/无实际数据/训练/归位，最后回收。
- 精确真实series只读检查0.253秒，226完成文件，latest227标签未完，label_blocked=true，未自动补标。
- 最后动态检查8000/8015/8016/8017/9101/9102无监听，GPU无计算进程；只是时点，下次重新核验。

本轮未模型/GPU加载、训练、CAN/Task2/相机或机器人动作；真实闭环待现场，不将替身称为模型/USB并发/真机效果。

## 测试边界

扩大RLT staging为201 passed/19 failed：Stage1缺openpi/lerobot及A6000固定scratch路径，既有配置测试期待2000而当前值200等。Task5converter缺pyarrow。旧gate异步测试关闭后仍断言已入队数据不写入，并在writer未停止时清目录；本次未改gate语义，关闭禁止新采样，已入队会排空。不能宣称全套项目测试通过。

旧VS Code HTTP400/501现场根因未完整复现；固定代理/JSON错误/Connection close已测试，现场连续使用待验收。历史action-conditioning冲突/后臂CAN fault仍fail-closed，不在本次自动恢复范围。

## 部署证据

正式overlay：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-vla/deployments/openpi-rlt/plug-insertion-stage1-v2/runtime-overlay/methods/openpi_rlt

仅同步18文件，新增operator目录rlt_up/down包装，不整树覆盖。保留正式plug rlt_model_server.sh，canonical旧Machine A路径不能直接替代。

Cobot证据根：
/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/runs/unified-console-20260916

- overlay-before-unified-lifecycle.tar SHA256 bde0c32eb964d6fe72e679c6945ac3df72601270183480a564e7466e6cd60e61
- unified-deployment-sha256.json SHA256 e8808a94f4eb89a817a409c197604f3354ec554fd1e7288598e5fb73f0bb6fa1
- Task5 baseline见平台审计。test-source/http-smoke仅诊断，禁止加入正式replay。

回退不自动解包：先停服务，核对当前文件与其他修改，按manifest精确恢复所需文件，不覆盖模型/replay或其他对话工作。

操作：[SOP](../UNIFIED_CONSOLE_SOP.md)；[Todo](../UNIFIED_CONSOLE_TODO.md)。下一步EP227审核与现场全流程。

## 工作区收尾检查

git diff --check通过；使用已有 /opt/miniconda3/bin/python（Python3.12.9）完成 index rebuild 和 doctor，0 error、7条既有legacy attachment warning。默认 /usr/bin/python3 为3.8且缺tomllib，未修改系统环境/框架工具。最后新增Task5关键20项与JS6项再验通过；正式overlay manifest18文件逐一哈希匹配。
