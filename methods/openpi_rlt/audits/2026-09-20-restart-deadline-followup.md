# 2026-09-20 重启后现场回归：deadline 与场景差异

用户报告在线模式自动暂停，固定demo比重启前离插孔更远；已放弃当前episode。只读Session实际为waiting_scene/policy_paused，放弃本轮不等于结束整个Session。

证据：重启后a7169aac三条和cefe4c16两条在线放弃均rtc_deadline_missed；中间c3d95d44三条固定demo均无deadline，通过用户pause/abort结束。全部actor5000/checkpoint a6273d48183df852863c3e7d44676e7592414a9cf20c0f097a20909fbf0023da，无在线更新；pending episodes=0，放弃不进训练。

用重启前1b9c0bc9和后c3d95d44保存首帧各复算6次，同观测reference与原记录逐元素一致，排除已测首帧的重启随机变化。两场景state最大差.001343rad，三相机像素MAE19.08/7.20/10.52；图像可见桌面物体和视野差异。两场景首段reference最大差.06399rad、rawactor .03148rad、conditioned .00726rad。不能从单段反事实确定整个轨迹差异唯一原因，更不能说重启没有任何运行时影响。

发现在线cycle每2秒在rollout中也执行36MB checkpoint校验和replay扫描。r10改为先判断Session phase，rollout/HIL/finalizing等阶段不扫描、不哈希、不写ledger。它是已确认的不必要工作，尚未证明是deadline唯一根因。新增Runtime慢推理>150ms或被丢弃时分段计时日志：model/actor/conditioning_guard/total/server_timing，接受trace保存latency_sec。未放宽200ms RTC边界、tracking、workspace；未改动作、平滑、actor权重。

部署descriptor为rtc-upstream-r2-5000-linear-runtime-r10.json，name/step仍原候选5000，增加runtime_patch并更新源码指纹；旧descriptor保留。为避免正在运行的旧cycle在文件更新窗口触发指纹错误，确认waiting_scene后仅短暂SIGSTOP精确注册cycle PID4096514，完成代码和pointer事务后SIGCONT。没有机器人动作、服务重启、模型卸载或重训；当前进程仍旧内存代码，需用户结束Session再启动加载r10。

31 pytest通过；实际Runtime.infer/Stage1/HTTPactor离线5保存场景共520理想跟随命令通过，热推理约98–128ms。该测试不包含现场录制/ROS全部负载，不声称deadline已根治。下一步先rlt_stop.sh再rlt_demo.sh，保持现有模型预加载，只验收1轮；确认耗时和方向后再恢复在线批次。看到r10-idle-cycle-timing-20260920才是新代码。

Cobot证据：projects/rlt/runs/plug_v2/diagnostics/restart-20260920/{recompute.json,scene-comparison.json,runtime-audit.json,patch.json,before-runtime.py,before-rtc_online_cycle.py}。未更改根框架其他项目。


## 2026-09-20 在线现场：Task5 422 诊断与有界 deadline 自动重锚

- 现场在线版本已从 step5000 经多批次门控更新到 actor/learner `6155`；本轮未改 checkpoint，新增不可变运行时 descriptor `rtc-online-20260920T204342-471205-runtime-r12-20260920T210751`，descriptor SHA-256 `8a1355093015c39f3f056564566f1681df0ae6bf1fa955d9b2b5f06c1e414c24`。真机持续提升仍待后续统计验收。
- 自动暂停的主要证据是 `rtc_deadline_missed`：Stage1 server 约80ms，但控制端偶发总等待约0.77–1.18s。r12只对这种纯deadline迟到执行“保持最后已发布安全指令 → 作废迟到future → 最新观测fresh d0重规划”，单episode最多3次；第4次仍fail-closed。tracking/workspace/硬件/recorder故障及12秒命令预算均未放宽。
- deadline重锚期间被动左臂和双夹爪继续使用episode起点的Stage1 d0 action-space锁存值，不允许新的d0重写；stale generation在actor/guard前丢弃。
- Task5 `HTTP 422` 出现在episode106放弃完成后的下一次start。事后状态为stopped/committed，prepare稳定返回next index 107，未留下episode107文件；旧API压缩了原始异常，不能事后断言具体ValueError来源。已让API在服务端记录完整异常类型/堆栈、向客户端返回安全类型，并让Task5Client保留有限响应detail；重启统一8015后health/status/prepare通过。
- 验证：32项RLT focused pytest、10项online unittest、19项replay unittest、9项upstream/clock unittest、47项平台API/proxy测试通过；发布选择器重新核验checkpoint、descriptor和源码指纹通过。
- Cobot最后只读快照：backend `ready_disarmed`，runtime phase `disarmed`，runtime revision `r12-bounded-deadline-reanchor-20260920`，recorder idle/ROS ready，下一索引107；Stage1已预加载，未arm、未录制、未发布机器人动作。
