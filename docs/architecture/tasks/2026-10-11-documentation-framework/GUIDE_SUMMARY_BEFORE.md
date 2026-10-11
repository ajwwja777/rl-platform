> 2026-10-11 整理前 guide 项目摘要快照，包含当日另一对话待合并经验说明。历史状态不代表当前运行。

# 统一 RL 实验平台

## 当前摘要（2026-10-09 复核）

原生输入→Actor/Learner、完整状态恢复及多组 CPU 候选研究已形成证据；尚无独立 TEST 或自主持续收益认证，pending Actor 不自动晋升。10 月 7–8 日发布准备、异步 trace 和发送前纯反馈采样修复已交付，录制 timing 字段已在实际 API 可见；最新既有快照为 offline/recorder idle、控制 mode=fault，设备正常后才进行首轮完整受控 Online，实际当前状态需重新查询。10 月 6 日 1,311 份旧笔记本产物已 SHA 归档并进入可恢复回收站；10 月 9 日剩余本地文件已核验归档并移入可恢复回收站，本地仅 AGENTS.md，详细结果见末尾完成记录。来源：所属项目 docs/MIGRATION.md、publication-feedback-20261008 回执及本摘要最新记录；本轮未操作现场。

## 2026-09-30：项目对话入口与并行协作交接

用户指定 rl-platform 对话负责本领域，允许多个专题及 fork 并行。笔记本 D:\Code\jiaan_workspace\rl-platform\AGENTS.md 指向本项目；职责、当前问题、worktree/任务范围登记与现场单一负责人约定见 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/AGENTS.md。对话不共享实时上下文，接管重查 Git 和现场，不沿用历史 PID/版本。
MC30 与 async_rtc20/30/40/50 为可选候选。2026-09-30 修复暂停旧时钟误报及录制 HTTP 占用推理预算，保留 Stage1 的现场重建通过；连续真机 50 Hz 与成功率仍待验收。
来源：cobot_rlt 本次用户指示及实际代码/记录核对；仅追加项目事实，guide Git 不提交/推送。


2026-09-30 实际稳定性修复验收：rl3902fa8/web6b84116 已push、13文件SHA同步。
暂停/HIL旧时钟误报已修复，录制HTTP移出RTC推理预算；web新增收尾未标注录制和通用保留模型重启。
现场活动writer的48帧测试文件提交、writer占用释放、runtime重建ready/disarmed，Stage1 PID233064/start_ticks20025113保留，7个采样硬件身份不变。
Replay3917/Learner7000/Actor3500未变，未执行推理/运动；50Hz连续真机与插入成功率仍待验收。
回执：/data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/outputs/pause-recovery-20260930/；
Cobot对应 /home/agilex/jiaan/project/cobot-web/runtime/verification/pause-recovery-20260930/。
具体恢复按钮、CLI和限制见web docs/WEB_RECOVERY.md与RL docs/RUNBOOK.md；guide Git不提交。

## 当前交接状态（2026-09-29）

固定RLT算法与参数保留；共享采集/评测边界和目录状态归 integrations/cobot_runtime，不再反向import web Python。录制HTTP仍由web同一个recorder提供，领域库归dagger。现场preflight为Learner5000/Actor2500/warmup锚点2567；在线契约78项、恢复Stage1契约35项通过。两冻结环境在A6000独立恢复导入通过。EXPO-FT保留独立仓库/同步训练，登记为待适配。没有启动真实Episode或更新生产Replay。

已push并核验 main：c6b640ead7d1c869dacce65f224f7d49cd18c553；Cobot副本 /home/agilex/jiaan/project/rl-platform 已逐文件SHA复核。A6000主项目 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform，先读README结构和docs/DEPLOYMENT.md，详细批次见docs/MIGRATION.md。

数据/模型实体保持 /media/agilex/Getea1/jiaan/{data,model}；本批未搬迁或新增其备份。完整跨项目交付：/data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/docs/HANDOFF_20260929.md。来源为本次实际源码、Git核验及现场只读检查；guide Git不由本会话提交。

框架复核（2026-09-29）：已只读核对 A6000 项目 HEAD 与 origin/main 一致，并对照实际项目 docs/MIGRATION.md 和 [跨项目交付记录](../../../../../cobot-web/docs/HANDOFF_20260929.md)。上述测试与现场状态来自项目验收记录，本次未连接 Cobot 或重跑测试；实时 PID、模型与采集状态需现场重新查询。

以下保留此前阶段记录；旧路径、PID与“尚未迁移”描述应按上述最新状态及所属项目迁移记录理解。

## 入口

- A6000：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`；先读AGENTS.md与README.md。
- Cobot：`/home/agilex/jiaan/project/rl-platform`。
- 仓库：[ajwwja777/rl-platform](https://github.com/ajwwja777/rl-platform)，独立仓库，main分支。
- 详细记录：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/MIGRATION.md`。

目标：按方法组织采样、Replay、学习、模型发布与评测；当前接入RLT，EXPO-FT沿已有独立实现推进。迁移保持原算法、5k warmup、数据比例和动作语义。

## 历史阶段状态（2026-09-28）

- 当前运行代码、模型、Replay、环境和数据均用新根；正式8015共享入口已加载固定5k及最新在线模型到ready/disarmed，再正常释放。
- 基线learner5000/actor2500/Replay2567；无新数据不更新。A6000与Cobot独立副本测试均通过，一条模拟transition驱动5次更新，重启不重复更新；正式权重和Replay不变。
- Stage1现场加载约44秒，首次固定输入编译16.8秒，后续约76ms。31文件/15.43GB跨机SHA一致；不是新真机成功率。
- 11,123数据文件/158.04GB和2内部链接验收后，旧data下rlt/evaluations/cobot-platform/record/test已删除。新数据在/home/agilex/jiaan/data。
- A6000旧proj-20260904-cobot-realworld-rl已完整归档并删除；479条目一致。历史warmup对比1,731文件已归入outputs/rlt/plug_v3_yyshadow/history。
- Cobot旧RLT历史39,001条目/122.21GB在A6000全量SHA通过，模型归models/history、数据归data/history；旧路径隔离冷加载/释放通过后，旧RLT及其别名已删除。7.49GB现场环境备份已在A6000完成SHA校验并保留，现场暂存tar已清理。
- 当前源码918a2bf已push并同步506文件；自有独立upstream ajwwja777/rlt-openpi固定8cef77e，不fork、不向上游提PR。

## 此前阶段的接续

本批迁移/归档/清理已完成。最终模型offline、无活动Session、无GPU计算进程，系统盘约77.6GiB可用；后续网页新启动的臂/相机任务保留，5臂反馈与3相机可用。本会话没有归位或真实Episode。现场需短轮次验证HIL、暂停、终止结果和复位，再连续在线采集。完整回执在outputs/migrations/20260928-retirement/cobot/，汇总在outputs/migrations/20260928-cutover/final-summary.json。

在线入口：采集页选择/home/agilex/jiaan/data/rlt/plug_v3_yyshadow/online和plug_v3-online-latest；加载完成后才开始Session。固定warmup/Reference是对照，评测不写Replay。完整终端与恢复命令见docs/RUNBOOK.md以及cobot-web/docs/COMMAND_LINE.md、WEB_RECOVERY.md。

## 协作

本项目负责Replay、奖励、learner与在线状态；硬件由cobot-control负责，网页/任务/API由cobot-web负责。录制库暂在web，后续单独交接cobot-dagger。历史58次/22次成功是既有37.9%基线，不能与新路径加载测试或单独Reference目录的成绩混用。

来源：cobot_rlt迁移会话，2026-09-28；对应项目迁移记录、现场测试与Git核验。历史阶段细节以实际项目MIGRATION为准；本会话只维护guide摘要，不提交或推送guide Git。

2026-09-28补充：用户确认数据/checkpoint按用途单机单份保存；原始采集与当前部署归Cobot，训练中间/历史模型归A6000。存放盘点及路径/占用见实际cobot-web/docs/STORAGE.md，各领域实际入口已注明相关资产。当前仍有重复副本待收尾，场景目录尚未改名；本次未删除数据或权重。


## 2026-09-28 存储方案更新

Cobot 数据与模型统一在 /media/agilex/Getea1/jiaan/data/ 和 /media/agilex/Getea1/jiaan/model/。数据按场景分、模型按项目/模型分；本轮不新增 A6000 权重备份。代码、安装环境、运行日志与 PID 留在 /home/agilex/jiaan/project/<项目>/。完整路径与批次状态见相邻 cobot-web/docs/STORAGE.md。
当前批次正在复制/验证及切换。以上旧 /home/agilex/jiaan/data、项目内 models 路径属于迁移前状态；最终完成结论以所属项目 docs/MIGRATION.md 最新批次为准。Guide 本轮只更新记录，由其负责 agent 提交。

## 2026-09-28 Getea1 迁移当前状态

主体数据/权重已迁移到 /media/agilex/Getea1/jiaan/{data,model}，新路径网页历史及 RLT 加载验收后清理了主体旧副本。20:00 Getea1 USB 掉线，FluxVLA 环境/暂存副本的验收和清理未完成；网页已正常停止，迁移进程已退出。恢复识别后先核对文件系统和资产校验，再续迁移，不要直接开始在线训练。详细证据见实际 cobot-web/docs/STORAGE.md 和所属项目 docs/MIGRATION.md。来源：cobot_rlt 迁移会话；未新增 A6000 数据/权重备份，guide Git 不由本会话提交。

## 2026-09-28 20:56：Getea1 存储迁移完成

本批已完成复制、哈希与运行验收、切换和对应旧文件清理。Getea1/jiaan 只保留 data、model；旧系统盘数据/模型目录移除。数据按场景/用途/方法归类，位姿与动作回放归 data/motion；模型按项目/模型/场景/版本归类。代码/环境/日志/PID 留在 /home/agilex/jiaan/project/<项目>。

USB 掉线重连后已完成已迁移资产的全量收据复核；尚不能据此认定硬件链路根因已消除。RLT 新路径暂停加载、在线状态恢复与历史媒体通过；FluxVLA 固定版本离线 baseline/prefix-RTC 通过；π0.5 两入口只做 dry-run。本批未启动真实 Episode 或机器人动作。

完整路径、占用、各项验证边界及回执见实际 cobot-web/docs/STORAGE.md。证据位于 rl-platform/outputs/migrations/20260928-getea-storage/cobot/（Cobot 去掉末尾 cobot/）。同批源码与项目记录已按各自仓库发布；guide Git 保持由其他会话管理。


## 2026-09-29：RLT录制保留模型恢复

来源：cobot_rlt用户要求，不释放已加载模型。先用原Session stop清除无未决Episode的启动故障，保留PID1436537；web新增录制预检/恢复页面与同API CLI、具体503原因，fault可结束Session。dagger修复实时预检clock先取值后等待cache锁的竞态；未修改RL算法/HIL/mask或数据格式。旧503无细节，不能将全部历史错误归因该竞态。

dagger dffc3f1 / web 8dd6983已push并同步，45/198文件SHA一致；254项Python通过、1项既有跳过，45项前端通过。模型保持暂停时连续3次12帧真实录制/放弃清理通过，无新增训练数据；最终模型ready、recorder idle、Session stopped，手动开始下一Session。Learner5090/Actor2545及模型、硬件PID保留。本轮未真实推理或运动，完整HIL仍需现场使用验收。

具体代码/终端步骤/现场回执：/data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/docs/WEB_RECOVERY.md、docs/MIGRATION.md、outputs/rlt-recorder-recovery-20260929/；采集领域细节见cobot-dagger/docs/MIGRATION.md。Guide只更新事实，不提交Git。

## 2026-09-29：RLT启动优化，保留当前π0.5

来源：用户指定先优化RLT，in_the_pot π0.5仍在使用。Stage1只建立VLA/encoder抽象结构，避免真实随机初始化后覆盖；Orbax选择性恢复跳过不参与部署的404,338,688个decoder参数，训练checkpoint保持原样。默认取消72%显存预占，增加项目内持久编译缓存和分阶段加载计时；BF16、seed42、去噪步数、动作/HIL/在线算法不变，未改固定上游。

17项相关冻结环境回归、全部保留参数逐值对比、真实encoder输出及完整固定输入动作/RL Token对比通过；再次启动明确命中编译缓存，输出仍逐值一致。A6000 CPU独立进程初始化+加载32.60→19.80秒，第二次20.10秒；这是CPU/cache条件下证据，不能写成Cobot冷启动从9分钟降到20秒。Cobot实际GPU冷读、编译和显存峰值待用户下一次切换RLT测量。

源码8c3cc48、记录7de2fa2已push并同步Cobot528文件SHA一致；π0.5 supervisor2019008/GPU2019058/客户端2025877与硬件PID/start_ticks保留，没有重启/加载模型/发运动。未搬迁、导出或删除数据/权重，也未将CPU缓存复制到Cobot。

主项目 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform；现场 /home/agilex/jiaan/project/rl-platform。说明与复现：docs/DEPLOYMENT.md、docs/MIGRATION.md、scripts/validate_stage1_loading.py；证据outputs/startup-optimization-20260929，现场runtime/verification/startup-optimization-20260929。Guide只更新事实，不提交/推送Git。

## 2026-09-29：优化后 Cobot 首次加载实测

16:46:18开始至16:51:08启动RL角色约4分50秒；Stage1权重恢复259.279秒，首次编译/固定输入推理21.016秒，后续76.79/72.18ms。仍主要受Getea1恢复限制；不是完整加载20秒。原日志见Cobot rl-platform/outputs/rlt/plug_v3_yyshadow/logs/model-20260929T084619Z.log，详细记录docs/MIGRATION.md，ce6a64e已push并同步528文件。本轮未重载模型或改算法，录制目录旧标签故障由dagger/web修正。

## 2026-09-29：RLT训练与诊断分析页面

来源：cobot_rlt用户要求新增分析板块、训练页整理对齐。web e775308已push并同步202文件，正式8015新增只读/api/analysis/rlt；训练保留进度/发布/参数/主要loss，核心分析归独立页面，历史Warmup图折叠保留。rl-platform 88b3e1c新增轻量聚合和离线Replay PCA/k-means，保留算法；该项目本批只同步3个新增分析文件，未覆盖另一对话的NVMe/probe现场改动。

4项RL测试、24项web Python、47项Node通过；1800/1200/760宽度同左/上边界、无横向溢出、0 JS异常。真实快照3917条transition，前两轴解释31.50%，属于姿态/动作覆盖，不是视觉阶段或失败因果。正式API Learner11750、published11500/Actor5750；日志旧心跳标历史快照，未冒充训练仍运行。只在模型offline/录制idle/无writer时持锁重载网页，9个硬件进程身份保持，未运动或加载模型。

说明：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/ANALYSIS.md；网页使用与记录：/data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/docs/COMMAND_LINE.md、docs/MIGRATION.md。Cobot快照在 /home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/analysis/replay_projection.json，手动运行 envs/online/bin/python scripts/analyze_replay.py 重新生成。HTTP回执归web/runtime/verification/rlt-analysis-20260929/。Guide仅追加事实，不提交/推送Git。

## 2026-09-29：Replay 构成、采样实验与图像诊断交付

来源：cobot_rlt 实际代码、离线实验及正式8015只读验收。
rl-platform e20edada 已 push 并选择性同步 Cobot，保留其他对话的现场配置。
3917 transition / 277轮；成功68.14%，新自主成功102条/2.60%。
完成4种成功采样配比×3种子×2000更新、14个发布Actor回看、
checkpoint11000梯度/Q探针、6记录帧三相机306次遮挡前向。
未改生产权重、Replay、loss/reward/UTD，也未产生机器人动作。

真实batch审计从下一次Learner项目入口启动生效；历史身份不能补造。
10项轻量测试及1项真实JAX Learner数值等价回归通过。
--versions-only固定保存的轮次集合，不随新增Replay改验证对象。
Online版本回看不是独立验证；未保留同一步Critic处留空，未实现自动GPU验证任务。
详细结果、复现命令及建议：
/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/DIAGNOSIS_20260929.md。
Cobot JSON：/home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/analysis/。
仅小报告回传A6000 outputs/rlt-diagnosis-20260929，无新增数据/权重备份。
Guide只更新事实，不提交/推送。

## 2026-09-30: credit-assignment candidate

Source: cobot_rlt dialogue, actual Replay experiments. Seven methods x three seeds x 2,000 updates complete. Terminal-only sampling worsened HIL MAE ~12%; MC30 improved early Q AUC .54 -> .67 with only ~0.55% HIL MAE improvement against matched uniform, so no proven robot success improvement. Registered plug-v3-credit-mc30 has separate resumable weights/logs and shared Replay. Actual candidate restore/5 updates per synthetic transition/publication/exact restart and spawned process startup/ordered stop passed without EnvDriver. Recorded Stage1 RTC reuse is verified, but integrated asynchronous RLT RTC/40 Hz and field success acceptance remain pending. A6000 full record: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/EXPERIMENTS_20260930.md. Cobot weights: /media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/history/candidates/credit_20260930/mc_30/online_candidate. Default original online preserved. Project Git pushed; guide Git left to its owner.

Final release verification (2026-09-30): main/origin ff3ad397b676e088a6df767073e7252c2d1ebefa; A6000 tree clean; selected Cobot files SHA256 match. Formal 8015 read-only verification passed; no robot motion. Guide Git not committed by this dialogue.

## 2026-09-30: Critic action-guidance follow-up

Actual frozen-state audit and three Actor-Q-loss-off ablations completed with no
robot movement or production weight/Replay changes. MC30 partially mitigates Q1
preference conflict (23% -> 36% HIL endpoints preferred), but gripper MAE worsens
~4.5%; early-AUC paired interval includes zero. Accepted Online Replay contains
six autonomous successes vs 23 assisted successes and 44 failures. Q-loss-off
imitation gain exceeds MC30 on this proxy, so autonomous benefit remains unproven.
Details / source / reproduction:
 /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/EXPERIMENTS_20260930.md.
Next: observation/action/HIL alignment and autonomous vs assisted credit diagnosis,
then independent field evaluation. Guide facts only; guide Git not committed.

Follow-up release verified: rl-platform d6548b329b3f571fe8e65e0a7cad446c68a5058c = origin/main, clean A6000 tree, four changed source/docs SHA256 match Cobot. Replay unchanged; model offline and GPU idle. Figure/release receipt in outputs/rlt-diagnosis-20260930. Guide Git untouched.

## 2026-09-30: optional asynchronous RTC execution
A6000 main code now owns RLT7D execution registry/hooks; shared RTC queue/sampler/
physical-time EMA remain in VLA integrations. Original entries synchronous20;
MC30 RTC20/30/40/50 selectable/manual, common candidate branch, no weight copies.
77 related offline contracts and3 pinned Stage1 loading tests passed; real Stage1+
Actor synthetic-GPU7-variant audit passed (original effective17.93Hz/p95 98.19ms;
async20 p95 50.03ms, async40 25.05ms). Native Replay -> two in-memory learner
updates passed. All are non-motion checks; real ROS/RPC/Learner-contention/robot
jitter and autonomous success still pending. Backbone training-time RTC not run;
Stage1 remains frozen. Replay hash unchanged, GPU idle, web offline, no Session.
Usage/rollback/structure: /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/RUNBOOK.md.
Reports: same project outputs/rlt-diagnosis-20260930/execution-gpu-comparison.json
and execution-final-release.json; Cobot same suffix below /home/agilex/jiaan/project/rl-platform.
Code committed/pushed before selective hash-verified sync; local/probe changes in
other dialogues preserved. Guide facts only, no guide Git operation. Source:
actual code/tests/Cobot live8015 catalog and this dialogue, 2026-09-30.

Final publication verification: rl-platform ecaa8aac2507444f1331c8523b8d90d2b6d86374 = origin/main, clean A6000 tree; selected Cobot SHA hashes match. Receipt in RL outputs/rlt-diagnosis-20260930/execution-final-release.json. Guide Git untouched.

Frozen A/B guidance clarified: use existing Warmup5k frozen entry with an execution override; MC30 online entries can learn on load. Documentation release 1afc52b9e359a70c281b798e2cbb1d2430f4499b pushed and RUNBOOK SHA synced. Guide Git untouched.

## 2026-09-30: RTC field timeout and retained-Stage1 recovery

Actual19:50:31 MC30 async_rtc50 EnvDriver failed because queue delay exceeded its
200 ms budget; final supervisor traceback was a consequence. Old inference
latency omitted recorder checks, so full contention cause is not established.
RL now records actual/allowed delay and inference/recorder timing; bounds and
algorithm settings unchanged. Web collection/deployment display root cause,
Stage1 retention and explicit status/output/runtime recovery controls.

Recovery verifies fully exited owned runtime and no pending writer/evaluation,
uses original launcher and refuses implicit Stage1 reload. No automatic inference,
home, label, data deletion or Replay commit. Completed orphan writer can permit
UI shutdown only after local process-group verification. Source/docs release:
rl-platform d4c798f, cobot-web4f5db63;18 Cobot hashes checked. Full backend709/6skip,
DOM56, selected RL40, additional shutdown/CLI32 passed. Formal8015 updated with
Stage1 PID233064 retained and14 hardware identities unchanged; recovery/start
were not invoked. Live restart/RTC contention acceptance still pending.

Usage: /data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/docs/WEB_RECOVERY.md
and /data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/RUNBOOK.md.
Evidence: /data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/outputs/rtc-runtime-recovery-20260930/;
Cobot /home/agilex/jiaan/project/cobot-web/runtime/verification/rtc-runtime-recovery-20260930/.
Source: actual code/tests/live API and cobot_rlt dialogue. Guide Git untouched.

Final record-only release 0d197b9b0bbb7f5a33f31d8487dedb54b6270ff0 = origin/main; A6000 clean and all18 selected Cobot source/docs hashes verified. Guide Git not committed.

Operator follow-up: retained recovery launched20:33:53, Stage1 still233064; at20:34:15 async_rtc50 then failed execution_clock_late. Recovery works, but50Hz field timing is not accepted. No recovery/start requested by agent. Evidence in web outputs/runtime rtc-runtime-recovery-20260930/operator-followup.json.

2026-09-30 交接发布核验：rl-platform 93b96eececc19fe6c0374e5dbcd59fb26a17f58a = origin/main，A6000工作树干净；本批选中文档/源码已逐文件SHA同步Cobot。详细回执 projects/cobot-web/outputs/pause-recovery-20260930/handoff-release.json；guide Git未提交。

2026-09-30：运行选项、录制暂存与历史补标签已交付。模型详情区分发布频率与逻辑／Replay 步频，支持 Hz／RTC／滤波选择；暂存结束当前 writer，保留文件与模型任务，完整未标注记录可补标签，不隐式提交 Replay。离线网页后端 725 passed／6 skipped、前端 62、采集 78、RL 79 通过；连续真机 50 Hz 和成功率仍未验收。

最终发布回执：web e75d3cbb、dagger bb85a8a6、RL d878bded；网页 PID 866075、模型 offline、录制 idle、ROS readiness ok，采样进程身份保持，未启动模型／动作或改生产数据。此前两段已被写成问号，现依据实际项目 docs/MIGRATION.md 与 cobot-web/outputs/recording-defer-rate-20260930/final-release.json 核对后重写；详细过程仍归所属项目。

## 2026-10-01：精简执行选项发布事实

来源：cobot-web 本次任务。模型步数项去除 Hz，运行 Hz/RTC/滤波独立成一行；历史结果成功/失败/未知自动保存，保留原备注与训练许可，CAS 与丢响应查询确认。VLA 共用执行配置、物理时间发布/滤波及可选 chunk 执行已接入当前暂停适配器，RLT 复用配置合同。

A6000 web af56ffc / VLA a9f87e0 / RL aee49de 已提交 push。Cobot web 209 个运行文件全 SHA 校验，VLA 16/RL 2 个本次文件校验；只重载 8015，新网页 PID 927182，模型 offline、录制 idle、无 writer/active lease。9 个采样硬件/模型进程身份保持。共用模块现场导入与 4 个 VLA 文件启动计划、两项 π0.5 dry-run 通过；未新加载 GPU、未真实 Episode/Replay 改写/动作，连续真机 Hz/效果待验收。

离线前端 67，web 后端 87（另 1 skipped），RLT 21，共用 VLA 27，G05 15，XR1 26 项通过。回执：/data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/outputs/execution-compact-20261001/final-release.json；Cobot 对应 runtime/verification/execution-compact-20261001/final-release.json。guide Git 不由本对话提交或推送。

## 2026-10-01：方法／步数、历史结果与 NVMe 路径现场事实

来源：cobot-web 当前任务。原版／MC30 方法与纯数字步数独立，固定 5000 保留，新增在线分支读取真实已发布步数；所选 50 Hz 的详情与实际生效设置分开。历史标签接口与目录迁移一致，普通采集节点历史保留记录结果，成功／失败／未知可修改并同步同目录显示。Stage1 清单和实际 CPU preflight 恢复 NVMe /home/agilex/jiaan/data/rlt/plug_insertion/reference_4999；在线资产按原登记引用，未搬数据／权重。

rl-platform 发布 bf111eb70cb25d3e65de58a0dd8a63c8f8b23cbe；三项目源码／记录 SHA 已核验，正式 8015 只重载网页 PID 970937。模型 offline、recorder idle、无活动 writer；9 个采样硬件／模型身份及固定 5000／历史标签 SHA 不变。离线前端 69、完整 web 734（6 skipped）、末次相关 60（1 skipped）、采集 78、RL 24 通过。8 条实际历史标签 GET 与目录 DOM 通过；未启动 GPU／训练／动作，真实在线分支与 50 Hz 连续运动、浏览器视觉仍待现场。详见实际项目 docs/MIGRATION.md 与 web outputs/catalog-results-20261001/final-release.json；guide Git 未提交或推送。

## 2026-10-05：离线输入合同与模型候选研究事实

rl-platform独立分支audit/q-guidance-20261005完成Stage1全部134训练见过Episode/19022帧的实际模型诊断、120专家/1186 transition动作重建，确认30Hz旧参考与20Hz训练目标合同不一致；不能直接给旧5k Actor换重采样20Hz参考。两条早期错误任务提示词轨迹的29条记录已精确追到Warmup训练，入口已增加ROS构造前任务检查；现有启动脚本任务设置正确。30组Actor继续训练、6组从零原生Warmup5k及此前6组Critic目标对照完成；没有新模型凭拟合/Q/TD自动放行。

可选staged发布实际CPU多进程验证候选2507/执行2500隔离、更新预算与恢复；3个固定真实输入在实际服务独立恢复间动作/Token完全一致，缓存数值严格阈值失败仍保留。软件412项通过，在线冻结环境相关28项通过。本批未部署/启停现场/运动，生产Replay/权重/默认配置/固定上游不改；独立自主收益与真机Online放行证据不足。

1728/1751条rollout动作窗口唯一精确追到trace，另11歧义/12缺失留空；完整覆盖Episode的记录时间跨度不符合统一20Hz目标，不能当作硬件发布时钟。五帧原录制输入的A6000 GPU0输出指纹探测没有唯一匹配，旧VLA图像身份仍缺证；未用最近帧补造输入。

来源：/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/docs/MIGRATION.md 2026-10-05记录、outputs/offline-model-selection-20261005/delivery.json及实际launch/report回执；Guide只追加事实，本对话不提交或推送guide Git。

## 2026-10-05：HIL 命令/反馈溯源补充与相机输入合同修复

本轮延续 q-guidance 隔离工作区，读取现有 216 份 HDF5 的低维字段。31 个 Episode 的 2,586 个严格身份配对 HIL 步骤中，2,585 个 action 等于 next-state 测量反馈；786 步有反馈完全相同的时间兼容原始帧，722 步候选命令相同，但缺逐步命令发布回执，未恢复为可直接训练的历史命令动作、未改 Replay。固定上游 HIL 常规路径使用 cmd_topic 收到的命令，首次无命令时才回退反馈；本地反馈目标与其语义差异已验证，自主收益因果证据不足。

修复 RosTask2IO 两条相机采样路径的 RGB 条件性合同错误，并在同步/异步 trace 记录步骤前后 I/O 证据，保留左右独立 coordinator 命令与源/接收时间，右臂有效性不依赖左臂新鲜度。该证据不是执行确认，不改变 Replay 动作或生产默认。现场只读查询三路相机当前均 rgb8，五帧 BGR 对照不匹配旧 Token/ref，不能把颜色称作已定位现场根因。

418 项整套回归、3 项分析专项测试通过，最后修正后 49 项定向测试通过；初始缺依赖环境失败日志保留。A6000 GPU0 只做五帧已有图像对照后释放，GPU1 未动；未启停现场、加载现场模型、运动或切权重。未新增训练或放行 Actor；模型独立能力与受控冻结验收仍未通过。详细身份、命令、图和边界：`outputs/hil-command-audit-20261005/`，上一批交付仍在 `outputs/offline-model-selection-20261005/`。

Source commit: a3e17a5436d650d984f4e17d3c59109db6bb54e0; field code/runtime not switched. Guide Git not committed or pushed.

HIL 配对 Critic 补证（同日）：CPU 原 5k Critic、229 个窗口/31 个 Episode；全 Episode 终端标签为辅助成功 30、失败 1。仅替换已观察命令候选相同的槽位，并与同槽位 FP32 反馈基准比较以分离量化影响。完整 trace 覆盖的 12 个训练 Episode 平均 ΔQ1=-0.01036，Episode bootstrap 95% [-0.01821,-0.00273]；反复使用的开发集 3 个完整 Episode 为 -0.00827，[-0.01691,-0.00224]。min-Q 也平均下降。缺失命令使其仅是条件性冻结 Critic 响应，不证明命令最优、精确历史动作身份或自主收益。初版私有脚本把选中非终端 success 标记当整轮结果的错误已修正并保存旧报告；Q 数值与区间完全不变。详细身份/配置/CPU命令/图：`outputs/hil-command-audit-20261005/command_critic_report.json`。

2026-10-05 Online provenance audit: Replay75 Online Episodes/1446 transitions, operator labels8 autonomous success/23 assisted/44failure;172 unique core trace matches,10complete Episodes. Saved Learner11000 vs Actor snapshot11500; original5k release is historical. Optional input receipts defaultoff, full426 + finaltarget17 tests; A6000 main bce172b563cf51dabbc3ab298f7c083976c88505. No Cobot deployment/GPU/motion, no model acceptance. Details projects/rl-platform/outputs/online-provenance-audit-20261005/.

2026-10-05 HIL target/trace observation contract: received right coordinator command at interval start, same takeover epoch, fail closed on missing/mixed evidence; optional defaults preserved. Fixed raw observation/next-state cache indexing and async evidence propagation; CPU454+82 tests, A6000 main e439d6c41549d16a19f1cbc35d74ecbaf616ff1b. No model training/promotion or field deployment. Details rl-platform/outputs/hil-target-contract-20261005/.

## 2026-10-05：私有输入到学习的原生闭环核验事实

rl-platform 32f513a36aaa7ff55dedfba62f556f1aef4e735d 已提交push，A6000主工作区/隔离工作区干净。真实5k权重与CPU私有原生Replay/Actor/Learner、合成HIL输入验证3条记录逐字段收据及实际归一化batch指纹，15Critic/7Actor更新、完整状态恢复；私有服务/端口退出。新增经验Actor抽样2/2/0，当前500更新导出配置在15更新结束时仍2500，flush后候选2507；冻结服务动作2500不变。采样覆盖与导出时序是机制证据，不是历史根因/自主提升证据。无现场/GPU/Stage1加载/生产修改/权重晋升。详细身份、命令、配置、图和验证边界：projects/rl-platform/outputs/input-learning-chain-20261005/；源码与事实记录见项目scripts/validate_input_learning_chain.py及docs/MIGRATION.md。Guide只追加事实，本对话不提交或push guide Git。

## 2026-10-05：近期经验采样模型对照事实

rl-platform事实记录95edbf7a3b57ea1bcc54f8bc36b23e0c8e13195f已push，A6000主/隔离工作区干净。CPU六组配对native uniform/stratified，从不可变5k各2000Critic/1000Actor更新，目标/loss/划分固定；本研究近期抽样约1.88%→40.03%。六旧辅助成功开发Episode关节相对uniform改善5.6%、夹爪变差8.8%，整Episode区间支持取舍，联合代理条件失败；无候选晋升或真机效果结论。可追trace动作的Online训练仅1个含人类动作Episode，对应Online开发0个，独立测试缺失；cached输入/反馈身份边界保留。研究池不是生产4013 Replay，不混用比例。六研究完整状态仅留A6000，来源/命令/配置/四图/结论：projects/rl-platform/outputs/recent-sampling-model-study-20261005/。PID720279已退出，GPU/现场/生产默认/Replay/权重/固定上游未改；后续离线定位夹爪loss/source梯度。Guide只追加事实，本对话不提交或push guide Git。

2026-10-05 Actor source/gripper study: CPU42 native cases,7 parity exact; gripper absolute roundtrip exact. Initial5k policy/expert updates worsen old6 assisted grip fit, HIL improves; noQ does not fix it. Six paired non-HIL gripBC0.5 training runs reach7000/3500, matched index hashes; stratified old6 grip improves0.03490mm vsnative but expert worsens0.02430mm and initial5k retention gate fails. No Actor promotion/field/GPU/production/default/upstream changes. Factual release 60e53aad5e8df3e4d23de4a0a55eeb6e316bc23e; outputs/gripper-update-diagnosis-20261005 REPORT/delivery/3figures and source identity. Independent capability/acceptance insufficient. Guide Git untouched.

2026-10-05 raw image identity audit:216HDF5/648camera arrays uncompresseduint8; rawWarmup170/31598frames andOnline46/4674, sourcecamera skew>100ms0, regressions0. One invalidzero frameWarmup9/155; ID9absent archived5k (notfullUUID proof). 59recordings65gaps>200ms, no fabricated continuity. Exact historical VLA input still insufficient; no newtraining/GPU/field/production changes or Actor acceptance. Record release 1c0fe7eed35d5963d852317860e438379fcfff1a; outputs/input-image-identity-20261005 REPORT/delivery/timingfigure. GuideGit untouched.

2026-10-05 frozen proposal/command probe: CPU13 states x2891 cachedwindows/220Episode, nativeRightArm first-command .03rad/.004m clips0; no future-state/field simulation. Initial5k old6 assisted DEV Q1 HIL-direction derivative allEpisodeavgnegative, mean-.04808 CI[-.06678,-.02938], Q2CIcross0/minQnegative. Not historical proposals/optimal commands or task proof. Multi-axisCI summary layout fixed;7 priorstate pairedendpoints agree, source/norm/stateSHA verified, worker826360 terminal. Recordrelease bdf59dd2141f8c38dd621eaa213bbf5e0d7d1da5; outputs/proposal-execution-diagnosis-20261005 REPORT/delivery/2figures. No training/GPU/field/production changes or Actor promotion. GuideGit untouched.

2026-10-05 bounded lossless Replay inputs: source release 0b50423331bedc752cb14bbc14709ee87bcbddb2, defaultsnapshotcount0, max3/EnvDriver and8MiB rawarrays each, explicitindependentroot+trace+inputaudit. Restore RGB/state/prompt/RTC/native/actualserializer receipts, no pickle/model loading. CPU27targeted/468full tests; private real5k syntheticchain3snapshots->15Critic/7Actor->staged2507/served2500 exactresume, ownedservices/ports exited. Real oldcandidateRGB+syntheticfeatures5.53MB->1.00MB roundtrip,130mswrite/29msread(singlefilesystemsample, not fieldlatency). No production/default/upstream/field/GPU changes or Actorpromotion. Modelconsistency/validcommands/independentcapability still insufficient. outputs/input-snapshot-verification-20261005 REPORT/delivery; GuideGit untouched.

2026-10-05 actualStage1 lossless-input CPU closure: sourcebase0b50423, two independent restores/4cores, retainedcandidateRGB+rightstate frames0/2/freshplugprompt, one synthetic native transition. All6 current/next z/proprio/ref native and serializer outputs exact;31assetSHA unchanged, workers879350/882640 exited. Factrelease b138e79834511958ac16f20d4b7e484f460b2db0, main/origin clean. No training/GPU/field/production changes/Actorpromotion; historicalinput/validHIL/independentcapability still insufficient. Details rl-platform/outputs/stage1-input-snapshot-model-20261005 REPORT/delivery. GuideGit untouched.

2026-10-05 realStage1->frozen5k Actor closure: CPUtwofreshdiagnosticanchors/4native-or-storedfeature calls, exactnativeRPC/direct outputs, actualcheckpoint/snapshottree identical, originalnormSHA verified; firstcommandclipping0. FP16featurequantization atmost.000474rad/.0399mm andQ1+.0036 on2anchors, notindependenttest/rootcause/taskgain. Privateanalysisnormidentity/logpath errorsfixed, failurespreserved; worker898496/Actor898526+failedActor896931 andports43105/35491 exited. No training/GPU/field/production/modelpromotion. Facts acd1cd2a3b84b322085deab838de1725d0d1d5fd main/origin clean; outputs/stage1-actor-chain-20261005 REPORT/delivery/figure. GuideGit untouched.

2026-10-05 preonlinepacket/entryfix: real5k nativeCLI formalseedblock oldexplicitstaged->automatic reproduced; rlt_up nowforwards explicitCOBOT_RLT_SEED_PUBLICATION_POLICY/COBOT_RLT_SEED_REPLAY_BUDGET_POLICY, defaultsautomatic/new_arrivals preserved, reusedbranch requestedpolicy+Actorpath conflictreject/no mutation.19targeted/478full, complete5k params/optimizer/RNG/sourceSHA andresume unchanged. 5320b568ebd1881a04ed8b43b2b6a3942ae9cd7a main/branch pushedclean. Outputs/preonline-delivery-package-20261005 REPORT/manifest/delivery/6keyfigures/standaloneoperatorprompt; modelupgrade andOnline notaccepted, no newActorpromotion. No GPU/field/services/training/productionassets/default/upstream modifications. GuideGit untouched.

2026-10-05 field static delivery: rl-platform88b1bdae71fdf34aa0a8bddb090bf0e451aeece0 pushed/mainclean,622fieldSHA verified;19oldfiles backedup/56added plus second audittool/doc backup, no conflicts/default changes. Field9imports/68CPUcontracts passed;31Stage1/immutable5k/norm matched. Actualshortprompt CPU Stage1 sixfields exact across2processes and4native ActorRPC exact. No fieldGPU/model load/services/motion; Replay unchanged; no newActor/Online acceptance. Pending explicit bounded non-motion GPUcheck + independent frozen robot evidence. Details projects/rl-platform/outputs/preonline-field-verification-20261005/; guideGit untouched.

2026-10-06 actual Cobot RTX4090 no-motion verification COMPLETE: native faithful20/async_rtc20 actual fixed5k Actor2500 each30 simulated steps, GPU backend, exit0/25.195sec. Fixed diagnostic wrong async backend and explicit phase to avoid Reference substitution;480 full+2fieldCPU regressions pass. faithful synthetic chunk-boundary interval130.445ms/effective17.924Hz vsasync20.000Hz, not robot outcomes.31Stage1/5k/norm/Replay/recording unchanged;web970937 retained, RLT offline, recorderidle, owned jobs exited/GPUreleased. No Actorpromotion/default/fieldservices/motion; independent autonomous tests0, Online acceptance pending. Source beec5db08f3631b8a1f7ae7cbf55b224e2bff906, report/8figures/receipts/operatorprompt projects/rl-platform/outputs/preonline-gpu-20261006/. Guide Git not committed or pushed.

### 2026-10-06 eRLT/开源核对与训练诊断交付

rl-platform最终250c5a2（代码9b7601e）已push/sync Cobot630文件SHA一致，A6000两目录495测试通过/61既有warning，Cobot实际Python3.10 CPU专项15通过。真实固定5k/归档2567训练数据CPU审计已有分路投影/LayerNorm、七维动作梯度非零；两私有8更新分支加诊断前后参数/优化器/RNG/旧指标完全一致。新增默认关闭原生TD诊断、缺指标留空训练曲线、完整Episode成功率AUC工具；eRLT2610.00913v1 AUC不是Critic ROC-AUC，外部开源cccf949c仍完整Actor输出，未照搬residual/截断/算法。详见rl-platform outputs/erlt-reference-review-20261006/REPORT.md及现场prompt。

没有新Actor晋升或模型能力改善证明，原5k仍基线；本批无现场GPU/模型加载、服务启停或运动，生产资产/默认/上游不变，web970937/RLT offline/recorder idle。可准备受控冻结真机验收，直接持续Online能力仍证据不足。此处只记所属项目事实，不修改guide治理、不提交guide Git。

2026-10-06 acceleration/HIL studies: code f5f11f5 and factual release 123b4d4af0972788cd1887174c34ea0f510bdd34; committed async actions now keep their old Reference/version/source, RTC on/off and delay0/2/4 verified. Opt-in async20/noRTC/no smoothing, defaults unchanged. CPU18 paired private continuations reject HIL-source deletion; four-pool and lower-budget benefits remain imitation-only with joint/grip/old-scene tradeoffs, no Actor promotion or independent test. A6000 503tests/61existingwarnings, CobotPython3.10CPU29,630sourceSHA verified and protected5k/ReplaySHA unchanged. Mount restored; web8015 read-only query refused, no field service/GPU/model/motion changes. REPORT/5figures/config/commands/SHA/failures/operatorprompt in rl-platform/outputs/acceleration-hil-ablation-20261006; GuideGit untouched.


## 2026-10-06：共享采集／评测trace终结修复

现场MC30 Frozen Actor3740与Warmup5k Actor2500均在轮次收尾因CollectionTrace缺少finalize导致env_driver退出。修复4ce0a88补齐共享包装器转发及评测NoTraceWriter空生命周期，保持本轮用途锁定，评测不写采集trace；记录2c28d5b。A6000相关88通过，9新增回归在基线均失败、修复后通过；全量383通过/19原有环境失败，与基线失败集合一致。Cobot冻结Python3.10实际14项收尾回归全部通过，630源码文件SHA校验一致。

源码锁内同步保留15个现有ROS/相机/臂/web/Stage1进程身份；用户随后自行重新加载Warmup5k（PID83992），16:26只读ready/disarmed/policy_paused=true、Session未开始、step0、Learner disabled。本会话未执行恢复POST、模型/网页/硬件启停或运动；真机修复后终结轮次尚待用户验证，不补写历史pending trace/Replay。16:14故障轮metrics transitions_written=0。详细事实及证据见主项目docs/MIGRATION.md、scratch/rl-platform/trace-writer-lifecycle-20261006和现场runtime/verification/trace-writer-lifecycle-20261006。Guide仅追加事实，不提交/推送Git。

后续现场观测更新：用户自主执行的Warmup5k Frozen评测连续三轮点击success均正常收尾，metrics原生episode245/246/247为success=1、transitions_written=0，Actor2500；日志三次success提交，无Traceback，env_driver85189仍存活，随后用户开始第四轮。证明本次评测终结崩溃已在真实调用路径消除；不将操作者标签当成独立成功率验收，采集入Replay、failure/aborted真机路径仍未在本批现场试验。证据live-terminal-observation.json（现场）/同名txt（A6000scratch）。


## 2026-10-06：A6000归拢与按轮更新候选

主仓库main4036c06a53c3cd1dd966e696e2295da7f11dcdd1已推送；独立experiment/roundwise-online-20261006验证后仅串行fast-forward，其他worktree保留。可选CPU按轮工具提供完整Episode/source/奖励检查、多次接管截断、四池归一化与Episode先采样、Actor全轨迹／无Q梯度、完整恢复／原生导出；默认Online／原5k／Replay／固定上游不变，未同步现场或操作服务／GPU／机器人。

两种子六完整候选各Critic1000／Actor1500：末段Actor损害旧专家和自主成功拟合；全轨迹／低LR缓解，仍无独立位置OOD任务及保持放行证据，候选不发布。CPU原生导出32状态最大误差1.1921e-7，恢复全部参数／优化器／RNG和双采样状态逐位一致。最终422通过／18基线同名环境失败；source SHA不变。生产Replay只读4013行279Episode，时间代理合格1800、混合37，非评审1861。

1311既有笔记本产物逐SHA归拢A6000（1227相同／84补齐／0冲突），收据scratch/rl-platform/laptop-artifact-migration-20261006；新增实验和图仅A6000。Windows副本删除被自动策略拒绝，旧副本保留未绕过。证据outputs/roundwise-online-review-20261006/REPORT.md、delivery.json及6图；正式流程docs/ROUNDWISE_ONLINE.md已展示后落盘。guide仅追加事实，不提交／推送guideGit。

早期adhoc网页查询拒绝为继承HTTP代理影响，禁用代理后200，不是服务故障；用户冻结评测Replay0／Learner禁用，不计为Online已学习改善。现场后续操作所有权仍归用户。


2026-10-06 follow-up：笔记本旧产物已采用可恢复Windows回收站方式清理，1311文件和临时3文件在原路径消失，回收站内SHA复验一致，未清空回收站。A6000迁移副本保留；清理收据scratch/rl-platform/laptop-artifact-migration-20261006/cleanup-recycle-20261006.json。早先永久删除策略拒绝为历史结果，不再描述为当前未清理；具体拦截规则未知。guide仅事实追加，不提交推送Git。


2026-10-06 frozen target/jitter localization: core2dae01b/final6c6c20f pushed,13fieldfilesSHA verified; A6000 and actual CobotPython3.10 CPU76each. Diagnostic-only aborted retention/queued Actor+Reference/physical schedule+feedback/request evidence and Episode-based analyzer; defaults/5k/Replay/sharedVLA/web/upstream unchanged.9selected processes and protected5k/ReplaySHA unchanged; fieldoffline/recorderidle, no model/GPU/motion. Next-runtime fixed5k diagnostic opt-in enabled; six measured-position frozen DEV Episodes required, no newActor/Online acceptance. Report/operator prompt/delivery projects/rl-platform/outputs/position-jitter-diagnosis-20261006; workflow docs/FROZEN_LOCALIZATION.md. GuideGit untouched.


2026-10-06 active-control correction and Critic-policy study: user confirms fixed left arm and gripper; withdraw inactive-gripper prediction error as task degradation/rejection gate.15 matched CPU continuations,2000Critic/1000Actor each, plus exact held-gripper external20DEV inference(all15). MC reduces HIL-vs-Actor Qgap without consistent active6-fit benefit; currentQ0 improves fitting versusQ0.1, small effects. Active6 teacher retention better preserves old TRAIN, external5autonomous/6assisted DEV fits approximately retained; CIs cross0, no autonomous/OOD gain claim or independent TEST.18 tests passed; no field/source/weights/Replay/default operations. Sourcef23e70c, final factual15b57d9 pushed main; main/worktree clean. Full hashes/config/commands/plots: projects/rl-platform/outputs/critic-policy-bootstrap-20261006/REPORT.md and delivery.json. GuideGit not committed/pushed.


## 2026-10-06 Supported7k 离线交付

主仓main/实验分支4805d545c1551fe239d0baebac22956ddef36586已推送。18组匹配CPU续训，实际新7000/Actor3500候选，MC30/Q.1/active6保持50/ActorLR1e-5；修复async RTC完整输入缓存身份与候选专家phase合同。2496窗/201Ep private TRAIN、90专项测试在3.11/冻结3.10均通过，原生8更新连续/4+4恢复逐位一致，冻结32输入最大误差1.19e-7。

A6000事实：projects/rl-platform/outputs/supported-online-20261006/{REPORT.md,delivery.json,progress.json,operator_prompt.txt,release/manifest.json}；流程docs/SUPPORTED_ONLINE_DELIVERY.md。新冻结/Online可选ID已登记于主代码，原5k默认未改。源码/76.8MB私有候选资产bundle保存在A6000，未安装Cobot；现场只读GET旧5k仍ready，未启停/加载/运动/占现场GPU，用户保持现场所有权。候选具备离线软件/恢复及代理稳定性证据，不具备独立自主Online持续增益证明；新RTC真实Stage1/延迟与新候选受控冻结验收仍需补证，staged候选更新不自动发布。guide只追加事实，无Git提交/推送。


## 2026-10-07：Online当前数据与末端曝光对照

主项目7a9c58c676cbc749a0a2563dabb78e5cb1357e60已main/branch推送；现有完整7000参数/Adam/target/RNG跨机逐叶身份一致，3臂204实际batch CPU对照只改失败terminal slot配额。quota4压低新末端Q1但损害旧HIL回报/mixed目标拟合，未采用。新增隔离工具及资产输出守卫7合同测试在A6000 Python3.11/3.10均通过，无现场代码/参数/权重/Replay/服务/动作/GPU修改。

现场2777窗208Episode，新增7条281窗全采样、Critic281/Actor140更新，保存7281/3640但served3500未发布；新1次HIL18窗7含human、53独特human步，首18更新Critic25/Actor13 HILdraw。281strict回执与journal数组SHA全匹配。TRAIN新自主拟合改善但旧HIL部分代理退化，独立TEST/自主收益不足。累计50.1ms时钟故障发生在Learner caught-up，尚未修复；末轮录制仍未完成。收尾标签到Replay10.993s中逐窗回执跨度10.632s，成本待细分。事实报告/4图/命令/全部SHA与可恢复进度：projects/rl-platform/outputs/terminal-exposure-20261007/{REPORT.md,PROGRESS.json}，正式记录docs/audits/2026-10-07-online-current-data-terminal-exposure.md与MIGRATION.md。用户只断机械臂电源；此处仅摘要追加，不提交或推送guide Git。

同批最终只读2026-10-07T18:39:28.669309+08:00：故障录制已由现场收尾，stopped/committed episode_000010.hdf5/3000帧，原“仍未完成”为早期快照；Native仍退出/Stage1保留，时钟问题未修复。事实补记d92d54d57b2762895e6b440d13d620feaade7e8d已推送main；guide Git不由本对话提交。


## 2026-10-07：Online 下一轮策略与近期Episode采样对照

主项目d808b4994e95b2cd2c5bc0ba78a800bff6b757dd已main/隔离分支推送；6组CPU、3采样种子配对、每组281Critic/140Actor更新。仅改recent51slots窗口→Episode，其他77slots配对不变。新1条HIL TRAIN拟合改善1.0–2.7%，重复DEV失败Q—行为回报误差跨种子均值恶化；未采用/发布。固定实际7281六batch局部梯度显示当前Q和保持对该HIL探针方向有帮助，混合BC四批冲突；不足以证明全局Q正确。新增工具11测试通过，全部指标有限，Episode级bootstrap/3图/命令SHA保存在projects/rl-platform/outputs/online-rollout-strategy-20261007，正式审计docs/audits/2026-10-07-online-rollout-strategy.md。累计失败draw12.33%但末段22.06%，不能固定理解为8:2。实际served3500、Learner7281/待发布3640；无现场服务/权重/配置/Replay/模型/GPU/动作改变。P0累计发布时钟未修，后续先修再采有位置标记的纠正TRAIN；无独立TEST/自主持续提升证明。Guide仅事实追加，不提交推送Git。


## 2026-10-07：发布准备时序修复已同步现场

源57b378f（最终记录5e5fdcd1cd12e472e8febe5826ff08ff64598ecd）已主项目main/隔离分支推送；把插值/EMA移到等待前、复用发送前控制检查反馈，物理限速检查从发送后移至发送前。保留50ms单次/C10累计保护和暂停/HIL/200ms阻塞停止；partial故障命令写日志，不伪造Replay。合成原40/50Hz正常准备开销累计超时可复现，修复后四频率各120逻辑步通过；A6000相关125、现场实际模块冻结Python3.10 CPU48通过。模型操作锁内同步单模块SHA b04a782b；Stage1 PID2073785/start8759073与19项配置/Replay/权重保护SHA保持。未恢复进程/加载模型/运动/占GPU/发布Actor；真实运行仍待用户按现有入口恢复并验证。证据projects/rl-platform/outputs/publication-preparation-20261007及docs/audits/2026-10-07-publication-preparation.md、MIGRATION.md。guide只追加事实，不提交推送Git。


## 2026-10-07：Online离线准入收尾

rl-platform main/隔离分支8e39f6337dc65965ac66df8288fadbc42b6d03e4已推送且干净。六组CPU三配对seed只改Actor近期51slots采样，Critic批次相同；新单条HIL TRAIN拟合改善1.0–2.63%，前次Critic采样代价明显缩小，仍未采用/发布。实际7281完整状态8vs4+4恢复、pending导出、64状态原生CPU推理及旧预算不重复通过；18专项测试通过。

实际3500/3640在20重复DEV370缓存状态复核两次逐位一致，但自主Q1回报代理变差、辅助p95微增，不能认证全面改善。新工具与3图/六候选/命令SHA/操作单保存在projects/rl-platform/outputs/online-admission-closure-20261007，详见docs/audits/2026-10-07-online-admission-closure.md。本批不改生产资产/配置/现场源码，无服务POST/GPU/运动；前次发布修复b04a782b仍已安装，20:51运行退出/录制完成/Stage1保持、7项SHA相同。可交接首轮受控Online验证，候选暂存，独立TEST/自主持续收益未通过；guide仅追加事实，不提交推送Git。


## 2026-10-08：再次控制发布超时（仍不放行正常Online）

实际新代码b04a782b在Episode10011/step476单次迟到135.2ms退出，之前1190物理发布调用max.622ms，trace append开始→下次prepare154.3ms，尚无历史分段确定全部原因。Learner7281一直caught up，Replay仍2777；录制UUID bbf2aa91-1d12-4013-9f6f-5983cf134f3e明确writer_queue_overflow（523采样/267写入），不完整、不入Replay、不计任务失败率。

源5c46cbb修复控制线程同步USB trace：异步有界数值队列64/最长pending1s、终止flush准入、错误/满队列fail closed，保留发布50ms保护并补fault分段；已先A6000提交push后现场锁内同步4模块，A6000/现场实际环境各85CPU回归通过，Stage1 PID3419827/start17242517与12项配置/权重/ReplaySHA保持。未恢复/加载/运动/GPU，不宣称根因全部消除。HDF5队列溢出仍待录制领域诊断，正常Online不放行。

最终记录d5071030dfab0dfc3ddb078d8e37798bb75b5690见主项目docs/audits/2026-10-08-recurrent-publication-timeout.md；全部证据/时序图/测试/原始失败/安装回执/recording-handoff.txt在projects/rl-platform/outputs/recurrent-publication-timeout-20261008。guide仅事实摘要，不提交推送Git。


## 2026-10-08：发布反馈采样修复及受控Online首轮交接

源58bdc76在A6000隔离分支验证提交push后同步Cobot两个执行模块；发送前的纯反馈/权限检查不再解码RGB，相机/关节新鲜度、HIL/暂停和50ms单次/C10保护保留。3ms三相机合成开销在旧版四频率复现51ms退出，新版各120逻辑步通过；A6000/现场Python3.10 CPU各87回归通过，7项权重/Replay/配置SHA不变。录制修复的timing字段已在实际API可见，前次“尚未生效”不再适用。

本批无服务重启、模型加载、GPU或运动，当前offline/recorder idle、控制mode=fault，硬件成因未诊断，设备正常后才进入首轮受控Online。修复随下一次RLT进程加载生效；首轮完整标注/录制/入库/更新需现场验证，不等同长期连续稳定或自主收益认证，pending Actor不自动晋升。事实、图、命令、保护SHA与回退见projects/rl-platform/outputs/publication-feedback-20261008及docs/audits/2026-10-08-publication-feedback.md。Guide Git不提交/推送。


## 2026-10-09：笔记本归档核对（清理待完成）

按最新目录约定完成遗留审计材料的逐文件大小/SHA256核对与必要补归档，区分 10 月 6 日收据覆盖内容及本次独立核验内容。本地入口已更新；本地删除被执行工具自动审批拒绝，原目录仍保留，不能标记整理完成。无训练资产、Replay、Cobot 服务或机器人变更。来源：主项目 docs/MIGRATION.md 的同日记录；详细收据留所属私有 scratch。guide Git 不由本任务提交/推送。


## 2026-10-09：笔记本目录整改完成

在用户明确选择可恢复清理后，重新核对全部本地文件与 A6000 保留副本，确认无遗漏，旧审计、输出和预览目录已移入 Windows 回收站并逐文件复核，未清空回收站。笔记本 rl-platform 仅保留 AGENTS.md，无本地必需运行文件或本批上传材料，不创建空目录。此前“清理待完成”为历史状态，本次整改已完成；无训练资产、Replay、服务或机器人变更。来源：主项目 docs/MIGRATION.md 同日完成记录；逐项证据保存在所属私有 scratch/rl-platform/laptop-layout-20261009/。guide Git 不由本任务提交/推送。


## 2026-10-10：Online实际采样占比复核

只读复核已注册 supported-online v4 的实际配置、batch身份、Learner指标及Replay SHA：仍是10月7日7001–7281共281次更新，不能冒充今日新增训练。近期Online累计约40.49%，Warmup58.85%；新一轮份额仍受窗口数及共享近期池稀释，单次新HIL人工窗口在随后Actor抽样中仅1.13%。详细边界、逐轮CSV、曲线和SHA见主项目 outputs/online-sampling-20261010/REPORT.txt 与 analysis.json；未改源码/配置/训练资产/Replay或现场服务。guide只追加事实，不提交推送Git。


## 2026-10-10：HIL8分批Online候选已安装

主项目完成3seed匹配单因素与组合对照，选择Actor每批替换8个近期真实HIL窗口＋稀疏终奖目标限幅，保留MC30/保持损失及50HzRTC滤波。新单条HIL TRAIN拟合较匹配baseline改善5.32–6.48%，20重复DEV在预设容差内；接管截断、Critic失败末端quota1/2有保持副作用，未采用。失败终点Q残差、独立TEST/位置OOD/自主持续提升仍未通过，不声称全面解决。

运行源码6e828bb已A6000提交push后Cobot锁内同步9文件；A6000 38专项、现场CPU22专项、原生8vs4+4全状态/双采样RNG恢复、32输入导出一致性通过。网页新增plug-v3-hil8-7281（冻结）及plug-v3-hil8-online（分批Online）available；初始7281/Actor3640/Replay2777，独立candidate/runtime、50步保存、pending不自动晋升。旧7资产SHA与现场服务身份保持，无现场Stage1加载/服务启停/GPU/运动。可进入首轮受控分批Online，不是自主持续增益认证。

详细方法、5图、失败尝试/回退、SHA/命令和现场操作单在所属项目 outputs/online-credit-repair-20261010；正式MD为docs/audits/2026-10-10-online-credit-repair.md及MIGRATION.md。Guide仅追加事实，不提交推送Git。

## 2026-10-11：研究经验库（待合并）

新增 9 条 RLT/plug_v3 研究经验（看动作物理变化量而非拟合百分比、锚定叠加导致越训越不动、跨日真机不可比、关键阶段与逐 chunk 偏差累积、critic 揭晓前窗口、TD 四项保护、动作多样性、残差 RL 与 Stage-1 分工、先确保能入库）。用户要求先记录、暂不提交：当前位于分支 worktree `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/online-review-20261011/rl-platform/docs/RESEARCH_LESSONS.md`，项目 AGENTS.md 的阅读入口同在该 worktree，合并 main 前新对话请直接读上述路径。证据：`projects/rl-platform/outputs/online-review-20261011/checkpoint_review.json`。来源：online-review-20261011 对话；未改生产 Replay/权重/现场。
