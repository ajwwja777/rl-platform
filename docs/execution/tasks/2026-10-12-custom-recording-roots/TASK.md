# Getea1 自选录制目录校验契约

日期：2026-10-12。负责人：cobot-web root；状态：完成。

问题：网页用户选 `/media/agilex/Getea1/xiqing` 被拒绝；网页允许根与 RLT profile_storage 校验不一致。验收：用户授权的 Getea1 根可 browse/prepare/新建，默认路径保持，兄弟路径与符号链接越界拒绝，不启动采集或运动。

本任务仅改 integrations/cobot_runtime/profile_storage.py 读取 COBOT_ADDITIONAL_DATA_ROOTS，不反向 import web；主机允许根由网页配置导出。RLT baseline e781f079，代码542f560e已提交／push并同步；worktree `/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/custom-recording-roots-20261012/rl-platform`。无 GPU／训练资源，全部离线CPU验证；范围外资产、模型状态、Replay及固定探索目录保持。现场重载仅限用户授权的网页，未调用 RLT 启停。验收不符即停止发布、不替换模型。

已验证：网页最终后台758 passed/6既有 skipped、前端84 passed；包含新增5项目录契约覆盖。现场 RLT CPU validate_root 返回 xiqing，storage selection 未改；正式网页 browse/prepare 200，新目录创建后清理，兄弟路径422。7采样模型／硬件进程身份保持；已有探索模型error/未运行保持，不以目录通过声称模型运行通过。

主实现／现场事实见同级 cobot-web/docs/MIGRATION.md 的2026-10-12自选目录记录；证据在 `/data/LFT-W02_data/jiaan/jiaan/projects/cobot-web/outputs/custom-recording-roots-20261012/`，现场旧文件在 cobot-web/runtime/verification/custom-recording-roots-20261012/before/rl-platform/。回退材料为已核验旧源码；本次没有执行回退。下一步：用户刷新网页选择并使用所需目录；没有自动模型加载或采集待办。
