# 文档框架整理结果

## 交付

已验证：项目采用 architecture/stage1/warmup/online/execution 五模块，每个模块有 README 与 tasks。顶层 docs/README.md 汇总进度、关键经验和当前重点；AGENTS 明确问题→证据→实验/修复→验证→交付的自主推进与多对话边界。小任务仅 TASK，复杂任务按需拆 ANALYSIS/RESULT。

已验证：98 份既有 tracked MD 完成盘点，95 份专题正文归所属模块，旧路径改为兼容页并保留原标题锚点；6 个此前失效链接明确保留历史路径。51 个既有 outputs 目录归模块导航，没有移动或改写报告原件。旧项目 README 与原 guide 完整摘要在本任务内保留；另一对话待合并研究记录保留。

已验证：guide 通用范本位于 `/data/LFT-W02_data/jiaan/jiaan/agent-guide/pipelines/project/DOCUMENTATION.md`，导航在同目录 README.md。项目摘要精简为 38 行，链接各模块、当前任务和待合并草稿。guide Git 未提交/推送，交由框架维护对话。

## 验证与版本

- 基线代码 `55516fbf0410638b1527f45b0306cc847c97f733`；分支 `docs/organization-20261011`。仅 Markdown 进入项目 Git 变更。
- 206 份项目 MD 加 guide 3 文件参与链接检查；本次最终数量和逐项结果以 scratch/validation.json、validation-main.json 为准。95 份迁移正文 SHA 按允许的链接变换全部匹配，两个入口快照保留完整原文，旧页标题锚点保留。
- 465 份 tracked 非 MD 文件、793 份 outputs 小报告/记录、另一对话 3 文件 SHA 均一致；git diff --check 通过。没有因文档变更执行训练/业务回归或重写实验结果。
- 可复现命令：`python3 /data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/docs-organization-20261011/validate_docs.py`；合并后加 `--main` 检查主目录。
- 本任务证据和最终 Git 提交/推送回执：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/docs-organization-20261011/`，发布身份见 `release.json`，完整修改路径见 [MODIFIED_FILES](MODIFIED_FILES.md)。

## 使用与回退

新对话从 AGENTS → docs/README → 模块 README → 任务 TASK 接管，核对 Git/coordination 与实际证据后推进。新增经验归模块，顶层只摘关键变化；旧路径页不再追加正文。

文档布局回退应以本任务发布提交为单位，在独立分支审核 revert；若随后其他对话已写入新布局，先保留/迁回其新增内容。不得使用 reset/clean 删除其他对话工作。guide 若需回退，参照本任务 `guide-before-final.md` 与 `guide-project-navigation-before.md`，先比对并保留后续他人修改。

## 尚未解决的业务问题

证据不足：失败终点 Q、独立 TEST、位置 OOD、自主持续收益与当前现场状态。本次是文档组织交付，不是新模型交付或 Online 放行认证。技术下一步在 [Online 接续任务](../../../online/tasks/2026-10-11-online-learning-evidence/TASK.md)。

另一对话 `online-review-20261011` 的未提交经验/脚本/AGENTS 保持，由原作者决定合并与归位；本任务没有替其完成研究。
