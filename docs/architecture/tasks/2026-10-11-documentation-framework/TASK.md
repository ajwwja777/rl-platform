# 按模块组织文档与任务闭环

状态：已完成；用户 2026-10-11 确认。仅 MD 整理，不执行历史训练/部署任务。

## 问题与验收

问题：项目分析分散，guide 历史追加过长，多对话难以找当前结论与下一步。
验收：五个实际模块具备入口；历史正文按归属保留，旧链接兼容；顶层只保留进度与关键经验；模块下任务可接续；guide 提供通用范本；全部变更仅 MD，保护代码/配置/资产/其他 worktree。

## 范围与身份

基线：`55516fbf0410638b1527f45b0306cc847c97f733`。
分支：`docs/organization-20261011`。
worktree：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/docs-organization-20261011/rl-platform`。
coordination：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/coordination/docs-organization-20261011.md`。
资源：A6000 文件读写/CPU，未连接 Cobot，不加载模型、不启停服务、不占 GPU。

允许：项目 MD、guide 项目摘要及通用范本/导航。禁止：代码、配置、outputs、数据、权重、Replay、其他 worktree；guide 不提交/push。
停止条件：发现内容遗漏、链接回归、非 MD 变化、其他对话文件被改变或合并冲突，先修复/协调，不覆盖。

## 进度

已完成：98 份既有 MD 盘点、465 份非 MD Git 文件和 793 份小产物 SHA 记录、其他对话 3 文件保护；95 份专题正文归位并保留兼容页；已有失效链接保留原路径并明确标记。
已完成：五模块导航、顶层进度/经验、guide 通用范本/导航/精简摘要、文档与保护 SHA 校验。
下一步：后续工作按所属模块任务接续；本次无待执行训练/现场动作。项目 Git 发布身份见 RESULT 所列回执，guide 留待所属对话提交。

详细逐文件映射、失效旧链接与验证见 [ANALYSIS](ANALYSIS.md)；完成交付见 [RESULT](RESULT.md)。
