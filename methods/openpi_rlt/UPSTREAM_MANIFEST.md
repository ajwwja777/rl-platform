# openpi-RLT 上游清单

## 固定版本

- Repository: `https://github.com/Yyshadow/openpi-RLT.git`
- Branch: `main`
- Commit: `c1e40ac360185778c98cf20da2820e22d2d415e7`
- Local read-only reproduction clone: `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/openpi-rlt/upstream`
- 2026-09-04 核验：clean worktree、无 submodule、约 224 MB。

## 环境边界

- Root/Stage 1：Python `>=3.11`，lock 固定 JAX CUDA12 `0.5.3`、PyTorch `2.7.1`、Transformers `4.53.2`、LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。
- `rlt_online_rl`：Python `>=3.10,<3.11`，必须使用独立环境。
- 上游 `scripts/train_rlt.py` 将 JAX compilation cache 指向 `~/.cache/jax`；本项目运行时设置 `JAX_ENABLE_COMPILATION_CACHE=false`，避免在用户主目录产生任务资产。
- `UV_CACHE_DIR`、`XDG_CACHE_HOME`、`HF_HOME`、`TORCH_HOME`、`WANDB_DIR`、checkpoint/log/run 均显式位于 `/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/`。

## 代码审计事实

- `debug_rlt` 配置为 RLT-only（`alpha=0`）。
- `debug_rlt_joint` 配置为 joint VLA+RLT（`alpha=1`）。
- Stage 1 reconstruction 路径对 VLA embedding 使用 stop-gradient；可训练参数过滤由配置决定，仍需以实际 smoke 的参数与 loss 证据验证。
- online runtime 包含 inference、replay、trainer、manual signal bridge、trace-to-replay 与测试；这些仅证明上游组件存在，不证明已适配 Cobot。
