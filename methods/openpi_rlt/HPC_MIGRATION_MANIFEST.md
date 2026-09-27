# HPC 迁移清单（preflight）

**状态：** source/data/base/stats/environment/CPU-preflight verified / allocation pending

## 目标根

`/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl/`

2026-09-04 已创建并核验该项目根；所有新资产均位于该精确的 `jiaan` 子目录。HPC 原先没有 π0.5 通用模型项目，本项目只按固定 manifest 保存训练依赖 cache，不将其宣称为新的模型所有权源。

## Source 与环境

- openpi-RLT upstream：`c1e40ac360185778c98cf20da2820e22d2d415e7`
- `pyproject.toml` SHA-256：`6949516fdfe3472760775b9831d1bc138fa0d955feea504d526cdaba67dd808b`
- `uv.lock` SHA-256：`c572338af893fbfba926ec3f180106e867a9f3c7f017fda4024c86ca67d09229`
- online runtime pyproject SHA-256：`6341334aa24a694a644e1126f60089428c1dc05507abb760824cb1cdba272c71`
- `uv python install` 与不同并发/link-mode 的 `uv sync` 均在该 NFS 上触发 `lock_rename`。最终方案不复制中转机 Python：使用 HPC 原生 Python 3.11.15 创建 venv，仅迁移中转机锁定且验证过的 site-packages。迁移前 638 个共享库的最高符号需求为 `GLIBC_2.28`/`GLIBCXX_3.4.22`，HPC 提供 2.28/3.4.25；迁移后 imports 和真实 batch 实测通过。
- cache、env、logs、runs、checkpoints 全部使用目标项目子目录，不写登录节点 home 或节点临时盘。

## 数据边界

首轮只迁移冻结的 `legacy40-v2.1` 或只读引用其已核验 canonical：`/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-vla/datasets/canonical/in-the-pot/legacy40-v2.1`。实测为 164 files、421,394,882 B、40 episodes、29,383 frames、120 videos、14D、三相机。若发布本项目副本，采用 dry-run、无 `--delete`、不覆盖同名资产，完成后核对 file count/bytes/checksum。

Task5 additional/rollout 仅在 mask/loss 设计通过后作为独立版本迁移，不能静默并入 legacy40。

## 已发布目录

- `code/openpi-rlt/`：固定上游快照，只读使用
- `methods/openpi_rlt/`：本项目 adapter/config/tests
- `envs/_python/cpython-3.11.15-linux-x86_64-gnu/`：逐文件发布并 checksum 无差异
- `envs/rlt-stage1-py311-relay-v1/`：正式 Stage 1 preflight 环境，51,269 files、7,970,105,670 B；初始迁移的 `site-packages` 经完整 checksum 确认等价。随后仅将两个 editable `.pth` 和两个 `direct_url.json` 中的中转机源码前缀定点改为 HPC 正式源码前缀，使环境不依赖中转机路径。
- `manifests/datasets/legacy40-v2.1.json`
- `manifests/dependencies/pi05-base.json`
- `assets/openpi-rlt/{cobot_rlt_pi05_joint,cobot_rlt_pi05_only}/legacy40-v2.1/norm_stats.json`
- `cache/openpi-assets/pi05_base/`：29 files、12,441,749,581 B
- `runs/openpi-rlt/`、`logs/openpi-rlt/`、`checkpoints/openpi-rlt/`
- `cache/`：环境与模型 cache；失败的 uv cache/env 不清理，避免强制处理 NFS D-state

## 首次远端 preflight

1. 读取并遵守 `shared/infrastructure/HPC_JUNJIE.md`（本轮用户已明确授权使用 HPC `jiaan`，但尚未申请 GPU）。
2. 已只读核验 hostname、项目根和数据；首次写入前仍需复核磁盘/quota、目标冲突和他人任务。
3. 核验 GPU 分区与所需卡数后再申请 allocation；登录节点只做轻量文件/环境检查。
4. 先完成 CPU import、数据 metadata 和 checksum；计算节点再做 CUDA/JAX/多卡通信、真实 batch canary 和吞吐 probe。
5. 未完成 adapter 门槛前不启动正式 B1/R1，不从历史 DAgger checkpoint 初始化。

## 当前迁移门槛

- 已完成：中转机完整 29,383-frame norm stats，SHA-256 `0e86c542fe27de2018a031c72d493494821697ed9f3a9c976511ee9ddfb1a2e8`；canonical 源/中转副本内容树 SHA-256 均为 `4bd2987f61900da803c030615b8c612b6547dab8c8219c116a475250aae835a0`。
- 已完成：trainer π0.5 base 的中转副本逐文件 checksum、真实 full-model RLT-only save/resume、HPC base cache 发布与登记大文件 hash 抽检。
- 已完成：HPC source commit、overlay、stats 和 manifest 发布。base 不进入 Cobot checkpoint 根，也不写到 `jhe724` 的 `jiaan` 之外。
- 已完成：HPC Stage 1 环境 ABI/内容核验；Python/JAX/PyTorch/LeRobot/OpenPI imports；fixed upstream clean；40 episodes/29,383 frames/三相机/14D metadata；真实 batch 的三路 `(2,224,224,3)`、state `(2,32)`、action `(2,50,32)`、finite 与零 padding；RLT-only/joint 4-device dry-run。
- 已完成：不设置 `PYTHONPATH` 时，`openpi`/`openpi_client` 分别解析到 HPC 正式 `code/openpi-rlt`；同一环境下 dataset validate 与 joint full-model dry-run 再次通过。四个定点修改文件的 SHA-256 分别为：`_editable_impl_openpi.pth` `7357a33505b32ffc33dddc63780d62214b0ebffde3a2c2d36d288196bfd38906`、`_editable_impl_openpi_client.pth` `6b17b8dee401cf2baaa4025992eab59ebae665e2493aa0cb365b333f8f230bc2`、`openpi direct_url.json` `afcd714f70c5036ebabd8085bce9b4fff06f832915846bc9619beee2a441b14e`、`openpi_client direct_url.json` `b6028f56ac7b5b3c1376505d2a5243b3ce1b4baedacf53e2b18a6d879a4e8`。
- 已完成：在项目 adapter 中显式固定 PyAV，规避 HPC 缺少 system FFmpeg 时 LeRobot 误选 TorchCodec；自动测试覆盖该分支。
- 风险：失败 uv 进程留下 4 个 zombie leader 及 `lock_rename` D-state threads；包装进程已结束，不强杀、不清理占用路径。正式环境不依赖失败目录。
- 待用户明确选择 GPU 数量后，才申请单独 allocation 做 JAX 多卡通信、真实 batch/save-resume 与 worker/batch 吞吐 probe；当前没有提交作业。
