# openpi-RLT R1 训练与迁移报告

## 结论

`R1` 已从固定 π0.5 base 在 `legacy40-v2.1` 上完成 joint VLA+RLT 全量后训练。正式
checkpoint 为零起始计数的 `4999`，不是目录名意义上的“少训练一步”。训练资产仍保存在
HPC 项目目录，部署所需的 `params/assets/metadata` 已逐文件校验后发布到 Cobot。

本报告只证明训练、恢复、部署加载和无动作 online 组件闭环；尚未证明真机任务成功率。

## 固定输入

- upstream：`Yyshadow/openpi-RLT@c1e40ac360185778c98cf20da2820e22d2d415e7`。
- base：固定 `pi05_base`，29 files、12,441,749,581 file bytes；10 个登记大文件 hash 匹配。
- 数据：`legacy40-v2.1`，40 episodes、29,383 frames、30 FPS、三相机、14D state/action。
- 数据内容树 SHA-256：`4bd2987f61900da803c030615b8c612b6547dab8c8219c116a475250aae835a0`。
- norm stats SHA-256：`0e86c542fe27de2018a031c72d493494821697ed9f3a9c976511ee9ddfb1a2e8`。
- run manifest：`methods/openpi_rlt/manifests/r1-joint-legacy40-v2.1-20260904.json`。

## 申请和运行方式

本次使用经当次授权的 `HPC3_jhe724`，Slurm allocation `593480`，节点 `ACD1-33`，
`acd_u/formal-user`，4×H100 80GB、48 CPU、512 GB。流程是项目内 tmux 保护的交互
allocation，而不是在登录节点直接训练：

```bash
project=/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl
socket="$project/runs/r4.sock"

tmux -S "$socket" new-session -d -s rlt4 \
  "exec salloc --job-name=cobot-rlt-4gpu --partition=acd_u --qos=formal-user \
  --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=48 --mem=512G --time=7-00:00:00 \
  srun --gres=gpu:4 --cpus-per-task=48 --mem=512G --chdir=$project --pty bash -l"
```

这只是本次已验证参数的内部复现模板。再次使用 `jhe724` 前仍须获得当次授权，并实时核验
partition、quota、队列和项目路径；不得把历史授权自动沿用。

allocation 进入计算节点后，先验证四卡可见性和 collectives，再运行项目自身入口：

```bash
envs/rlt-stage1-py311-relay-v1/bin/python \
  methods/openpi_rlt/scripts/stage1.py train \
  --project-root "$PWD" \
  --upstream-root "$PWD/code/openpi-rlt" \
  --dataset-root ../cobot-realworld-vla/datasets/canonical/in-the-pot/legacy40-v2.1 \
  --base-params "$PWD/cache/openpi-assets/pi05_base/params" \
  --exp-name r1-joint-legacy40-v2.1-b32w16-s42-20260904 \
  --batch-size 32 --num-workers 16 --fsdp-devices 4 \
  --num-train-steps 5000 --rlt-alpha 1.0
```

正式命令以 run manifest 和远端日志为准；上面不包含资源采样 wrapper 的实现细节。

## smoke、吞吐和正式训练

正式训练前依次通过：真实数据单 batch、joint forward/backward、step 0/1 保存、从 step 1
独立恢复到 step 2，以及四卡通信。probe 结果为：

| global batch / workers | 结果 |
|---|---|
| 16 / 4 | 约 8.7 s/update，吞吐低 |
| 32 / 8 | 约 9.5 s/update，worker 不足 |
| 32 / 16 | probe 稳态约 4.86 s/update，入选 |
| 32 / 24 | 无稳定收益，冷启动更慢 |

正式 run 使用 global batch 32、workers 16、4 devices、5,000 updates、seed 42、BF16、
`rlt_alpha=1.0`。正式窗口从 22:52:55 至最终文件时间 23:48:04，约 55 分钟；包含初始化
折算约 48 samples/s。该正式结果显著快于短 probe，后续估时应以同节点正式稳态窗口为准，
不能只外推冷启动 probe。

资源日志的活动采样显示：四卡显存峰值约 63.7 GiB/卡；活动期平均利用率
97.4%–97.8%；平均功耗约 608–633 W/卡。未选择 24 workers，避免继续增加 CPU/文件系统
压力而无吞吐收益。

## loss 与 checkpoint

- final step：`4999`。
- total loss：`0.2014`。
- RLT/mse：`0.1974`。
- VLA：`0.0040`。
- grad norm：`1.3880`。
- HPC 完整 checkpoint：
  `/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl/checkpoints/openpi-rlt/cobot_rlt_pi05_joint/r1-joint-legacy40-v2.1-b32w16-s42-20260904/4999`。
- Cobot deployment checkpoint：
  `/media/agilex/Getea1/jiaan/projects/cobot-realworld-rl/checkpoints/openpi-rlt/cobot_rlt_pi05_joint/r1-joint-legacy40-v2.1-b32w16-s42-20260904/4999`。

部署包为 36 files、15,428,374,911 file bytes；relay→Cobot 的
`rsync --checksum --dry-run` 无差异。完整 optimizer/train state 只保留在 HPC，Cobot 仅保留
推理和 online 阶段所需资产。

## 环境与迁移经验

1. HPC 的 NFS 与 `uv` rename/lock 流程曾产生 D-state 残留；最终使用原生 Python 加已校验
   site-packages 的环境，并在迁移前检查 GLIBC/GLIBCXX 上限。
2. 大资产先进入项目 `.staging-*`；先核对数量、大小和关键 hash，再用 checksum rsync 发布。
   NFS rename 阻塞时不反复 `mv`，也不强杀 D-state 线程。
3. canonical 数据在共享 HPC 上只读引用，不为训练重复复制；部署到 Cobot 时复制等价小数据集，
   不修改源数据。
4. 环境迁移后要检查解释器链接、editable `.pth`、`direct_url.json` 和绝对路径；能 import 不等于
   能保存/恢复 checkpoint。
5. allocation 属于调度资源，训练结果必须写共享项目目录。用户要求不再训练后应立即释放；
   交互 allocation 被主动取消会在 `sacct` 显示顶层 `CANCELLED`，不能据此推翻已由日志、退出
   状态和最终 checkpoint 独立证明完成的训练。

## 已知方法风险

- fixed upstream 对 14D 物理动作补零至 32D，flow loss 会监督全部 32 维，没有 action-dim mask。
- episode horizon 的 `action_is_pad` 未进入 fixed upstream flow loss。
- 论文报告 π0.6，本项目走公开 π0.5 路线，应称为官方代码路径上的 Cobot adaptation。
- loss 下降只证明优化过程成立，不等价于优于普通 SFT；公平结论仍需同协议的 B1/R1/R2 真机评测。
