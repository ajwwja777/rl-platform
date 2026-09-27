# Plug V3 数据与 Stage 1 训练 SOP

本流程用于 `plug_v3_yyshadow`：右臂单臂动作与状态，中臂、左腕、右腕三路 RGB。原始录制保留 14D，正式训练发布只取索引 7–13 的右臂 7D。

## 状态机

`collecting -> audited -> frozen -> converted -> validated -> transferred -> stats -> stage1 -> offline-gated -> deployable`

任何阶段失败都停在当前状态，不用后续文件的存在代替前一阶段的验证。原始 HDF5 至少保留到 Stage 1 checkpoint、离线推理和数据 lineage 全部验收通过。

## 1. Cobot 最终审计与冻结

原始目录：

```text
/home/agilex/jiaan/data/rlt/plug_v3_yyshadow/demonstrations/recording_tmp
```

质量工具：

```text
/home/agilex/jiaan/project/rl-platform/methods/openpi_rlt/plug_v3_yyshadow/tools/audit_expert_hdf5.py
```

冻结清单：

```text
/home/agilex/jiaan/project/rl-platform/outputs/rlt/plug_v3_yyshadow/collection/expert_selection_frozen.json
```

门禁包括：完整 HDF5、最少帧数、右臂 action/state 一致性、固定左臂、三相机有效性与同步、episode 内部无无效训练帧。只允许裁去连续的无效尾帧。

## 2. Cobot 本地转换与深度验证

转换器：

```text
/home/agilex/jiaan/project/rl-platform/methods/openpi_rlt/plug_v3_yyshadow/tools/convert_experts.py
```

输出：

```text
/home/agilex/jiaan/data/rlt/plug_v3_yyshadow/demonstrations/lerobot-validated-v1
```

固定契约：LeRobot 0.1.0 / v2.1、H.264 CRF10、30 Hz、H=50、7D state/action、三路 640x480 RGB。转换器必须完成：

- Parquet 中全部 7D state/action 与源 HDF5 逐值一致；
- 全部视频可完整解码且帧数一致；
- 代表帧 RGB MAE 不超过阈值；
- 转换结束时重新计算每个 HDF5 与 sidecar 的 SHA-256；
- 只有全部通过才以排他方式写出 `conversion_manifest.json`。

## 3. 最快且可恢复的传输

推荐路径：

```text
Cobot --校内直连 rsync--> A6000 --校内 rsync/scp--> HPC
```

Cobot 到 A6000 的 TCP/22 已验证可达。当前缺少专用免密 key；完成一次专用 key 配置后，从 Cobot 运行：

```bash
rsync -a --partial --append-verify --info=progress2 \
  --exclude conversion_manifest.json \
  /home/agilex/jiaan/data/rlt/plug_v3_yyshadow/demonstrations/lerobot-validated-v1/ \
  LFT-W02@10.12.1.245:/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/data/rlt/plug_v3_yyshadow/plug_v3_yyshadow_demonstrations/
```

视频已经压缩，不加 `-z`。先传全部 payload 并校验，再单独传 manifest：

```bash
scp /home/agilex/jiaan/data/rlt/plug_v3_yyshadow/demonstrations/lerobot-validated-v1/conversion_manifest.json \
  LFT-W02@10.12.1.245:/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/data/rlt/plug_v3_yyshadow/plug_v3_yyshadow_demonstrations/
```

A6000 到 HPC 同样先排除 manifest，完成校验后最后传 manifest。HPC 只经 A6000 访问，不把 HPC 私钥复制到 Cobot。

三端都在数据根执行：

```bash
find . -type f | wc -l
find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

只有文件数和树哈希同时一致才放行训练。若没有 Cobot→A6000 专用 key，可退回笔记本 `scp -3` 中转；它可用但速度较慢且逐文件开销较大。

## 4. Stage 1 训练

入口与配置：

```text
methods/openpi_rlt/plug_v3_yyshadow/train_stage1.py
configs/plug_v3_yyshadow/stage1_training.json
configs/plug_v3_yyshadow/stage1_preflight_hpc.sbatch
configs/plug_v3_yyshadow/stage1_hpc.sbatch
```

固定上游 commit：`c1e40ac360185778c98cf20da2820e22d2d415e7`。配置等价于 `rlt_pi05_agilexbag_image_delta_joint`，只把物理 state/action 收窄为右臂 7D；不改上游 loss、AdamW、CosineDecay、EMA 或 RLT 结构。

正式参数：π0.5 base，动作 horizon 50，模型 action dim 32，前 6 维 delta、夹爪 absolute；RLT token 1、encoder 2 层、embedding/input 2048、alpha 1.0；seed 42；5000 steps；global batch 32；8 workers；EMA 0.99；4 GPU FSDP；每 1000 steps 保存。

HPC 首先在 `debug` 队列用 1 GPU 执行正式数据的 `validate -> stats` 预检。通过后，Slurm 正式训练使用 `acd_u/formal-user`、1 node、4 GPU、48 CPU、512GB、3h，并按 `validate -> stats -> train` 执行；任何一步失败立即退出。

## 5. 训练后门禁

- loss、吞吐和学习率均为有限值，无 NaN/Inf；
- step 1000/2000/3000/4000/5000 checkpoint 可读，最终参数与 EMA 完整；
- 固定验证集离线动作推理通过，输出为 7D、H=50，数值和动作范围合理；
- 与 base/reference 在同一验证 episode 上对比，不以训练 loss 单独声称真机提升；
- 通过后才同步到 Cobot，先 frozen/reference rollout，再进入 warmup 与在线 RL。

## 2026-09-24 首次执行证据

- 冻结：141 原始，134 采用，7 拒绝，19,022 帧；train/val=120/14。
- 转换：542 文件，402 视频，57,066 个解码视频帧；RGB 抽样最大 MAE 1.2584。为兼容 HPC 的 `datasets==3.6.0 / pyarrow==20.0.0`，134 个 Parquet 的向量元数据已规范为 `Sequence[7]`，随后重新完成全量验证。
- 最终 manifest SHA-256：`c1c3fd4ad2ca0d1602d611c99f9ad97c4792a17882aca22224c7d375e76f1171`；payload tree SHA-256：`96e6e2b30016d75fdaf099a256f324398fcd472d0c8326bb2ce033edf49af101`。
- Cobot、A6000、HPC 三端最终完整树哈希均为 `c160a1d239a4ee43247cbf7ba3ddefe1f2b86ca33eb3e0cb71fea19a1574d23f`，文件数均为 542。
- 作业 `646397` 在 stats 门禁发现元数据不兼容并 fail-closed，未开始训练；兼容性修复后的预检作业 `646558` 已完成（594/594 stats batch，退出码 0）；正式 4 GPU 作业为 `646566`。
