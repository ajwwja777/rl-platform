# 插孔 π0.5/RLT R2 明日训练入口

状态：`framework prepared / no real data / no training launched`

## 今天已准备

- `configs/plug_insertion_r2.yaml`：确定性 eval、左臂主动、夹爪 hold、双臂限速/加速度、128 transition 最小预灌和 2,000 update 上限。
- `configs/plug_insertion_stage1.yaml`：固定 π0.5 base 的 joint VLM+RL-token Stage 1 候选；2-step smoke、seed 42、候选 1,000 update。
- `manifests/plug-insertion-dataset-template.json`：真实采集前模板，不是假数据 release。
- `scripts/build_plug_training_release.py`：只接受 `status=frozen` 和人工审核区间，生成不跨区间的 chunk/stride transition plan。
- `scripts/prepopulate_expert_replay.py`：当前只做 `--dry-run` 预灌计划；真实 `z_rl/ref_chunk` 未生成前不会写 replay journal。
- `scripts/prepare_plug_training_run.py`：生成 `prepared-not-trained` run manifest、validate/smoke/full 命令和冻结门。

## 数据到位后的顺序

1. 在 Cobot segmented-teach 页面录 1–3 条 pilot，核对 node 图、三相机、14D、暂停裁剪、HDF5/sidecar UUID 和 review 区间。
   同时现场确认夹持插头的物理臂；配置里的 `active_arm: left` 目前只是候选，不得在确认前当作冻结事实。
2. 正式采集后固定 dataset ID、episode UUID、selected intervals、帧数、三相机/action schema 和源文件 SHA-256；状态从 `collecting` 变为 `frozen`。
3. 生成 transition 计划：

   ```bash
   python methods/openpi_rlt/scripts/build_plug_training_release.py \
     --dataset-manifest <frozen-dataset.json> \
     --chunk-len 10 --stride 2 \
     --output <run-root>/manifests/expert-transition-plan.json
   ```

4. 只读检查专家预灌规模：

   ```bash
   python methods/openpi_rlt/scripts/prepopulate_expert_replay.py \
     --transition-plan <run-root>/manifests/expert-transition-plan.json \
     --min-transitions 128 --max-updates 2000 --dry-run \
     --output <run-root>/manifests/expert-prepopulate-plan.json
   ```

5. 将审核区间转换为 LeRobot release 并冻结 tree hash。当前真实 sidecar 样本尚未产生，因此
   `task5_segmented_teach.converter.selection` 到 LeRobot 的最后字段映射必须在首条 pilot 后补验，
   不能现在凭模板猜测。
6. 生成训练 run manifest：

   ```bash
   python methods/openpi_rlt/scripts/prepare_plug_training_run.py \
     --dataset-manifest <frozen-lerobot-release.json> \
     --dataset-root <lerobot-root> \
     --project-root <remote-cobot-rl-project-root> \
     --upstream-root <fixed-openpi-rlt-root> \
     --base-params <fixed-pi05-base-params> \
     --run-id plug-r2-s1-<date> --full-steps 1000 \
     --output <run-root>/run-manifest.json
   ```

7. 依次执行 manifest 里的 validate、真实单 batch、2-step save/restore、batch `[8,16,32]` 与 worker probe。候选 1,000 update 不是冻结参数；只有 held-out loss、动作连续性、吞吐和资源稳定后才确定正式步数。

## 冻结与后续在线阶段

- Stage 1：从 π0.5 base 同时训练 VLM 与 RL-token 模块，建立新数据上的 reference；不继承现有 version 10000 actor。
- Stage 1 验证后：参数名审计必须证明 VLM、RLT encoder/RL token 全部冻结。
- Stage 2：只更新 actor/critic；每个 episode 固定一个 actor version，episode 内不热切换。
- Eval：无探索；Explore：仅主动臂 chunk 级相关扰动，夹爪与被动臂无噪声。
- success/failure/aborted、HIL mask、actor version 和 Task5 UUID 必须逐 episode 对账。

## 仍未完成的门

- 没有真实新数据，所以没有 LeRobot release、norm stats、真实 batch 或 checkpoint。
- 专家 transition 目前只有 plan builder；Machine A feature 物化与新 replay journal 要在真实数据后完成。
- 未访问 HPC、未申请 GPU、未启动训练。
- R2 overlay 尚未发布 Cobot，也未做修复后的 zero-publisher shadow。
- 自动 reset 仍关闭；轴向退插 waypoint 未经现场验证，禁止从插入终点直接 front home。
