## 2026-09-18：plug_v2 HPC 训练准备通过，尚未开始训练

- 新左视角82条成功插入专家数据已迁至HPC，670文件/573,780,744 bytes逐文件SHA256一致；83派生片段/9119有效帧，整条UUID train74/val8、完整H50窗口4610/483。
- HPC作业629492在2026-09-18 01:10:07获配ACD1-11；GPU0–3均H100 80GB HBM3，初查显存1MiB/卡、利用率0%；8小时allocation只保留shell，不自动训练。此实测优先于旧17:51排队预测；动态资产后续仍需核验。
- train-only归一化统计完成；73个有效片段与真正Cobot数值transform逐个一致。两组实际完整输入链（3张224RGB、tokenizer/discrete state、H50×32动作、统一统计）均通过CPU检查，无模型初始化/训练。
- 两组独立准备：pi05_reference用固定上游train.py；rlt_stage1用train_rlt.py、alpha1。已确认alpha0在train_rlt会冻结VLA，只训练token重建，不能作为纯pi05训练。
- 配置seed42/batch32/FSDP4/workers8/H50；计划先2000更新审核，再分别续到4000。AdamW/梯度裁剪1及cosine默认保留；peak LR2.5e-5、LR warmup1000（不是RL warmup）、EMA0.99。训练时长待获准20步probe测量，不凭CPU检查推断。
- HPC新根内 data/rlt/plug_v2/demonstrations/lerobot、configs/plug_v2、runs/plug_v2；入口./scripts/train_plug_v2.sh reference|rlt。新代码methods/openpi_rlt/plug_v2，旧run/model/env不覆盖。旧actor/critic/warmup不复用，原始HDF5已验证删除。已清除A6000本次573.78MB中转副本，校验/过程记录保留。
- 下一步：用户确认开始后，在已分配节点测吞吐/显存，再顺序训练对照组与RLT Stage 1；不启动真机、服务、自动warmup或在线RL。本轮不commit/push，其他项目dirty修改保留。

## 资产和验证证据
HPC：/data/user/jhe724/jiaan/research-workspace/projects/cobot-realworld-rl。
- runs/plug_v2/preparation.json、preparation.log：上游commit/数据校验、train-only统计、73次真实数值transform对照、真实PyAV reader。
- runs/plug_v2/input_validation.json、input_validation.log、validate_inputs.py：两组配置均加载SHA256 373d9a01bbcc0907dbbc27b091768774f98933cb28cdf3404a14f378ee02b49d统计；实际resize/tokenize/normalize/pad输入链，索引0/2305/4609通过；无GPU训练。
- configs/plug_v2/transfer_checksums.json及两组JSON配置；methods/openpi_rlt/plug_v2/train.py、selected_anchor.py、README.md；scripts/train_plug_v2.sh。
- runs/plug_v2/resource_snapshot.json：H100初始只读资源快照；tmux socket runs/pv2.sock/session plug-v2持有allocation。
- A6000 scratch/plug_v2-hpc-preparation-20260918/{transfer.log,transfer_receipt.json,preparation_receipt.json,relay_staging_cleanup.json}：转移与删除本次中转副本的证据；HPC/Cobot发布数据未删除。
Python AST与bash -n通过。环境未安装/升级；固定openpi-rlt上游c1e40ac360185778c98cf20da2820e22d2d415e7且Git clean。既有torchvision视频API弃用warning不影响已验证PyAV reader。初次共享盘依赖读取较慢；修正共同统计目录时只替换本次CPU准备进程，重算核对相同统计，最终完整输入链通过。无warmup/在线RL/GPU probe/训练/checkpoint创建。
