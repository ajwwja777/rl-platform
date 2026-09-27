## 2026-09-18：plug_v2 专家训练集转换及原始文件清理已完成

- 用户确认82条全部成功，仅录近插孔准备位置到插入动作。原9120帧/30Hz；派生83片段/9119有效帧，episode_000077第134坏帧切断，不跨断点，不插值填补。
- 原始UUID固定74训练/8验证，无同源片段泄漏；派生75训练/8验证。完整H50窗口5093；49帧和8帧两短片段完整保留，完整H50采样不使用，后续其他采样需显式处理padding。
- 正式训练集：Cobot /media/agilex/Getea1/jiaan/data/rlt/plug_v2/demonstrations/lerobot，LeRobot v2.1，固定已有package0.1.0，H264 CRF10；最终573,780,744 B（573.78MB）。图像为高质量压缩，非原RGB逐像素无损。
- 验证：249视频全量解码27357图像帧；抽样RGB最大MAE1.13293/255；逐帧state/action精确一致，原始所有非图像dataset（含NaN及字符串）精确归档；产物及源文件hash、source inventory、可见打开句柄检查通过。
- 已按用户明确要求删除本批82个HDF5（25,302,369,106 B）和6个重复preview.mp4；该录制根剩余HDF5=0。未删除训练视频、其他项目数据或权重。删除后官方reader仍可读取9119帧/83片段/三相机/H50序列，坏帧前后及验证split实读通过。
- 历史条数：同目录recording_summary.json；conversion_manifest.json、splits.json、selection_manifest.json、artifact_checksums.json、deletion_receipt.json及source_metadata/保留来源/成功确认/节点/审核/划分/原始非图像事实，不依赖HDF5统计。
- Cobot运行证据：projects/rlt/runs/plug_v2-data-conversion-20260918，conversion.log/deletion.log、post_delete_reader_validation.json、test_validation.json。转换PID329416与删除PID360657均已完成，临时真实测试数据已清理，证据脚本保留。
- HPC作业629492最近仍PENDING / Priority；最新预计2026-09-18 17:51:49（ACD1-18只是调度预选节点，非已分配节点），已超过原10小时等待范围；估计会变化，申请保留，无GPU占用。此快照优先于此前08:36估计。
- 下一步：将已验证训练集迁入HPC独立plug_v2路径、配置只用train split的归一化及训练/验证入口，核验已有环境并优化读取效率；尚未迁移新数据、尚未启动训练/服务/机器人动作，未提交/push。


执行边界：本次用户明确要求转换，并在确认不需要网页回放后授权删除源HDF5。只处理上述plug_v2/demonstrations清单，不启用普通网页自动转换删除，不复用旧A100、旧Slurm allocation或历史GPU端口。

转换使用Cobot已有.venv-server解释器，仅CPU操作，CUDA_VISIBLE_DEVICES为空；未安装或修改历史环境。固定83片段保留每条有效连续区间，episode_000077拆为[0,134)及[135,143)。原始capture侧车未改写；基于用户成功确认，通过原SegmentReviewStore为82条已有区间写新审核版本。

转换启动版本与删除检查加固版本的代码及运行记录均保留在Cobot运行证据目录。产物逐帧、源事实逐dataset和源文件SHA256检查后才独立删除；错误产物/源/可见活动句柄拒绝场景在临时fixture实测通过。HDF5删除是不可逆源格式清理，H264压缩不能还原原始RGB逐像素数据；这是用户不保留回放/原始大型文件的明确选择。后续专家预灌应使用LeRobot及source_metadata，不调用依赖已删除HDF5的历史输入流程。

LeRobot库meta统计覆盖整库，不能直接用于训练归一化；需只按train UUID/片段及实际动作变换重新计算。全部source成功不是新模型成功率，尚未训练或真机验证新模型。
