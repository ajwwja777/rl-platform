# 方法与运行结构

```text
rl-platform/
├── methods/openpi_rlt/           # 现有RLT算法适配
├── third_party/openpi-rlt/       # 固定上游，不合并训练循环
├── integrations/cobot_runtime/  # 采集/评测边界、路径、目录选择
├── configs/methods.json         # 方法接入级别与能力
├── configs/deployment_models.json
├── configs/rlt/                 # 原始超参数及发布清单
├── scripts/                     # 启停、preflight、隔离验证
└── docs/
```

RLT与EXPO-FT分别拥有环境、训练循环、Replay、reward和checkpoint。统一登记、任务控制与原始数据/评测记录，不统一算法内部数据。EXPO-FT仓库保留../expo-ft，Cobot观测/动作适配未完成；methods.json明确禁用执行能力，不能当作RLT模型加载。后续先按原始同步流程接入，async/RTC/EMA单独配置验证。

本批未改loss、reward、数据比例、5000 warmup、更新频率和动作限幅。Stage1 checkpoint索引4999（训练5000步）、Learner5000、Actor2500是三个不同字段。推理快照不是可恢复训练checkpoint；续训必须配套learner/优化器/Replay/normalization。

共享环境从web迁入integrations/cobot_runtime，web保留兼容导入。运行脚本读取本项目配置与显式环境变量。录制HTTP由web提供，是已登记的传输依赖；recorder_url在configs/local.json。未来可独立部署HTTP提供者，但须保留单writer和协议，本批不复制另一份API。

机器模板configs/machine.example.json；目录选择状态runtime/storage-selection.json。迁移只复制旧选择状态，数据实体不移动。
