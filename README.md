# RL 平台

维护 RLT / EXPO-FT 方法、Replay、采样、训练、Actor 发布与 RLT 在线执行。A6000 是主代码、Git 和文档位置；Cobot 是现场运行副本。

先读 [AGENTS](AGENTS.md)，再读 [当前进度、经验与模块导航](docs/README.md)。详细结构见 [架构模块](docs/architecture/README.md)，现场操作见 [执行模块](docs/execution/README.md)。

```text
rl-platform/
├── AGENTS.md                      # 协作、边界、自主推进
├── README.md                      # 项目与实现入口
├── docs/
│   ├── README.md                  # 全项目进度、关键经验
│   ├── architecture/              # 架构、接口、环境、迁移
│   ├── stage1/                    # 示范、输入、Reference、表征
│   ├── warmup/                    # 初始化、训练、候选选择
│   ├── online/                    # Replay、采样、学习、Actor 发布
│   └── execution/                 # Session、时序、RTC、现场交付
├── methods/openpi_rlt/            # RLT 适配与独立可选实验
├── third_party/openpi-rlt/        # 固定上游
├── integrations/cobot_runtime/    # 采集/评测与运行边界
├── configs/                       # 方法、实际配置和模型登记
├── scripts/                       # 工具与验证入口
└── outputs/                       # 既有实验产物，不随 MD 整理移动
```

每个文档模块有 README，具体问题放模块内 tasks；简单任务仅 TASK.md，复杂任务按需增加 ANALYSIS.md/RESULT.md。旧 MD 路径仅保留兼容跳转。

| 内容 | 实现入口 |
|---|---|
| RLT 运行/学习适配 | [cobot_adapter](methods/openpi_rlt/cobot_adapter/) |
| 可选算法/采样实验 | [experiments](methods/openpi_rlt/experiments/) |
| Stage1 与右臂环境 | [plug_v3_yyshadow](methods/openpi_rlt/plug_v3_yyshadow/) |
| 采集、评测和目录接口 | [cobot_runtime](integrations/cobot_runtime/) |
| 模型选择 | [deployment_models.json](configs/deployment_models.json) |
| 工具/回归 | [scripts](scripts/) / [tests](tests/) |

A6000：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform`。Cobot：`/home/agilex/jiaan/project/rl-platform`。数据/权重通常在 `/media/agilex/Getea1/jiaan/{data,model}`；已登记的 NVMe Stage1 例外按实际 manifest，不自动迁移。EXPO-FT 保留独立算法、环境和资产，接入能力以实际配置为准。

共享 RTC/滤波归 VLA，页面/API 恢复归 web，录制/mask 归 dagger，硬件归 control。RLT 不反向 import web Python，录制使用 HTTP。完整历史见 [迁移记录](docs/architecture/MIGRATION.md)；模型是否正在运行及机器人能力必须另取当前证据。
