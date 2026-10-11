# 探索采集与 W2 Online 就绪复核 — 2026-10-11

结论：新探索方案尚不能直接从网页采集；W2 尚不能作为已验收的边采边学版本。没有修改其他对话代码或现场。

## 已验证

A6000 main 为 48edc2e；另一对话基线 55516fb 的 exploration.py、3 行 runtime hook、AWBC 与分析脚本均未提交。Cobot 探索模块不存在、部署清单与实际只读 API 没有探索模型入口。接口当次返回 offline/process_started=false/model_ready=false/session_active=false；旧 pi05 选择记录不是正在加载 RLT 的证据。

隔离复制 exploration.py 和 6 项 numpy 单元测试，在 A6000 .venv、禁用 CUDA 下重跑：6 passed。验证了偏移边界、夹爪不动、种子、相关性、默认关闭及确定性旁路；未据此宣称真实全链路通过，截图所述 73 项未在本次重跑。

21 组锚定实验与 9 组 AWBC 的 comparison.json 存在且有 7281/8281 最终评估。本次解析 W2 三种子与同种子同步数 A0：重复 DEV 自主成功动作拟合 p95 增加 6.17–12.18%（7281）与 15.08–25.64%（8281）。失败末段 j2 对 Reference 的有符号平均偏移，W2 为 1.214–1.593 mrad（7281），1.113–1.339 mrad（8281）；该 TRAIN 代理不是最优纠正方向/自主成功证据，也不应把 2 mrad 当通用验收阈值。

## 未通过的交付条件

1. 未提交/合并/推送、未现场同步、无探索入口。
2. shared_model_env.prepare_config 调用 evaluation_env.prepare_config，后者强制 actor_deterministic=True（109 行）；探索 hook 在该条件下跳过。因此现有冻结入口仅增加 exploration 字段不会产生探索。需要明确分开 Actor 权重冻结、行为确定性、采集写入及 Learner 启用四个开关，并对真实入口生成配置验证。
3. 探索记录目前在 planner hook 中同步 mkdir/open/write；没有 request_id、logical_step、Actor identity 或异步计划采纳/丢弃对应身份。实际执行动作已有 trace，但偏移日志到真正执行段的可连接性，以及慢盘对时序的影响尚未验证。先补齐/验证后再用于探索归因。
4. AWBC 声明仅离线使用，未接入现场训练入口，没有新的原生恢复/导出/发布回执。旧 DEV 尾部拟合代价需要明确工程保持门槛与选择证据；不是已经通过的新 Online 配方。

## 下一步建议

先完成“固定已知 Actor + 探索开启 + Learner 关闭 + 独立完整采集输出”的专用入口及上述无动作合同验证，提交同步后由现场人员做少量首批采集，核对冻结版本、真实执行动作/偏移/HIL/位置/结果是否可追溯，再扩展批次。W2 作为离线候选继续对照，不因探索采集可用就自动开启在线学习或晋升 Actor。

## 身份与产物

完整源文件 SHA、实际只读查询、6 项测试日志、W2 配对提取结果：`/data/LFT-W02_data/jiaan/jiaan/scratch/rl-platform/exploration-readiness-review-20261011/` 下 review.json、field-files.txt、field-status.txt、unit-test.txt、w2-comparison-extract.json。源数据 journal SHA：029d09050303a6d87b0f40b52ad69ced36c9caf20c9d189e103f52ea53ec2ceb（既有报告身份，本次不反序列化 Replay）。

只读命令：git status/log/diff；rg/static source reads；Cobot python3 读取部署文件和 urllib 禁用代理 GET /api/deployment/status、/api/collection/model；A6000 .venv/bin/python -m pytest -q -p no:cacheprovider --rootdir <unit-copy> <unit-copy>/tests/test_exploration.py。原始实验在 projects/rl-platform/outputs/online-review-20261011。没有真实 GPU 前向、新训练、模型加载、服务启停或机器人动作。
