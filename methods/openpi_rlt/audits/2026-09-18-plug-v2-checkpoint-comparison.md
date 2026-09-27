# plug_v2 500–20000步候选保留与离线对比

## 完成与范围

用户明确授权补跑20k并同步离线验证。Cobot4090、相同seed42/目标/真实v2缓存，2026-09-18 12:39:45–12:42:45 UTC，优化177.38秒。保留100/500/2000/5000/10000/20000完整actor/critic/optimizer/target，每个36943083bytes。重跑500 actor全部权重与原部署500逐值一致（最大差0），指标完全复现。所有候选通过原固定512留出批次门槛；完整23个heldout制品1084行另做组别诊断，不能混同原抽样指标。原生产actor SHA未变，Session仍stopped/paused/actor500，未自动切模型/启动机器人动作，未访问HPC/commit/push。

执行目录 `/media/agilex/Getea1/jiaan/projects/rlt/runs/plug_v2/learning/checkpoint-comparison-20260918T123855Z`。原始training.log、operation/launch/ready.json、checkpoints/step_N.pt、offline-comparison.json、summary.json、completion-receipt.json；全量离线分析脚本analyze_candidates.py可复查。运行PID1716714/start_ticks8622165已完成。

## 对比结果

| Critic步 | 原抽样留出BC | 原抽样留出TD | 全量失败状态条件化改变量均值rad | 最后3有效窗口改变量rad | 有监督HIL动作Q高于reference比例 |
| --- | --- | --- | --- | --- | --- |
| 100 | .0002205 | .06168 | .0006378 | .0009360 | 41.2% |
| 500 | .0002145 | .02296 | .0003443 | .0005174 | 49.6% |
| 2000 | .0002269 | .003895 | .0002248 | .0003435 | 54.0% |
| 5000 | .0002350 | .004941 | .0003027 | .0003517 | 47.6% |
| 10000 | .0002372 | .005471 | .0001717 | .0001936 | 46.9% |
| 20000 | .0002366 | .004950 | .0002088 | .0002925 | 47.3% |

后两列是固定缓存状态的配对诊断，不是真机轨迹/成功率；最后窗口仅为每episode最后3个有效decision，未标定为接触窗口。Q偏好比较仅正向BC mask行的录制HIL动作与同状态条件化reference，使用min(twin Q)，不是成功分类准确率，缺少reference反事实结果。

20k较500：失败实际纠偏均值小39.4%，98.7%有效关节动作差<.001rad（500为92.1%）；右臂约29.8%输出坐标触发物理clamp。确定性失败留出状态的目标梯度诊断：reference_anchor加权梯度L2 .04435，Q加权梯度 .002080，约21.3倍（500约6.7倍）。此为相对于原始actor动作、经条件化后的局部梯度，不是全训练参数梯度平均，训练reference dropout随机性未计。证据支持当前目标逐渐把失败纠偏拉回reference，增加步数未解决。

20k critic不是完全没学到：HIL成功中能沿真实successor到终局的540行，Q与观察折扣回报MSE从500的.06312降到20k的.006288；专家439行从.07474降到.01040。但动作优劣偏好没有相应改善。失败heldout33行全部终局未连接/truncated，没有完整MC回报依据，不伪造failure终局来做校准。

原抽样smooth：500=.0000087687，20k=.0000094397。全量成功组20k smooth=.000005041/reference=.000003355，增加约50.3%；专家组增加约38.8%。所以原混合抽样门槛通过不代表每组都满足30%门槛，应保留这一限制，20k仅供受控固定模型对照，不称效果更好。

## 交付代码与测试

learning.py新增可选 `--output-dir`（限定learning根、新warmup、不覆盖已有训练）与 `--save-checkpoints`，不改本轮loss/sampling/默认发布方式。新增方法checkpoint_selection.py及唯一平台薄包装scripts/rlt_select.sh，跨平台修改仅此选择入口，无CAN/机械臂控制代码更改。

选择要求结束Session、持learning lease、匹配候选SHA/cohort/Stage1/stats及accepted状态，先备份生产actor/ready，再原字节原子替换。正在运行/暂停episode不能切换；新Session才加载。选择器5项fixture测试通过：活动Session拒绝、lease互斥、坏hash拒绝、完整20k模型与ready一致、旧模型备份；真实生产模型未修改。默认warmup100/online10 CPU发布与optimizer/target resume测试通过，包装bash-n/--help通过。completion-receipt记录源码SHA：learning=6c920ea19fbeb45affb1dc807768748ace577de7f9be982f7670270521a555a4，selection=b4363068302c41b6cfe4c967af308c05ce27b271b2068993ec55b46182d7795b。

## 现场可切换对照

先结束Session。在 `/media/agilex/Getea1/jiaan/projects/cobot-platform`：

```bash
./scripts/rlt_select.sh 20000
./scripts/rlt_up.sh --frozen-actor --no-record --restart
# 对照回500：结束Session后
./scripts/rlt_select.sh 500
./scripts/rlt_up.sh --frozen-actor --no-record --restart
```

无需rlt_down，网页应显示所选actor_version。选择warmup不会覆盖已有online actor；默认latest若存在online仍会优先它，因此此次对照明确--frozen-actor。同位置自主测试，HIL单列，不保证20k成功率更高。

下一步建议：先降低失败reference_anchor的独立权重做单因素离线消融，保持正向示教/物理限速，进一步检查boundary残差惩罚与Q动作排序；不直接提高explore或仅加训练步数。未启动这项新消融。
