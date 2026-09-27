# 已撤销：现场验收失败，禁止继续此候选

详见STATUS最新现场失败记录。原交付以下保留为历史。

# 2026-09-19 保守在线学习交付：先固定 rollout，再在线更新

## 状态与科学边界

代码、初始权重、版本指针与在线调度已安装到 Cobot；状态 offline_validated_onsite_pending。未执行机器人动作；驻留旧Session仍stopped，下一次rlt_up才加载r7。预加载Stage1未重启。
本版不是RLT论文原样复现：保留冻结Stage1/RTC/RL特征和有界残差actor，学习端改用IQL式state value expectile + factual Q + advantage-weighted BC。原因是直接Q最大化候选在HIL动作方向探针仅33%，不应继续利用未观测动作上的Q梯度。
方法依据：https://arxiv.org/abs/2110.06169 。这里是带选择性成功BC、参考约束与平滑项的残差变体，不宣称完整IQL benchmark复现。
未证明自主成功率上升或持续在线改善。旧重复验证集不是独立盲测。HIL辅助回报不是自主成功概率。

## 现场流程

在 /media/agilex/Getea1/jiaan/projects/cobot-platform，已有CAN、机械臂、相机及plug初态准备不变。

1. ./scripts/ui_up.sh
2. ./scripts/rlt_up.sh --frozen-actor
3. 打开 http://127.0.0.1:8015/，等待ready，现场arm，再开始Session。首次新版本actor=500，rev=r7-versioned-online-v4-20260919，candidate=supported-iql-r1。
4. 固定模式整个Session锁定参数，默认保留有效数据供之后学习；需要纯不记录测试可加 --no-record。rlt_demo.sh现在等价固定模式，不再指向被拒的旧corrective。
5. 验收后网页结束Session或Ctrl+C，然后 ./scripts/rlt_up.sh 启动在线模式。正常结束后不需要 --restart。
6. 在线mode=latest，至少5条新增有效episode且处于waiting_scene时准备/训练。成功和失败可用，aborted/shadow/no-record不参与。新批次只在轮间处理，不能训练时开始下一轮。
7. accepted后下一轮取新版本；一轮内参数不变。rejected保留模型、保留失败批次原因；再有至少5条新数据才重新尝试，避免无止境重复训练。
8. Ctrl+C只结束Session保留模型；./scripts/rlt_down.sh才释放整个后端/模型。

先不加 --explore。该参数仍是每chunk .001标准差的右臂扰动，不是启动RL的开关。

## 部署配置

30Hz异步RTC、chunk10、delay6；右关节速度 .10rad/s、加速度 .9rad/s²、tracking .04rad、Piper关节限位。训练和推理相同滤波，参考模式也使用相同profile便于比较。
保留r6防止游走的相对工作空间检查及360条自主命令上限（累计12秒，不含暂停/HIL时间）。触发是暂停/故障提示，不是自动成功；不会靠加长rollout时间无限走。HIL释放后保持暂停。不是碰撞检测或接触力保证。
正常终局归位仍使用既有all --pose plug，未修改机械臂驱动/CAN/相机。

## 学习配置与数据

- replay v4按注册release位置读取warmup和online，不再把source_facts固定到warmup目录。只连接经时间核验的真实接管边，暂停缺口不臆造连接。
- 采样近似expert30% / HIL20% / success-policy25% / failure-policy25%，HIL掩码与独立BC指标写入metrics。
- gamma=.99每控制tick，奖励保留真实终局标注。
- expectile .7、优势温度 .2、权重上限5；actor lr3e-5，critic/value lr1e-4；raw residual天然最多 .05rad，没有旧corrective的4倍放大。
- 训练只使用实际动作评价Q；actor从有效动作提取策略，不对未观测动作直接最大化Q。因果BC辅助项不混入RL奖励。
- 初始训练5000步、耗时约108秒，按预设验证综合评分选step500；不是默认拿最终step。
- 在线每批步数min(2000,max(100,5×新增train transitions))，完整episode划分保留。既有11条online已纳入初始训练并登记；不重算为新批次。
- 门槛包含有限值、残差上界、BC/平滑/TD不退化；在线综合BC+平滑评分须比当前版改善至少0.1%。离线门槛不是实际成功率保证。
- 在线训练只用CPU4线程，不抢Stage1 GPU；实测复用旧数据的365步更新约7.6秒，不含准备/转换，真实HIL特征准备需额外时间。

## 产物（Cobot）

根 /media/agilex/Getea1/jiaan/projects/rlt：
- runs/plug_v2/learning/supported-initial-20260919/actor.pt，选中500。
- SHA256 9ecccc0152055477a480e2f2aa82920322325994200f91e5d48641485ee1914d。
- 同目录report.json、release-comparison.json、runtime-first-plan-check.json、online-integration-test.json。
- runs/plug_v2/learning/v4/current.json：唯一已接受版本指针，原子更新；immutable checkpoint路径及hash。
- runs/plug_v2/learning/v4/cycle.json：consumed/rejected UUID与批次状态；发布后崩溃可从版本report恢复去重。
- runs/plug_v2/learning/v4/batches：每批独立权重/报告。旧模型未覆盖。
- methods/openpi_rlt/plug_v2/{online_contract,supported_learner,online_release,online_cycle_v4}.py；runtime/cli/training_flow接入。
- runs/plug_v2/deploy-backup-online-v4-20260919：改动前源文件及旧demo包装。
- 历史online_learner.py是直接Q诊断实验，生产v4路径只调用supported_learner.py。

## 验证

10项learner/发布/调度测试 + 13项runtime/RPC测试通过，包含真实checkpoint的隔离发布/选择、拒绝不改变指针、崩溃去重、HIL/暂停图、Torch/NumPy滤波一致。
真实actor HTTP恢复输出验证通过；实际Runtime.infer对保存场景首计划约0.121秒，边界检查通过，暂停清空队列。该检查没有构造RTCIO，没有ROS初始化/订阅/发布，不是闭环真机验收。
完整准备→恢复训练→门槛→发布演练使用旧数据、mock发布；600/865候选评分未改善，被拒且真实发布次数0。不把这次演练称新在线学习或进步。
同一验证batch/部署滤波：
reference factual BC=.1615247、human raw BC=.0804732、smooth=.000471340；
旧warmup20000=.1593099/.1244576/.000489289；
新supported500=.1551007/.0725658/.000456527。
相对reference分别约改善4.0%/9.8%/3.1%，相对旧warmup约改善2.6%/41.7%/6.7%。指标为按.05rad归一化的动作误差/加速度，不是成功率。
