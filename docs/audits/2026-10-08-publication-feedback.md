# 2026-10-08 发布前反馈采样修复

**结论：修复一个可确定复现的累计超时机制，保留原50ms保护；允许下一次先做1轮受控Online链路验收，不宣称长期稳定或自主学习增益已通过。**

## 证据

|事项|状态|直接证据及边界|
|---|---|---|
|原始残余故障|失败|10-08 18:09:20，model-20261008T175554.log，evaluation.yaml，Actor3500，logical_step618，C10累计50.143ms停止。不是Learner争用故障。|
|33.03ms录制检查阻塞控制线程|证据不足|源码显示report_chunk在独立health_pool；last_recorder_check_ms是后台耗时。阻塞其后台worker的回归仍能完成25物理发布。不能当直接原因。|
|发送前权限/反馈检查触发图像解码|已验证|旧wait_active→sample→RosTask2IO.sample_control，每遇新帧会在控制锁内转换3路RGB；每次发送后的累计顺延保护会计入这些非必要耗时。|
|这一机制能独立导致累计超时|已验证|基线d507103、真实RosTask2IO采样方法+合成消息/确定性时钟，注入每相机1ms、每次检查新帧；20/30/40/50Hz分别只完成8/8/5/5逻辑步即51ms停止。是受控压力复现，不是对真实相机到达过程的重建。|
|候选解决相同压力|已验证|仅改发送前采样路径，四频率各完成120逻辑步、物理120/180/240/300命令；无压缩发布间隔，逐行trace/action一致，模型请求与逻辑Replay仍含完整图像。|
|现场CPU开销|已验证|相同Python3.10，3×480×640RGB，500次交替合成调用：全采样median .5647/p99 .6304/max4.378ms；纯反馈median .0360/p99 .0448/max.1011ms。使用cv2色彩转换，不冒充真实cv_bridge/ROS并发或硬实时上界。|
|所有历史/未来超时均被消除|证据不足|原故障没有逐次解码回执，存在调度/反馈/储存争用等其他因素；尚未运行真实Stage1/Learner/机械臂并发。|

数据身份：合成控制消息/随机图像，不使用训练、开发或测试集衡量模型能力，不计算任务成功率或以窗口独立性给置信区间。真实故障仅来自已有日志，只读；原录制/HIL/Replay/权重没有改写。

## 实现与保护

RosTask2IO新增sample_publication_control，复用sample_control的相机/关节存在、时间戳过期/未来/同步偏差、finite state检查，保留mode/paused/outcome及反馈证据；只跳过RGB转换和缓存更新。此返回值仅用于物理发送检查，不能作为模型或Replay观测。原sample_control默认完整采样不变。

AsyncExecution.wait_active使用轻量采样；终止信号仍锁存到逻辑终结，暂停/HIL前后复查和发送时IO仲裁保留。真实200ms反馈停顿、发布阻塞、累计慢速、过期观测、错误安全钳位依然停止，不取消或增大50ms单次/C10限制。不改频率、RTC/EMA、Actor/Reference、训练/采样参数或Replay语义。

新增diagnostic publication_sample_mode及全观测/控制采样耗时；recorder_check_thread明确为background。

## 验证与发布

A6000 Python3.11和Cobot实际Python3.10 CPU各87 passed，包含新16项与既有执行、HIL/暂停、原生EnvDriver/临时Replay及两步CPU Learner组合回归。初次新增测试4项因“零顺延时统计字典无该键”的断言错误失败；改为核对真实clock_window_shift_sec，保留失败日志；没有为通过测试改宽守卫。

代码在独立fix/publication-feedback-20261008工作区，基线d507103。先提交push，再确认模型/Session/录制空闲后锁内同步两模块、逐SHA校验；源同步和模型启动分别记录。Cobot仅运行隔离测试/合成benchmark，没有模型加载、服务重启、机器人动作、现场GPU或Actor发布。

完整命令/环境：test-command.json、run_field_tests.py、sampling_benchmark.py、compare_timing.py。产物根：`/data/LFT-W02_data/jiaan/jiaan/projects/rl-platform/outputs/publication-feedback-20261008/`。source-release.json与field-sync.json给出最终代码SHA、保护资产和备份。图01-publication-feedback.png为基础设施诊断。

## 下一步边界

录制器timing已在真实API出现；本次RLT两模块在下一次新加载运行进程生效，不需要为它重载网页。用户可按原Supported7k交付操作做1轮有明确结果的受控Online采集；核对录制完成、输入审计、Replay新增、更新预算和候选checkpoint后再扩大至3–5轮。执行Actor3500保持，训练写pending Actor，不自动发布。不是承诺本批一定成功或算法已获独立TEST增益。

再出现deadline、录制积压/overflow、进程异常、输入身份不一致即停止该轮并留证；不重标不完整数据为成功。故障日志现在区分全观测与发送反馈采样开销、后台HTTP、trace积压。回退按field-sync.json的两个文件原件，保留前序trace/录制修复，禁止替换权重/Replay来掩盖故障。
