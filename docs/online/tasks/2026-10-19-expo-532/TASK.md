# EXPO SFT2000 独立 50/30/20 模式

用户要求新模式从SFT2000开始，示教50%、在线成功30%、在线失败20%，每5有效回合触发更新。基线646bd33，worktree scratch/rl-platform/expo-532-20261019。仅新增模式，不加载模型、不启动Session或机器人动作，不改旧实验Replay/权重。初始沿用25条示教3301步；在线Replay为空，优化器/更新计数从零建立，不接续Actor166或121。

实现：outcome_sampling.py按来源前缀和finished.json整回合outcome分池，成功不是逐帧reward>0；未知/放弃回合沿用训练排除。每1280条为640示教、384在线成功、256在线失败；每64条32示教，成功19/20、失败13/12循环分配。分池内均匀有放回，排除在线跨回合backup窗口；不是改变磁盘数据数量。缺任一池时不训练、保留额度，后续五回合边界再检查。基础Actor成功样本规则保持。K30、UTD20、warmup10保持，首个正常触发在第15个有效在线回合结束。默认d5/edit.01，参数对独立目录机制保持。

验证：149 CPU测试通过，包含真实Learner调用链配方路由和提交去重；原生Replay320条gather/OpenPI转换实际160/96/64，恢复RNG逐数组一致。未验证新配方GPU训练或自主成功率收益。旧现场Session仍active，本次无生命周期操作。

源码配置configs/expo_532_batch5/{learner,actor}.json为明确来源；新目录outputs/expo-532-batch5-sft2000-20261019/online和Cobot native_remote_532_batch5_sft2000_20261019，采集目录expo-532-batch5-sft2000。这是Online训练入口，固定checkpoint仅部署功能不在本次实现中。发布/网页验收结果随后追加。
