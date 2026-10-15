# Actor11 夹爪反馈修正独立测试版本

2026-10-15；用户授权新增可在 Cobot Web 手动测试的独立版本。

证据：expo-edit001-batch5 回合4/5均Actor11、无HIL、failure；夹持反馈约27mm而目标约0mm，开口指令按4mm/100ms爬升，手臂撤离时尚未释放。原始策略输出未保存，不能证明这是唯一成因。

新增固定Actor11的 original / feedback 两入口；相同edit0.01、RTC5和网络权重，无Learner、Replay更新或训练。原batch5入口保持。Actor11独立只读硬链接留存、189个文件SHA验证，避免源运行后续prune影响对照。

feedback仅当策略目标超过实际开度1mm、上一指令低于反馈1mm时重设张开参考。夹紧目标、六关节限速、4mm/100ms物理张开增量保留。每个未承诺逻辑步使用新鲜反馈重新规划夹爪；RTC前缀在提交前更新，提交后保持不变，最长5步内不重新锚定。插值也从修正后的开度起算；已提交前缀和实际发布端点在完整执行循环测试中一致。反馈超过250ms或非法值拒绝执行，HIL/暂停沿用原门控。

验证：208 passed / 4 skipped（CPU环境下GPU布局用例）；原执行、固定预览、snapshot、在线及传输回归覆盖，未新增GPU训练或机器人动作。未进行现场夹爪/插入验证，不承诺成功率提升。

实现：gripper_feedback.py、gripper_preview.py、ros_runtime.py三处默认兼容hook；脚本expo_gripper_preview_up.sh；模型登记expo-plug-actor11-gripper-{original,feedback}。现场源码交付与HTTP可用性以runtime/verification/gripper-feedback-20261015收据为准。

## 接入最新 batch5 Online

用户报告固定Actor11修正版真机效果很好，明确要求将同一修正用于最新batch5并继续Online。复用FeedbackExecution，batch5启动脚本显式开启--gripper-feedback，原逐回合Online与固定原版对照不启用。训练脚本、超参数、Replay、权重目录、model_id和每5回合更新策略不变；接续当前Actor40/Critic800/17有效回合/online2804/Replay6105/checkpoint3，而非回退Actor11。

新回合started/finished元数据保存execution_profile=gripper_feedback_v1；历史回合不改写。Replay仍从/master/joint_left记录实际协调器输出，不用原模型输出替代限速修正后的动作。212 CPU通过、4 GPU布局用例跳过，覆盖Online真实发布循环的前缀一致性、关闭开关时旧行为、回合标签保存及实际动作不被二次限速；既有冻结预览测试继续通过。现场源码/加载回执归runtime/verification/batch5-gripper-20261015。

2026-10-15现场交付通过：6个运行/文档文件逐SHA与操作锁安装；Cobot现有测试文件版本不同，保留不覆盖。释放已暂停且无Session的Actor11预览，通过正式Web采集模型API加载原batch5入口，50Hz/RTC/滤波保持。Runtime1594980 ready/manual-paused，gripper_feedback=true、execution_profile=gripper_feedback_v1；远端与本地Actor40一致，Critic800/VLA40/Replay6105、双A6000、batch0/5，published0。未开始真实回合、运动或额外训练。Learner committed.json与contract.json安装前后SHA完全不变。详细回执Cobot runtime/verification/batch5-gripper-20261015/{install,acceptance}.json；用户可开始Session继续Online。
