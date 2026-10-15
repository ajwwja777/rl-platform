# Actor11 夹爪反馈修正独立测试版本

2026-10-15；用户授权新增可在 Cobot Web 手动测试的独立版本。

证据：expo-edit001-batch5 回合4/5均Actor11、无HIL、failure；夹持反馈约27mm而目标约0mm，开口指令按4mm/100ms爬升，手臂撤离时尚未释放。原始策略输出未保存，不能证明这是唯一成因。

新增固定Actor11的 original / feedback 两入口；相同edit0.01、RTC5和网络权重，无Learner、Replay更新或训练。原batch5入口保持。Actor11独立只读硬链接留存、189个文件SHA验证，避免源运行后续prune影响对照。

feedback仅当策略目标超过实际开度1mm、上一指令低于反馈1mm时重设张开参考。夹紧目标、六关节限速、4mm/100ms物理张开增量保留。每个未承诺逻辑步使用新鲜反馈重新规划夹爪；RTC前缀在提交前更新，提交后保持不变，最长5步内不重新锚定。插值也从修正后的开度起算；已提交前缀和实际发布端点在完整执行循环测试中一致。反馈超过250ms或非法值拒绝执行，HIL/暂停沿用原门控。

验证：208 passed / 4 skipped（CPU环境下GPU布局用例）；原执行、固定预览、snapshot、在线及传输回归覆盖，未新增GPU训练或机器人动作。未进行现场夹爪/插入验证，不承诺成功率提升。

实现：gripper_feedback.py、gripper_preview.py、ros_runtime.py三处默认兼容hook；脚本expo_gripper_preview_up.sh；模型登记expo-plug-actor11-gripper-{original,feedback}。现场源码交付与HTTP可用性以runtime/verification/gripper-feedback-20261015收据为准。
