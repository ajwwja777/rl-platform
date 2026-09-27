# 2026-09-14 更新请求文件轮询竞态修复

17:46 session：episode24 的32步更新完成，验证误差/速度/加速度等明显恶化，候选被拒绝，正式actor仍5000。episode25终局时worker在 update_request.json 的 exists/read_text 两步之间遇到 FileNotFoundError，随后env等待更新时读取error状态退出，shutdown连接重置为后续错误。

根因边界：确认代码存在TOCTOU且未处理可选IPC文件短暂缺失。实际请求文件随后仍存在；路径挂载为fuseblk，但未证明文件短暂不可见的底层来源。没有发现代码显式删除该请求文件。
修复：online_cycle.read_optional_json直接读取，仅FileNotFoundError返回空消息，下轮继续。worker请求轮询和request_cycle状态轮询共用；坏JSON/权限错误继续报错；模型验证和更新步数不变。
验证：38项相关回归通过；Cobot部署环境5项新增竞态测试通过。Cobot隔离目录复制EP24/25完成两次32步更新，约1.29/1.30s，成功写出两笔事务。隔离测试只含两条新增回放，不等同于正式历史混合数据，候选accept结果不应外推。
正式replay完整episodes：13(12),14(13),16(27),23(32),24(22),25(16) transitions。processed到24，25待更新；原始数据未改，正式actor/checkpoint未改。
重启原online命令会恢复actor5000并可能在开始新rollout前补处理已持久化请求对应的episode25；无需重复采集24/25。
备份：Cobot runs/deploy-backup-request-poll-20260914；部署文件online_cycle.py、online_cycle_worker.py，manifest已更新。未提交push，未操作机器人。
