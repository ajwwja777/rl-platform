# 2026-09-26 面板图标按钮统一

源项目：Cobot `/media/agilex/Getea1/jiaan/projects/cobot-platform`。

8015 的设置、本机参数、相机/输出停靠面板及命令工具条共用 `panel-tool-button`：30×30 按钮，16×16 SVG，8px 圆角，相同边框、背景、线宽、悬停/按下/键盘焦点样式。设置页原字符 × 替换为同一个 SVG 关闭图标；旧输出浮窗也统一。悬停仅高亮，不位移；关闭、展开、复制、刷新行为保持原绑定。未改设备/采集/训练/部署逻辑。

静态修改：`app/backend/segmented_frontend/panel_controls.css`（新增）、`dock_ui.js`、`workspace_ui.js`、`index.html`。仅对同类图标工具应用共享样式，组合删除等具有独立语义的控件保留原行为。

验证：JS 语法；本地预览及线上 12 个同类按钮的尺寸、边框、背景、文字和 SVG 样式一致；真实 hover 不位移且高亮；关闭本机参数保留设置，关闭设置同时收起子浮层；1600/820/520px 无页面横向溢出，JS 异常 0。验证页面拦截所有非只读请求，无机器人动作。

19:43 发布 4 个静态文件，原 SHA256 校验及依赖优先原子替换；未重启服务。

备份：`runtime/backups/panel-buttons-20260926T194319/`。

证据：`runtime/diagnostics/panel-buttons-20260926/`（manifest、浏览器验收脚本、结果和截图）。回退恢复备份中的原 3 个文件，新增样式文件保留不影响旧 index。
