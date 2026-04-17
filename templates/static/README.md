# templates/static — 前端静态资源

本目录包含 AI Lab 智能中枢前端的所有 CSS 和 JavaScript 文件，
从原 `templates/index.html` 单文件解耦而来。

## 目录结构

```
static/
├── css/
│   └── main.css                  # 全局样式（布局、面板、卡片、动画）
└── js/
    ├── state.js                  # 全局变量、DOM 引用、共享工具函数
    ├── notification.js           # 右上角浮动通知 showNotification()
    ├── ui/
    │   ├── panel.js              # 面板 z-index 管理
    │   ├── menu.js               # 模式菜单 & 硬件子菜单
    │   └── input_state.js        # 输入区域 UI 状态 & 消息追加工具
    ├── chat/
    │   └── chat.js               # 消息发送、流式响应、响应分发
    ├── extraction/
    │   ├── extraction.js         # 文献提取字段确认逻辑
    │   └── file_upload.js        # PDF 拖拽/点击上传
    ├── hardware/
    │   ├── hardware.js           # 硬件操作确认卡片
    │   ├── task_stream.js        # SSE 任务流监听与中断
    │   └── step_panel.js         # 单步控制面板
    ├── analysis/
    │   ├── analysis.js           # 数据分析、文件选择器、算法执行
    │   └── algorithm_panel.js    # 左侧算法库面板
    └── experiment/
        ├── experiment_chat.js    # 实验设计 Agent 对话启动
        ├── experiment_confirm.js # 实验步骤确认/修改/跳过
        └── experiment_design.js  # 实验设计面板（画布、JSON、保存/执行）
```

## 加载顺序

`index.html` 按以下顺序加载脚本，确保依赖关系正确：

1. `state.js` — 全局变量，所有模块依赖
2. `ui/panel.js` → `ui/menu.js` → `ui/input_state.js`
3. `notification.js`
4. `extraction/` → `hardware/` → `experiment/confirm` → `experiment/design`
5. `hardware/task_stream.js` → `hardware/step_panel.js`
6. `analysis/` → `experiment/chat`
7. `chat/chat.js` — 最后加载，依赖所有其他模块

## 注意事项

- Flask 通过 `static_folder='templates/static'` 将此目录映射到 `/static/` URL
- 所有模块共享 `state.js` 中声明的全局变量，无需 import/export
- `collectToolParams(tool)` 在 `state.js` 中定义，供 `step_panel.js` 和 `experiment_design.js` 共用
