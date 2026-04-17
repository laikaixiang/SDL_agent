# templates — 前端模板与静态资源

本目录包含 SDL_agent 前端的所有 HTML 模板和静态资源。

## 目录结构

```
templates/
├── index.html              # 主界面（HTML骨架，CSS/JS已解耦）
├── extraction_mode.html    # 文献提取模式页面
└── static/                 # CSS 和 JS 模块（Flask 映射到 /static/ URL）
    ├── css/
    │   └── main.css        # 全局样式（布局、面板、卡片、动画）
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

## JS 加载顺序

`index.html` 按以下顺序加载脚本，确保依赖关系正确：

1. `state.js` — 全局变量，所有模块依赖
2. `ui/panel.js` → `ui/menu.js` → `ui/input_state.js`
3. `notification.js`
4. `extraction/` → `hardware/` → `experiment/confirm` → `experiment/design`
5. `hardware/task_stream.js` → `hardware/step_panel.js`
6. `analysis/` → `experiment/chat`
7. `chat/chat.js` — 最后加载，依赖所有其他模块

## 注意事项

- Flask 通过 `static_folder='templates/static'` 将 `static/` 映射到 `/static/` URL
- 所有模块共享 `state.js` 中声明的全局变量，无需 import/export
- `collectToolParams(tool)` 在 `state.js` 中定义，供 `step_panel.js` 和 `experiment_design.js` 共用

---

# 如何添加右侧面板

## 面板系统架构

### 布局结构

系统使用 `app-wrapper` 作为主容器，所有面板都是其直接子元素：

```html
<div class="app-wrapper" id="app-wrapper">
    <!-- 左侧：主聊天界面 -->
    <div class="chat-container">...</div>

    <!-- 右侧：PDF面板 -->
    <div class="pdf-panel" id="pdf-panel">...</div>

    <!-- 右侧：单步控制面板 -->
    <div class="step-control-panel" id="step-control-panel">...</div>

    <!-- 在这里添加新的面板 -->
</div>
```

### 分屏模式

当任何右侧面板打开时，`app-wrapper` 会添加 `split-mode` 类：

```css
.app-wrapper.split-mode .chat-container {
    flex: 1;
    max-width: none;
}
```

## 添加新面板的步骤

### 步骤1: 添加HTML结构

在 `templates/index.html` 中，在 `app-wrapper` 内部添加新面板的HTML：

```html
<div class="data-analysis-panel" id="data-analysis-panel">
    <div class="panel-header">
        <span>数据分析</span>
        <button onclick="closeDataAnalysisPanel()"
                style="background:none;border:none;color:#fff;font-size:1.5rem;cursor:pointer;padding:0;line-height:1;">
            ×
        </button>
    </div>
    <div class="panel-body" id="data-analysis-body">
        <div style="text-align:center;color:#9ca3af;padding:40px 20px;">等待加载...</div>
    </div>
    <div class="panel-status-bar" id="data-analysis-status">就绪</div>
</div>
```

### 步骤2: 添加CSS样式

在 `static/css/main.css` 中添加面板样式：

```css
.data-analysis-panel {
    position: absolute;
    top: 0;
    right: 0;
    width: 50%;
    height: 100%;
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    display: flex;
    flex-direction: column;
    z-index: 50;
    transform: translateX(100%);
    opacity: 0;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

.data-analysis-panel.open {
    transform: translateX(0);
    opacity: 1;
}
```

通用面板子元素样式（已在 `main.css` 中定义，无需重复）：

```css
.panel-header { padding: 15px 20px; background: #1f2937; color: #fff; ... }
.panel-body    { flex: 1; overflow-y: auto; padding: 20px; background: #f9fafb; }
.panel-status-bar { padding: 12px 20px; background: #f3f4f6; border-top: 1px solid #e5e7eb; ... }
```

### 步骤3: 添加JavaScript控制函数

在对应的 JS 模块文件中添加打开和关闭函数：

```javascript
async function openDataAnalysisPanel() {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();

    const panel = document.getElementById('data-analysis-panel');
    panel.classList.add('open');
    document.getElementById('app-wrapper').classList.add('split-mode');
    bringPanelToFront('data-analysis-panel');

    await loadDataAnalysisContent();
}

function closeDataAnalysisPanel() {
    const panel = document.getElementById('data-analysis-panel');
    panel.classList.remove('open');
    removePanelFromTracking('data-analysis-panel');

    setTimeout(() => {
        if (activePanels.size === 0) {
            document.getElementById('app-wrapper').classList.remove('split-mode');
        }
    }, 400);
}
```

### 步骤4: 添加到菜单（可选）

```html
<div class="mode-menu" id="mode-menu">
    <!-- 现有菜单项... -->
    <div class="mode-item" onclick="openDataAnalysisPanel()">数据分析</div>
</div>
```

## 面板 z-index 管理系统

全局变量（定义在 `state.js`）：

```javascript
let panelZIndexCounter = 100;  // 从100开始递增
const activePanels = new Set(); // 跟踪当前打开的面板ID
```

管理函数（定义在 `ui/panel.js`）：

```javascript
function bringPanelToFront(panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        panelZIndexCounter++;
        panel.style.zIndex = panelZIndexCounter;
        activePanels.add(panelId);
    }
}

function removePanelFromTracking(panelId) {
    activePanels.delete(panelId);
}
```

## 注意事项

- 每个面板应独立，不应互相调用打开/关闭
- 始终使用 `bringPanelToFront()` 管理 z-index，不要在 CSS 中硬编码高值
- 关闭面板时检查 `activePanels.size === 0` 才退出 `split-mode`
- 使用 `setTimeout(fn, 400)` 等待动画完成后再检查分屏状态
- 面板宽度建议 50%，确保内容区域可滚动

## 调试技巧

```javascript
// 查看当前打开的面板
console.log('当前打开的面板:', Array.from(activePanels));
console.log('当前z-index计数器:', panelZIndexCounter);

// 检查单个面板状态
function debugPanel(panelId) {
    const panel = document.getElementById(panelId);
    console.log(`面板 ${panelId}:`, {
        isOpen: panel.classList.contains('open'),
        zIndex: panel.style.zIndex,
    });
}

// 重置面板系统
function resetPanelSystem() {
    activePanels.forEach(id => document.getElementById(id)?.classList.remove('open'));
    activePanels.clear();
    panelZIndexCounter = 100;
    document.getElementById('app-wrapper').classList.remove('split-mode');
}
```

## 后端API示例

如果面板需要从后端获取数据，在 `app.py` 中添加路由：

```python
@app.route('/api/experiment_history', methods=['GET'])
def get_experiment_history():
    try:
        history = load_experiment_history()
        return jsonify({'experiments': history, 'total': len(history)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```
