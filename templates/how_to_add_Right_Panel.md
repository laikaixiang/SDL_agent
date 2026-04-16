# 如何添加右侧面板

本文档说明如何在SDL_agent系统中添加新的右侧面板（如单步控制面板、PDF阅读面板等）。

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
<!-- 新面板示例：数据分析面板 -->
<div class="data-analysis-panel" id="data-analysis-panel">
    <!-- 面板头部 -->
    <div class="panel-header">
        <span>📊 数据分析</span>
        <button onclick="closeDataAnalysisPanel()" 
                style="background:none;border:none;color:#fff;font-size:1.5rem;cursor:pointer;padding:0;line-height:1;">
            ×
        </button>
    </div>
    
    <!-- 面板主体内容 -->
    <div class="panel-body" id="data-analysis-body">
        <div style="text-align:center;color:#9ca3af;padding:40px 20px;">
            等待加载...
        </div>
    </div>
    
    <!-- 面板底部状态栏（可选） -->
    <div class="panel-status-bar" id="data-analysis-status">
        就绪
    </div>
</div>
```

### 步骤2: 添加CSS样式

在 `<style>` 标签中添加面板样式：

```css
/* 数据分析面板样式 */
.data-analysis-panel {
    /* 定位：绝对定位在右侧 */
    position: absolute;
    top: 0;
    right: 0;
    
    /* 尺寸：占据右侧50%宽度 */
    width: 50%;
    height: 100%;
    
    /* 外观 */
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    
    /* 布局 */
    display: flex;
    flex-direction: column;
    
    /* 层级：基础z-index，会被JavaScript动态调整 */
    z-index: 50;
    
    /* 初始状态：隐藏在右侧外 */
    transform: translateX(100%);
    opacity: 0;
    
    /* 动画 */
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

/* 打开状态 */
.data-analysis-panel.open {
    transform: translateX(0);
    opacity: 1;
}

/* 面板头部 */
.panel-header {
    padding: 15px 20px;
    background: #1f2937;
    color: #fff;
    font-weight: bold;
    font-size: 0.95rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 2;
}

/* 面板主体 */
.panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: #f9fafb;
}

/* 面板状态栏 */
.panel-status-bar {
    padding: 12px 20px;
    background: #f3f4f6;
    border-top: 1px solid #e5e7eb;
    font-size: 0.85rem;
    color: #6b7280;
    text-align: center;
}
```

### 步骤3: 添加JavaScript控制函数

在 `<script>` 标签中添加打开和关闭函数：

```javascript
// =====================================
// 📊 数据分析面板控制
// =====================================

/**
 * 打开数据分析面板
 * 
 * 功能：
 * 1. 显示面板（添加 'open' 类）
 * 2. 启用分屏模式
 * 3. 将面板带到最前面（z-index管理）
 * 4. 加载面板内容
 */
async function openDataAnalysisPanel() {
    // 关闭菜单
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();

    // 获取面板元素
    const panel = document.getElementById('data-analysis-panel');
    
    // 显示面板
    panel.classList.add('open');
    
    // 启用分屏模式
    document.getElementById('app-wrapper').classList.add('split-mode');
    
    // 将此面板带到最前面（z-index管理）
    bringPanelToFront('data-analysis-panel');
    
    // 加载面板内容
    await loadDataAnalysisContent();
}

/**
 * 关闭数据分析面板
 * 
 * 功能：
 * 1. 隐藏面板（移除 'open' 类）
 * 2. 从面板跟踪中移除
 * 3. 如果没有其他面板打开，退出分屏模式
 */
function closeDataAnalysisPanel() {
    const panel = document.getElementById('data-analysis-panel');
    
    // 隐藏面板
    panel.classList.remove('open');
    
    // 从跟踪中移除
    removePanelFromTracking('data-analysis-panel');

    // 延迟检查是否需要退出分屏模式
    setTimeout(() => {
        const wrapper = document.getElementById('app-wrapper');
        
        // 检查是否还有其他面板打开
        const hasOpenPanels = activePanels.size > 0;
        
        // 如果没有打开的面板，退出分屏模式
        if (!hasOpenPanels) {
            wrapper.classList.remove('split-mode');
        }
    }, 400); // 等待动画完成
}

/**
 * 加载数据分析面板内容
 */
async function loadDataAnalysisContent() {
    const body = document.getElementById('data-analysis-body');
    body.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:40px 20px;">加载中...</div>';
    
    try {
        // 从后端获取数据
        const res = await fetch('/api/data_analysis_content');
        const data = await res.json();
        
        // 渲染内容
        renderDataAnalysisContent(data);
    } catch (e) {
        body.innerHTML = '<div style="text-align:center;color:#ef4444;padding:20px;">加载失败，请刷新重试</div>';
    }
}

/**
 * 渲染数据分析内容
 */
function renderDataAnalysisContent(data) {
    const body = document.getElementById('data-analysis-body');
    // 根据数据渲染内容
    body.innerHTML = `<div>数据内容: ${JSON.stringify(data)}</div>`;
}
```

### 步骤4: 添加到菜单（可选）

如果需要从菜单打开面板，在菜单中添加选项：

```html
<div class="mode-menu" id="mode-menu">
    <!-- 现有菜单项... -->
    
    <!-- 新增：数据分析面板 -->
    <div class="mode-item" onclick="openDataAnalysisPanel()">
        📊 数据分析
    </div>
</div>
```

## 面板z-index管理系统

为了确保后打开的面板显示在上面，系统使用了z-index管理机制：

### 全局变量

```javascript
// 在 <script> 标签开头添加
let panelZIndexCounter = 100; // 面板z-index计数器，从100开始
const activePanels = new Set(); // 跟踪当前打开的面板ID
```

### 管理函数

```javascript
/**
 * 将面板带到最前面
 * 
 * @param {string} panelId - 面板的DOM元素ID
 */
function bringPanelToFront(panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        // 递增计数器并设置z-index
        panelZIndexCounter++;
        panel.style.zIndex = panelZIndexCounter;
        
        // 添加到活跃面板集合
        activePanels.add(panelId);
        
        console.log(`[面板管理] ${panelId} 已带到前面，z-index: ${panelZIndexCounter}`);
    }
}

/**
 * 从跟踪中移除面板
 * 
 * @param {string} panelId - 面板的DOM元素ID
 */
function removePanelFromTracking(panelId) {
    activePanels.delete(panelId);
    console.log(`[面板管理] ${panelId} 已从跟踪中移除，剩余面板: ${activePanels.size}`);
}
```

### 使用示例

在打开面板时调用：

```javascript
async function openMyPanel() {
    const panel = document.getElementById('my-panel');
    panel.classList.add('open');
    
    // 重要：将面板带到最前面
    bringPanelToFront('my-panel');
    
    document.getElementById('app-wrapper').classList.add('split-mode');
}
```

在关闭面板时调用：

```javascript
function closeMyPanel() {
    const panel = document.getElementById('my-panel');
    panel.classList.remove('open');
    
    // 重要：从跟踪中移除
    removePanelFromTracking('my-panel');
    
    // 检查是否需要退出分屏模式
    setTimeout(() => {
        if (activePanels.size === 0) {
            document.getElementById('app-wrapper').classList.remove('split-mode');
        }
    }, 400);
}
```

## 完整示例：添加"实验历史"面板

### HTML

```html
<!-- 实验历史面板 -->
<div class="experiment-history-panel" id="experiment-history-panel">
    <div class="panel-header">
        <span>📜 实验历史</span>
        <button onclick="closeExperimentHistoryPanel()" 
                style="background:none;border:none;color:#fff;font-size:1.5rem;cursor:pointer;padding:0;line-height:1;">
            ×
        </button>
    </div>
    <div class="panel-body" id="experiment-history-body">
        <div style="text-align:center;color:#9ca3af;padding:40px 20px;">
            加载中...
        </div>
    </div>
    <div class="panel-status-bar" id="experiment-history-status">
        就绪
    </div>
</div>
```

### CSS

```css
/* 实验历史面板 */
.experiment-history-panel {
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

.experiment-history-panel.open {
    transform: translateX(0);
    opacity: 1;
}
```

### JavaScript

```javascript
// =====================================
// 📜 实验历史面板
// =====================================

async function openExperimentHistoryPanel() {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();

    const panel = document.getElementById('experiment-history-panel');
    panel.classList.add('open');
    
    document.getElementById('app-wrapper').classList.add('split-mode');
    bringPanelToFront('experiment-history-panel');
    
    await loadExperimentHistory();
}

function closeExperimentHistoryPanel() {
    const panel = document.getElementById('experiment-history-panel');
    panel.classList.remove('open');
    removePanelFromTracking('experiment-history-panel');

    setTimeout(() => {
        if (activePanels.size === 0) {
            document.getElementById('app-wrapper').classList.remove('split-mode');
        }
    }, 400);
}

async function loadExperimentHistory() {
    const body = document.getElementById('experiment-history-body');
    body.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:40px 20px;">加载中...</div>';
    
    try {
        const res = await fetch('/api/experiment_history');
        const data = await res.json();
        renderExperimentHistory(data);
    } catch (e) {
        body.innerHTML = '<div style="text-align:center;color:#ef4444;padding:20px;">加载失败</div>';
    }
}

function renderExperimentHistory(data) {
    const body = document.getElementById('experiment-history-body');
    let html = '<div style="padding:10px;">';
    
    if (data.experiments && data.experiments.length > 0) {
        data.experiments.forEach(exp => {
            html += `
                <div style="background:#fff;padding:15px;margin-bottom:10px;border-radius:8px;border:1px solid #e5e7eb;">
                    <div style="font-weight:bold;color:#1f2937;margin-bottom:5px;">
                        ${exp.name}
                    </div>
                    <div style="font-size:0.85rem;color:#6b7280;">
                        时间: ${exp.timestamp}
                    </div>
                    <div style="font-size:0.85rem;color:#6b7280;">
                        状态: ${exp.status}
                    </div>
                </div>
            `;
        });
    } else {
        html += '<div style="text-align:center;color:#9ca3af;">暂无实验历史</div>';
    }
    
    html += '</div>';
    body.innerHTML = html;
}
```

## 注意事项

### 1. 面板独立性

- 每个面板应该是独立的，不应该互相调用
- 打开一个面板不应该自动打开或关闭其他面板
- 使用 `activePanels` 集合来跟踪所有打开的面板

### 2. z-index管理

- 始终使用 `bringPanelToFront()` 来管理z-index
- 不要在CSS中硬编码高z-index值
- 基础z-index从50开始，动态z-index从100开始递增

### 3. 分屏模式

- 只有当至少有一个右侧面板打开时才启用 `split-mode`
- 关闭面板时检查 `activePanels.size`，如果为0则退出分屏模式
- 使用 `setTimeout` 等待动画完成后再检查

### 4. 动画和过渡

- 使用 `transform: translateX(100%)` 实现滑入效果
- 过渡时间建议为 0.35s
- 使用 `cubic-bezier(0.4, 0, 0.2, 1)` 缓动函数

### 5. 响应式设计

- 面板宽度建议为50%
- 在小屏幕上可以考虑使用100%宽度
- 确保面板内容可以滚动

## 调试技巧

### 查看当前打开的面板

```javascript
console.log('当前打开的面板:', Array.from(activePanels));
console.log('当前z-index计数器:', panelZIndexCounter);
```

### 检查面板状态

```javascript
function debugPanel(panelId) {
    const panel = document.getElementById(panelId);
    console.log(`面板 ${panelId}:`, {
        isOpen: panel.classList.contains('open'),
        zIndex: panel.style.zIndex,
        opacity: getComputedStyle(panel).opacity,
        transform: getComputedStyle(panel).transform
    });
}
```

### 重置面板系统

```javascript
function resetPanelSystem() {
    // 关闭所有面板
    activePanels.forEach(panelId => {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.remove('open');
        }
    });
    
    // 清空跟踪
    activePanels.clear();
    
    // 重置计数器
    panelZIndexCounter = 100;
    
    // 退出分屏模式
    document.getElementById('app-wrapper').classList.remove('split-mode');
    
    console.log('面板系统已重置');
}
```

## 后端API示例

如果面板需要从后端获取数据，添加相应的Flask路由：

```python
@app.route('/api/experiment_history', methods=['GET'])
def get_experiment_history():
    """
    获取实验历史记录
    """
    try:
        # 从数据库或文件读取历史记录
        history = load_experiment_history()
        
        return jsonify({
            'experiments': history,
            'total': len(history)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

## 总结

添加新的右侧面板需要：

1. **HTML结构** - 添加面板容器和内容
2. **CSS样式** - 定义面板外观和动画
3. **JavaScript控制** - 实现打开/关闭逻辑
4. **z-index管理** - 使用 `bringPanelToFront()` 和 `removePanelFromTracking()`
5. **分屏模式** - 正确管理 `split-mode` 类
6. **后端API** - 如果需要，添加数据接口

遵循这些步骤和最佳实践，可以确保新面板与现有系统无缝集成。
