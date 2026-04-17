# 实现计划：单步控制和实验设计模式的UI改进

## 背景

本计划解决SDL_agent硬件控制系统中的两个问题：

1. **单步控制和PDF界面的独立性**: 单步控制面板和PDF阅读界面应该是完全独立的、平级的面板，不应该互相调用。如果它们重叠，后打开的应该显示在上面。同时需要提供文档说明如何添加新的右侧面板。

2. **实验设计模式卡住**: 实验设计模式会卡在"准备连接解析引擎..."然后没有反应，即使API已确认正常工作。

## 问题分析

### 问题1: 单步控制 + PDF界面的独立性

**当前行为:**
- `openStepPanel()` (index.html第1452行) 打开单步控制面板
- 它添加 `split-mode` 类来启用分屏视图
- PDF面板是独立控制的，不会自动打开

**需要改进的地方:**
- 确保两个面板完全独立，互不调用
- 需要正确的z-index管理，确保后打开的面板在上面
- 需要提供清晰的文档说明如何添加新的右侧面板

### 问题2: 实验设计模式卡住

**当前流程:**
1. 用户选择实验设计模式
2. 前端调用 `startExperimentChat()` (第1278行)
3. POST到 `/api/experiment_chat` (app.py第762行)
4. 后端在后台线程启动agent (第786-812行)
5. 返回 `task_trigger` 响应
6. 前端调用 `startTaskStream()` (第738行)
7. 打开SSE连接到 `/api/task_stream` (app.py第102行)
8. 显示 "准备连接解析引擎..." (第746行)
9. **卡在这里** - 没有从SSE接收到消息

**可能的根本原因:**
- Agent线程静默失败，没有推送错误消息
- Agent初始化时的异常没有被正确捕获
- Agent没有向task_manager推送初始状态消息
- SSE连接时序问题（前端在后端准备好之前就连接了）

## 关键文件

- `templates/index.html` - 前端UI和JavaScript逻辑
- `app.py` - 后端Flask路由和线程管理
- `core/experiment_agent.py` - 实验设计agent实现
- `core/config.py` - 配置文件（包括系统提示词）

## 实现计划

### 阶段1: 确保单步控制和PDF界面的独立性

**步骤 1.1: 确保面板独立性**
- 文件: `templates/index.html` 第1452行附近
- 确认 `openStepPanel()` 不会自动打开PDF面板
- 确认两个面板可以独立打开和关闭

**步骤 1.2: 实现正确的z-index分层**
- 文件: `templates/index.html` CSS部分
- 为 `.step-control-panel` 和 `.pdf-panel` 设置合适的基础z-index
- 添加动态z-index管理，确保后打开的面板在上面
- 使用JavaScript跟踪面板打开顺序

**步骤 1.3: 创建面板管理文档**
- 创建新文件: `docs/如何添加右侧面板.md`
- 说明面板的HTML结构
- 说明CSS样式要求
- 说明JavaScript控制逻辑
- 提供完整的代码示例

### 阶段2: 修复实验设计模式卡住问题

**步骤 2.1: 在agent线程中添加错误处理和日志**
- 文件: `app.py` 第786-812行
- 在agent初始化周围添加try-catch
- 线程启动后立即推送初始"连接中"消息
- 添加详细的错误日志到控制台

**步骤 2.2: 添加超时和心跳机制**
- 文件: `app.py` 中的 `run_agent_thread()`
- 在运行agent之前推送初始状态消息
- 添加agent启动的超时检测
- 确保异常总是被捕获并报告

**步骤 2.3: 改进前端错误处理**
- 文件: `templates/index.html` 中的 `startTaskStream()`
- 添加SSE连接的超时检测
- 如果在合理时间内没有响应，显示错误消息
- 添加重试机制或清晰的错误报告

**步骤 2.4: 添加调试输出**
- 文件: `core/experiment_agent.py`
- 在agent执行的关键点添加日志
- 验证agent初始化成功
- 记录agent开始处理用户消息的时间

### 阶段3: 测试和验证

**测试用例1: 单步控制 + PDF独立性**
1. 打开单步控制面板
2. 验证PDF面板不会自动打开
3. 手动打开PDF面板
4. 验证两个面板可以同时显示
5. 关闭单步控制面板，验证PDF面板保持打开
6. 再次打开单步控制面板，验证它显示在PDF面板上面（如果重叠）
7. 关闭PDF面板，再打开，验证它显示在单步控制面板上面

**测试用例2: 实验设计模式**
1. 选择实验设计模式
2. 输入测试命令
3. 验证连接消息快速出现（2秒内）
4. 验证agent开始处理（不卡住）
5. 如果发生错误，验证显示清晰的错误消息
6. 使用各种命令测试以确保健壮性

## 实现细节

### 问题1: 单步控制和PDF界面的独立性

**1.1 添加面板z-index管理系统**

```javascript
// 在 index.html 的 <script> 部分添加全局变量
let panelZIndexCounter = 100; // 面板z-index计数器，从100开始
const activePanels = new Set(); // 跟踪当前打开的面板

// 面板打开时调用此函数来更新z-index
function bringPanelToFront(panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        panelZIndexCounter++;
        panel.style.zIndex = panelZIndexCounter;
        activePanels.add(panelId);
    }
}

// 面板关闭时调用此函数
function removePanelFromTracking(panelId) {
    activePanels.delete(panelId);
}
```

**1.2 修改 openStepPanel() 函数**

```javascript
// 修改现有的 openStepPanel() 函数
async function openStepPanel() {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();

    const panel = document.getElementById('step-control-panel');
    panel.classList.add('open');
    
    // 将此面板带到最前面
    bringPanelToFront('step-control-panel');
    
    // 不自动打开PDF面板 - 保持独立性
    document.getElementById('app-wrapper').classList.add('split-mode');
    
    if (stepPanelTools.length === 0) {
        await fetchStepTools();
    } else {
        renderStepTools();
    }
}
```

**1.3 修改 closeStepPanel() 函数**

```javascript
function closeStepPanel() {
    const panel = document.getElementById('step-control-panel');
    panel.classList.remove('open');
    
    // 从跟踪中移除
    removePanelFromTracking('step-control-panel');

    setTimeout(() => {
        const wrapper = document.getElementById('app-wrapper');
        const pdfPanel = document.getElementById('pdf-panel');
        const pdfVisible = pdfPanel && parseFloat(getComputedStyle(pdfPanel).opacity || '0') > 0;
        
        // 只有当PDF面板也关闭时才退出split-mode
        if (!pdfVisible) {
            wrapper.classList.remove('split-mode');
        }
    }, 400);
}
```

### 问题2: 实验设计模式卡住

**2.1 修改 app.py 中的 run_agent_thread() 函数**

```python
# 在 app.py 第786行附近，修改 run_agent_thread() 函数
def run_agent_thread():
    """后台线程：运行实验设计 Agent"""
    try:
        # 立即推送初始状态消息
        print("[实验设计] 线程已启动，开始初始化...")
        task_manager.put_task_message({
            "type": "info",
            "data": "正在初始化实验设计引擎..."
        })
        
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print("[实验设计] 事件循环已创建")
        
        # 定义异步回调函数
        async def send_event_async(event):
            task_manager.put_task_message(event)
        
        # 推送就绪消息
        task_manager.put_task_message({
            "type": "info",
            "data": "引擎已就绪，开始分析需求..."
        })
        print(f"[实验设计] 开始运行agent，用户消息: {user_message[:50]}...")
        
        # 运行agent
        result = loop.run_until_complete(
            experiment_agent.run(session_id, user_message, send_event_async)
        )
        
        print(f"[实验设计] Agent执行成功，结果: {result[:100]}...")
        
        # 推送完成消息
        task_manager.put_task_message({
            "type": "complete",
            "data": {"agent_reply": result}
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[错误] 实验设计Agent线程失败:")
        print(error_detail)
        
        # 推送错误消息
        task_manager.put_task_message({
            "type": "complete",
            "data": {"error": f"实验设计Agent错误: {str(e)}"}
        })
    finally:
        loop.close()
        task_manager.task_running = False
        print("[实验设计] 线程已结束")
```

**2.2 在 core/experiment_agent.py 中添加日志**

```python
# 在 run() 方法中添加日志（第71行附近）
async def run(self, session_id: str, user_message: str, send_event) -> str:
    """运行实验设计智能体"""
    print(f"[ExperimentAgent] 开始处理会话 {session_id}")
    print(f"[ExperimentAgent] 用户消息: {user_message}")
    
    session = self._get_or_create_session(session_id)
    
    # 如果有关联的 PDF，在消息前附加路径提示
    if session["pdf_path"]:
        full_input = f"[Current PDF is at: {session['pdf_path']}]\n\n{user_message}"
        print(f"[ExperimentAgent] PDF路径: {session['pdf_path']}")
    else:
        full_input = user_message
        print(f"[ExperimentAgent] 无关联PDF")
    
    # 创建依赖容器
    deps = Deps(send_event=send_event, agent=self, session_id=session_id)
    print(f"[ExperimentAgent] 依赖容器已创建")
    
    # 调用 PydanticAI Agent
    print(f"[ExperimentAgent] 开始调用PydanticAI Agent...")
    result = await self._agent.run(
        full_input, deps=deps, message_history=session["history"],
    )
    print(f"[ExperimentAgent] Agent调用完成")
    
    # 更新会话历史
    session["history"] = list(result.all_messages())
    print(f"[ExperimentAgent] 会话历史已更新，共 {len(session['history'])} 条消息")
    
    return result.output
```

## 风险缓解

- **向后兼容性**: 更改保持其他模式的现有功能
- **错误处理**: 全面的try-catch块防止静默失败
- **用户体验**: 清晰的错误消息在问题发生时引导用户
- **测试**: 每次更改后的增量测试确保稳定性

## 成功标准（7条TODO）

### TODO 1: 打开单步控制面板时不会自动打开PDF面板 ✅
**状态**: 需要验证  
**文件**: `templates/index.html` 第1452行  
**任务**: 
- 检查 `openStepPanel()` 函数，确认没有调用PDF面板打开逻辑
- 确认只添加 `split-mode` 类，不操作PDF面板

### TODO 2: 两个面板可以独立控制 ✅
**状态**: 需要验证和改进  
**文件**: `templates/index.html` 第1468行  
**任务**:
- 检查 `closeStepPanel()` 函数，确认关闭单步控制时不会关闭PDF
- 检查 `closePdfPanel()` 函数（如果存在），确认关闭PDF时不会关闭单步控制
- 改进 `closeStepPanel()` 逻辑，只在所有面板都关闭时才退出 `split-mode`

### TODO 3: 面板重叠时有正确的z-index分层（后打开的在上面） 🔧
**状态**: 需要实现  
**文件**: `templates/index.html` JavaScript部分  
**任务**:
1. 添加全局变量：
   - `let panelZIndexCounter = 100;`
   - `const activePanels = new Set();`
2. 实现 `bringPanelToFront(panelId)` 函数
3. 实现 `removePanelFromTracking(panelId)` 函数
4. 在 `openStepPanel()` 中调用 `bringPanelToFront('step-control-panel')`
5. 在 `closeStepPanel()` 中调用 `removePanelFromTracking('step-control-panel')`
6. 在PDF面板的打开/关闭函数中也添加相应调用

### TODO 4: 实验设计模式立即显示进度消息 🔧
**状态**: 需要实现  
**文件**: `app.py` 第786-812行  
**任务**:
1. 在 `run_agent_thread()` 函数开始处立即推送消息：
   ```python
   task_manager.put_task_message({
       "type": "info",
       "data": "正在初始化实验设计引擎..."
   })
   ```
2. 在创建事件循环后推送消息：
   ```python
   task_manager.put_task_message({
       "type": "info",
       "data": "引擎已就绪，开始分析需求..."
   })
   ```
3. 添加 `print()` 日志输出到控制台

### TODO 5: 如果agent初始化失败，显示清晰的错误消息 🔧
**状态**: 需要实现  
**文件**: `app.py` 第786-812行 和 `core/experiment_agent.py` 第71-104行  
**任务**:
1. 在 `app.py` 的 `run_agent_thread()` 中：
   - 添加完整的 try-except 块
   - 捕获所有异常并打印详细堆栈
   - 推送错误消息到前端
2. 在 `core/experiment_agent.py` 的 `run()` 方法中：
   - 添加日志输出到关键步骤
   - 记录会话ID、用户消息、PDF路径等信息
   - 记录agent调用的开始和结束

### TODO 6: 没有卡住或无限等待的状态 🔧
**状态**: 通过TODO 4和5解决  
**依赖**: TODO 4, TODO 5  
**验证方法**:
- 启动实验设计模式后，2秒内应该看到进度消息
- 如果发生错误，应该立即显示错误消息
- 不应该出现无响应的情况

### TODO 7: 提供清晰的文档说明如何添加新的右侧面板 ✅
**状态**: 已完成，需要移动文件  
**文件**: `temporal/如何添加右侧面板.md`  
**任务**:
- 将文件从 `temporal/` 移动到 `templates/` 文件夹
- 或者根据用户指示放置到正确位置

## 执行顺序

### 第一批：面板独立性和z-index管理（TODO 1, 2, 3）
1. 验证TODO 1和2的现有实现
2. 实现TODO 3的z-index管理系统
3. 测试面板的独立性和层级关系

### 第二批：实验设计模式修复（TODO 4, 5, 6）
1. 实现TODO 4的立即消息推送
2. 实现TODO 5的错误处理和日志
3. 测试TODO 6，确认没有卡住

### 第三批：文档整理（TODO 7）
1. 移动文档文件到正确位置
