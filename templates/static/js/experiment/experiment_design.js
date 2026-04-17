/**
 * experiment_design.js — 实验设计面板
 *
 * 管理实验设计面板的完整生命周期：打开/关闭、步骤增删改排序、
 * 拖拽重排、JSON 同步、保存/执行/导出实验方案。
 *
 * 依赖：state.js, ui/panel.js, ui/menu.js, ui/input_state.js,
 *       notification.js, hardware/step_panel.js（fetchStepTools, renderStepTools, setStepStatus）,
 *       hardware/task_stream.js（startTaskStream）
 */

/** 同时打开单步控制面板和实验设计画布面板，切换到实验设计模式，显示辅助函数栏。 */
function openExperimentDesignDialog() {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();
    setMode('hardware_design', '实验设计：', '🧪 实验设计对话');

    const wrapper = document.getElementById('app-wrapper');
    wrapper.classList.add('experiment-design-mode');

    // 打开单步控制面板（工具列表）
    const stepPanel = document.getElementById('step-control-panel');
    stepPanel.classList.add('open');
    bringPanelToFront('step-control-panel');
    document.getElementById('toggle-step-panel-btn').style.display = 'block';

    if (stepPanelTools.length === 0) fetchStepTools();
    else renderStepTools();

    // 打开实验设计画布面板
    const expPanel = document.getElementById('experiment-design-panel');
    expPanel.classList.add('open');
    bringPanelToFront('experiment-design-panel');

    document.getElementById('helper-functions').style.display = 'flex';
    setStepStatus('实验设计模式 — 双击工具添加到右侧画布', false);
}

/** 关闭实验设计面板和单步控制面板，隐藏辅助函数栏，恢复普通输入模式。 */
function closeExperimentDesignPanel() {
    document.getElementById('experiment-design-panel').classList.remove('open');
    document.getElementById('helper-functions').style.display = 'none';
    document.getElementById('toggle-step-panel-btn').style.display = 'none';
    removePanelFromTracking('experiment-design-panel');

    setStepStatus('就绪 — 单击 ▶ 展开工具，点击右侧 ▶ 执行', false);

    const wrapper = document.getElementById('app-wrapper');
    wrapper.classList.remove('experiment-design-mode');

    const stepPanel = document.getElementById('step-control-panel');
    stepPanel.classList.remove('open', 'collapsed');
    removePanelFromTracking('step-control-panel');

    setMode('normal', '', '');
}

/** 折叠/展开单步控制面板侧边栏，折叠时隐藏工具列表和状态栏，展开时恢复。 */
function toggleStepPanelCollapse() {
    const panel  = document.getElementById('step-control-panel');
    const btn    = document.getElementById('toggle-step-panel-btn');
    const isExpDesignMode = document.getElementById('app-wrapper').classList.contains('experiment-design-mode');

    stepPanelCollapsed = !stepPanelCollapsed;
    if (stepPanelCollapsed) {
        panel.classList.add('collapsed');
        btn.textContent = '▶';
        document.getElementById('step-panel-body').style.display = 'none';
        document.getElementById('step-status-bar').style.display = 'none';
        document.getElementById('helper-functions').style.display = 'none';
    } else {
        panel.classList.remove('collapsed');
        btn.textContent = '◀';
        document.getElementById('step-panel-body').style.display = 'block';
        document.getElementById('step-status-bar').style.display = 'block';
        if (isExpDesignMode) document.getElementById('helper-functions').style.display = 'flex';
    }
}

/** 从单步控制面板的参数表单中收集当前值，将工具步骤追加到 experimentSteps 并刷新画布和 JSON。 */
function addStepToExperiment(toolName) {
    const tool = stepPanelTools.find(t => t.name === toolName);
    if (!tool) return;

    const params = collectToolParams(tool);

    experimentSteps.push({ type: 'tool', name: toolName, description: tool.description, params });
    renderExperimentSteps();
    updateExperimentJSON();
    setStepStatus(`✅ 已添加步骤: ${toolName}`, false);
}

/** 根据 fnType 创建预设参数的辅助步骤（LOOP/GROUP/WAIT/CONDITION），追加到 experimentSteps 并刷新画布。 */
function addHelperFunction(fnType) {
    const templates = {
        LOOP:      { type: 'helper', name: 'LOOP',      description: '循环执行',  params: { iterations: 3, steps: [] } },
        GROUP:     { type: 'helper', name: 'GROUP',     description: '步骤组',    params: { name: '步骤组', steps: [] } },
        WAIT:      { type: 'helper', name: 'WAIT',      description: '等待',      params: { duration: 5000 } },
        CONDITION: { type: 'helper', name: 'CONDITION', description: '条件判断',  params: { condition: 'temperature > 100', then_steps: [], else_steps: [] } },
    };
    const step = templates[fnType];
    if (step) {
        experimentSteps.push(step);
        renderExperimentSteps();
        updateExperimentJSON();
    }
}

/** 清空画布后重新渲染 experimentSteps 中的所有步骤卡片，每张卡片绑定拖拽重排和上移/下移/编辑/删除按钮。 */
function renderExperimentSteps() {
    const canvas    = document.getElementById('exp-canvas-area');
    const emptyState = document.getElementById('exp-empty-state');

    // 清空现有步骤元素
    canvas.querySelectorAll('.exp-step-item').forEach(el => el.remove());

    if (experimentSteps.length === 0) {
        emptyState.style.display = 'block';
        return;
    }
    emptyState.style.display = 'none';

    experimentSteps.forEach((step, index) => {
        const stepEl = document.createElement('div');
        stepEl.className = 'exp-step-item';
        stepEl.draggable = true;
        stepEl.dataset.index = index;

        // 拖拽重排
        stepEl.addEventListener('dragstart', () => { draggedStepIndex = index; stepEl.classList.add('dragging'); });
        stepEl.addEventListener('dragend',   () => { stepEl.classList.remove('dragging'); draggedStepIndex = null; });
        stepEl.addEventListener('dragover',  (e) => e.preventDefault());
        stepEl.addEventListener('drop', (e) => {
            e.preventDefault();
            if (draggedStepIndex !== null && draggedStepIndex !== index) {
                const [moved] = experimentSteps.splice(draggedStepIndex, 1);
                experimentSteps.splice(index, 0, moved);
                renderExperimentSteps();
                updateExperimentJSON();
            }
        });

        // 步骤参数展示
        let paramsHtml = '';
        if (step.type === 'tool') {
            for (const [k, v] of Object.entries(step.params)) {
                paramsHtml += `<div><strong>${k}:</strong> ${v}</div>`;
            }
        } else {
            paramsHtml = `<div style="color:#7c3aed;font-style:italic;">${JSON.stringify(step.params)}</div>`;
        }

        stepEl.innerHTML = `
        <div class="exp-step-header">
            <div class="exp-step-type">${index + 1}. ${step.name}</div>
            <div class="exp-step-controls">
                <button class="exp-step-btn" onclick="moveStepUp(${index})" title="上移">▲</button>
                <button class="exp-step-btn" onclick="moveStepDown(${index})" title="下移">▼</button>
                <button class="exp-step-btn" onclick="editStep(${index})" title="编辑">✏️</button>
                <button class="exp-step-btn" onclick="deleteStep(${index})" title="删除">🗑️</button>
            </div>
        </div>
        <div style="font-size:0.8rem;color:#6b7280;margin-bottom:6px;">${step.description}</div>
        <div class="exp-step-params">${paramsHtml}</div>`;

        canvas.appendChild(stepEl);
    });
}

/** 将 experimentSteps 序列化为 JSON 并写入底部代码编辑器，保持画布与 JSON 同步。 */
function updateExperimentJSON() {
    document.getElementById('exp-code-content').value = JSON.stringify({
        experiment_name: experimentName,
        created_at: new Date().toISOString(),
        steps: experimentSteps
    }, null, 2);
}

/** 切换底部 JSON 代码区域的最小化/展开状态，同步更新折叠图标。 */
function toggleCodeAreaMinimize() {
    const codeArea = document.getElementById('exp-code-area');
    const icon     = document.getElementById('minimize-icon');
    const minimized = codeArea.classList.toggle('minimized');
    icon.textContent = minimized ? '▲' : '▼';
}

/** 将 index 位置的步骤与前一个步骤交换，刷新画布和 JSON。 */
function moveStepUp(index) {
    if (index > 0) {
        [experimentSteps[index], experimentSteps[index - 1]] = [experimentSteps[index - 1], experimentSteps[index]];
        renderExperimentSteps();
        updateExperimentJSON();
    }
}

/** 将 index 位置的步骤与后一个步骤交换，刷新画布和 JSON。 */
function moveStepDown(index) {
    if (index < experimentSteps.length - 1) {
        [experimentSteps[index], experimentSteps[index + 1]] = [experimentSteps[index + 1], experimentSteps[index]];
        renderExperimentSteps();
        updateExperimentJSON();
    }
}

/** 弹出 prompt 让用户以 JSON 格式编辑指定步骤的参数，解析成功后刷新画布和 JSON。 */
function editStep(index) {
    const step = experimentSteps[index];
    const newParams = prompt(`编辑步骤参数 (JSON格式):\n当前: ${JSON.stringify(step.params)}`, JSON.stringify(step.params));
    if (newParams) {
        try {
            step.params = JSON.parse(newParams);
            renderExperimentSteps();
            updateExperimentJSON();
        } catch (e) {
            alert('JSON格式错误: ' + e.message);
        }
    }
}

/** 弹出确认框，用户确认后从 experimentSteps 中删除指定索引的步骤并刷新画布。 */
function deleteStep(index) {
    if (confirm('确定删除此步骤？')) {
        experimentSteps.splice(index, 1);
        renderExperimentSteps();
        updateExperimentJSON();
    }
}

/** 弹出确认框，用户确认后清空 experimentSteps 和实验名称，重置画布和 JSON 编辑器。 */
function clearExperimentDesign() {
    if (confirm('确定清空所有实验步骤？')) {
        experimentSteps = [];
        experimentName  = '未命名实验';
        renderExperimentSteps();
        updateExperimentJSON();
    }
}

/**
 * 解析标准实验 JSON，将步骤写入 experimentSteps，刷新画布和 JSON 编辑器。
 * 兼容旧格式（action 字段）和新格式（name 字段）。
 * @param {Object} json - 标准实验 JSON 格式 { experiment_name, steps }
 */
function loadExperimentFromJSON(json) {
    try {
        experimentName  = json.experiment_name || '未命名实验';
        experimentSteps = json.steps.map(step => ({
            type:        step.type || 'tool',
            name:        step.name || step.action || '',
            params:      step.params || {},
            description: step.description || ''
        }));
        renderExperimentSteps();
        updateExperimentJSON();
    } catch (e) {
        alert('加载实验设计失败: ' + e.message);
    }
}

/** 弹出 prompt 让用户输入实验名称，然后将当前实验方案 POST 到 /api/save_experiment_design 保存到服务器。 */
async function saveExperimentDesign() {
    const name = prompt('请输入实验名称:', experimentName);
    if (!name) return;
    experimentName = name;
    updateExperimentJSON();

    try {
        const res = await fetch('/api/save_experiment_design', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experiment_name: experimentName, created_at: new Date().toISOString(), steps: experimentSteps })
        });
        const data = await res.json();
        if (data.success) appendMessage(`✅ 实验设计已保存: ${data.filepath}`, 'ai');
        else appendMessage(`❌ 保存失败: ${data.message}`, 'ai');
    } catch (e) {
        appendMessage(`❌ 保存异常: ${e.message}`, 'ai');
    }
}

/** 弹出确认框后将当前实验方案 POST 到 /api/execute_experiment_design，成功则启动 SSE 监听执行进度。 */
async function executeExperimentDesign() {
    if (experimentSteps.length === 0) { alert('请先添加实验步骤'); return; }
    if (!confirm(`确定执行实验 "${experimentName}"？\n共 ${experimentSteps.length} 个步骤`)) return;

    appendMessage(`🚀 开始执行实验: ${experimentName}`, 'user');
    try {
        const res = await fetch('/api/execute_experiment_design', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experiment_name: experimentName, created_at: new Date().toISOString(), steps: experimentSteps })
        });
        const data = await res.json();
        if (data.type === 'task_trigger') {
            appendMessage(data.reply, 'ai');
            startTaskStream();
        } else {
            appendMessage(data.reply || '执行失败', 'ai');
        }
    } catch (e) {
        appendMessage(`❌ 执行异常: ${e.message}`, 'ai');
    }
}

/** 弹出 prompt 让用户输入保存路径（默认从服务器获取会话路径），然后将实验 JSON POST 到 /api/export_experiment_json。 */
async function exportExperimentJSON() {
    const filename = `${experimentName}_${Date.now()}.json`;
    let defaultPath = `experiment_designs/${filename}`;

    try {
        const res = await fetch('/api/get_session_path?subdir=experiment_designs');
        const data = await res.json();
        if (data.success) defaultPath = `${data.path}/${filename}`;
    } catch (e) {
        // 无法获取会话路径时使用默认路径
    }

    const savePath = prompt('请输入保存路径:', defaultPath);
    if (!savePath) return;

    try {
        const res = await fetch('/api/export_experiment_json', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                json_data: { experiment_name: experimentName, created_at: new Date().toISOString(), steps: experimentSteps },
                filepath: savePath
            })
        });
        const data = await res.json();
        if (data.success) appendMessage(`📤 实验设计已导出: ${data.filepath}`, 'ai');
        else appendMessage(`❌ 导出失败: ${data.message}`, 'ai');
    } catch (e) {
        appendMessage(`❌ 导出异常: ${e.message}`, 'ai');
    }
}
