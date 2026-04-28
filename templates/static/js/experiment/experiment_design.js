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

expCodeJSON = document.getElementById('exp-code-content').value;
let editingStepIndex = -1;  // 当前正在编辑的步骤索引，-1 表示无

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

/** 根据 fnType 创建预设参数的辅助步骤（LOOP/GROUP/WAIT/CONDITION/END/USER_INPUT），追加到 experimentSteps 并刷新画布。 */
function addHelperFunction(fnType) {
    const templates = {
        LOOP:       { type: 'helper', name: 'LOOP',       description: '循环执行',  params: { iterations: 3, steps: [] } },
        GROUP:      { type: 'helper', name: 'GROUP',      description: '步骤组',    params: { name: '步骤组', steps: [] } },
        WAIT:       { type: 'helper', name: 'WAIT',       description: '等待',      params: { duration: 5000 } },
        CONDITION:  { type: 'helper', name: 'CONDITION',  description: '条件判断',  params: { condition: 'temperature > 100', then_steps: [], else_steps: [] } },
        END:        { type: 'helper', name: 'END',        description: '结束点',    params: {} },
        USER_INPUT: { type: 'helper', name: 'USER_INPUT', description: '用户输入',  params: { prompt: '请输入参数', variable_name: 'user_value' } },
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

    canvas.querySelectorAll('.exp-step-item').forEach(el => el.remove());

    if (experimentSteps.length === 0) {
        emptyState.style.display = 'block';
        editingStepIndex = -1;
        return;
    }
    emptyState.style.display = 'none';

    const wasEditing = editingStepIndex;

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

        const isEditing = (index === wasEditing);
        stepEl.innerHTML = `
        <div class="exp-step-header">
            <div class="exp-step-type">
                <span>${index + 1}. ${step.description || step.name}</span>
                <span class="exp-step-type-en">${step.name}</span>
            </div>
            <div class="exp-step-controls">
                <button class="exp-step-btn" onclick="moveStepUp(${index})" title="上移">▲</button>
                <button class="exp-step-btn" onclick="moveStepDown(${index})" title="下移">▼</button>
                <button class="exp-step-btn" id="edit-btn-${index}" onclick="toggleStepEdit(${index})" title="编辑">${isEditing ? '✖' : '✏️'}</button>
                <button class="exp-step-btn" onclick="deleteStep(${index})" title="删除">🗑️</button>
            </div>
        </div>
        <div class="exp-step-edit-area" id="edit-area-${index}" style="${isEditing ? '' : 'display:none;'}"></div>`;

        canvas.appendChild(stepEl);

        // 如果之前在编辑此步骤，重新填充编辑表单
        if (isEditing) {
            _fillEditArea(index);
        }
    });
}

/** 将 experimentSteps 序列化为 JSON 并写入底部代码编辑器，保持画布与 JSON 同步。仅在 JSON 视图模式下更新。 */
function updateExperimentJSON() {
    const steps = experimentSteps.map(step => {
        const s = { type: step.type, name: step.name, params: step.params, description: step.description };
        if (step.type === 'software') {
            if (step.input_file)  s.input_file  = step.input_file;
            if (step.output_file) s.output_file = step.output_file;
            if (step.user_params) s.user_params = step.user_params;
        }
        return s;
    });
    const jsonStr = JSON.stringify({
        experiment_name: experimentName,
        created_at: new Date().toISOString(),
        steps
    }, null, 2);
    expCodeJSON = jsonStr;
    if (expCodeViewMode === 'json') {
        document.getElementById('exp-code-content').value = jsonStr;
    }
}

/** 在 JSON 和 Python 代码视图之间切换。 */
async function switchCodeView(mode) {
    expCodeViewMode = mode;
    document.getElementById('code-view-json').classList.toggle('active', mode === 'json');
    document.getElementById('code-view-python').classList.toggle('active', mode === 'python');

    if (mode === 'json') {
        document.getElementById('exp-code-content').value = expCodeJSON;
    } else {
        await _loadPythonCodeView();
    }
}

/** 编译画布内容并切换到 Python 代码视图。 */
async function compileToPythonView() {
    if (experimentSteps.length === 0) {
        showNotification('请先添加实验步骤', 'warning');
        return;
    }

    // 先切换到 Python 视图（会触发编译加载）
    expCodeViewMode = 'python';
    document.getElementById('code-view-json').classList.remove('active');
    document.getElementById('code-view-python').classList.add('active');
    await _loadPythonCodeView();
}

/** 调用编译 API 获取 Python 代码并显示在编辑器中。 */
async function _loadPythonCodeView() {
    if (experimentSteps.length === 0) {
        document.getElementById('exp-code-content').value = '# 暂无实验步骤，请先添加步骤';
        return;
    }

    document.getElementById('exp-code-content').value = '# 正在编译...';

    try {
        const res = await fetch('/api/compile_experiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment_json: {
                    experiment_name: experimentName,
                    created_at: new Date().toISOString(),
                    steps: experimentSteps
                }
            })
        });
        const data = await res.json();
        if (data.success) {
            expPythonCode = data.code;
            document.getElementById('exp-code-content').value = data.code;
        } else {
            document.getElementById('exp-code-content').value = `# 编译失败: ${data.message}`;
        }
    } catch (e) {
        document.getElementById('exp-code-content').value = `# 编译异常: ${e.message}`;
    }
}

/** 切换底部 JSON 代码区域的最小化/展开状态，同步更新折叠图标。 */
function toggleCodeAreaMinimize() {
    const codeArea = document.getElementById('exp-code-area');
    const icon     = document.getElementById('minimize-icon');

    // 如果在全屏状态，先退出全屏
    if (codeArea.classList.contains('fullscreen')) {
        toggleCodeAreaFullscreen();
        return;
    }

    const minimized = codeArea.classList.toggle('minimized');
    icon.textContent = minimized ? '+' : '−';
}

/** 切换代码区域全屏显示，占满整个实验流程面板。 */
function toggleCodeAreaFullscreen() {
    const codeArea = document.getElementById('exp-code-area');
    const icon     = document.getElementById('fullscreen-icon');
    const isFullscreen = codeArea.classList.toggle('fullscreen');
    icon.textContent = isFullscreen ? '⛶' : '⛶';

    if (isFullscreen) {
        // 全屏时自动展开
        codeArea.classList.remove('minimized');
        document.getElementById('minimize-icon').textContent = '−';
    }
}

/** 从下方 JSON 代码重新解析并同步到上方画布，以代码为准。 */
function syncCodeToCanvas() {
    try {
        const codeContent = document.getElementById('exp-code-content').value;
        const expData = JSON.parse(codeContent);

        if (!expData.steps || !Array.isArray(expData.steps)) {
            alert('JSON格式错误：缺少 steps 数组');
            return;
        }

        // 更新全局状态
        experimentName = expData.experiment_name || '未命名实验';
        experimentSteps = expData.steps;

        // 重新渲染画布
        renderExperimentSteps();

        showNotification('✅ 已从代码同步到画布', 'success');
    } catch (e) {
        alert('JSON解析失败: ' + e.message);
    }
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

/** 展开/收起指定步骤的内联编辑面板。每次只有一个步骤处于编辑状态。 */
function toggleStepEdit(index) {
    if (editingStepIndex === index) {
        // 关闭当前编辑
        editingStepIndex = -1;
        const area = document.getElementById('edit-area-' + index);
        if (area) area.style.display = 'none';
        const btn = document.getElementById('edit-btn-' + index);
        if (btn) btn.textContent = '✏️';
        return;
    }

    // 关闭之前的编辑
    if (editingStepIndex >= 0) {
        const prevArea = document.getElementById('edit-area-' + editingStepIndex);
        if (prevArea) prevArea.style.display = 'none';
        const prevBtn = document.getElementById('edit-btn-' + editingStepIndex);
        if (prevBtn) prevBtn.textContent = '✏️';
    }

    editingStepIndex = index;
    const area = document.getElementById('edit-area-' + index);
    if (area) area.style.display = '';
    const btn = document.getElementById('edit-btn-' + index);
    if (btn) btn.textContent = '✖';
    _fillEditArea(index);
}

/** 填充指定步骤的内联编辑区域表单。 */
function _fillEditArea(index) {
    const step = experimentSteps[index];
    const area = document.getElementById('edit-area-' + index);
    if (!area) return;

    let formHtml = '';
    formHtml += `<div class="edit-field"><label>中文描述:</label><input type="text" id="edit-desc-${index}" value="${step.description || ''}" /></div>`;

    // 根据步骤类型生成参数表单
    if (step.type === 'tool') {
        const tool = stepPanelTools.find(t => t.name === step.name);
        if (tool && tool.parameters) {
            for (const param of tool.parameters) {
                const currentValue = step.params[param.name] !== undefined ? step.params[param.name] : (param.default || '');
                formHtml += `<div class="edit-field">
                    <label>${param.name}:</label>
                    <input type="text" data-param="${param.name}" value="${currentValue}" placeholder="${param.description || ''}" />
                </div>`;
            }
        } else {
            for (const [k, v] of Object.entries(step.params)) {
                formHtml += `<div class="edit-field">
                    <label>${k}:</label>
                    <input type="text" data-param="${k}" value="${v}" />
                </div>`;
            }
        }
    } else if (step.type === 'software') {
        const inputPath = step.input_file || '';
        const outputPath = step.output_file || '';
        formHtml += `<div class="edit-field">
            <label>输入文件:</label>
            <div class="edit-picker-row">
                <span class="edit-picker-value" id="edit-input-display-${index}">${inputPath || '未选择'}</span>
                <button type="button" class="algo-picker-btn" onclick="pickInputFile(${index})">选择</button>
                <input type="hidden" id="edit-input-file-${index}" value="${inputPath}" />
            </div>
        </div>`;
        formHtml += `<div class="edit-field">
            <label>输出目录:</label>
            <div class="edit-picker-row">
                <span class="edit-picker-value" id="edit-output-display-${index}">${outputPath || '未选择'}</span>
                <button type="button" class="algo-picker-btn" onclick="pickOutputDir(${index})">选择</button>
                <input type="hidden" id="edit-output-file-${index}" value="${outputPath}" />
            </div>
        </div>`;
        for (const [k, v] of Object.entries(step.params || {})) {
            formHtml += `<div class="edit-field">
                <label>${k}:</label>
                <input type="text" data-param="${k}" value="${v}" />
            </div>`;
        }
    } else {
        // helper 类型
        for (const [k, v] of Object.entries(step.params)) {
            const valueStr = typeof v === 'object' ? JSON.stringify(v) : v;
            formHtml += `<div class="edit-field">
                <label>${k}:</label>
                <input type="text" data-param="${k}" value="${valueStr}" />
            </div>`;
        }
    }

    formHtml += `<div class="edit-actions">
        <button onclick="saveInlineStepEdit(${index})" class="btn-yes">保存</button>
    </div>`;

    area.innerHTML = formHtml;
}

/** 从内联编辑表单收集数据，更新步骤并刷新。 */
function saveInlineStepEdit(index) {
    const step = experimentSteps[index];
    const area = document.getElementById('edit-area-' + index);
    if (!area) return;

    // 更新描述
    const descInput = document.getElementById('edit-desc-' + index);
    if (descInput) step.description = descInput.value;

    // 更新参数
    const paramInputs = area.querySelectorAll('input[data-param]');
    paramInputs.forEach(input => {
        const paramName = input.dataset.param;
        let value = input.value;
        try {
            const parsed = JSON.parse(value);
            value = parsed;
        } catch (e) {}
        step.params[paramName] = value;
    });

    // software 类型特殊字段
    if (step.type === 'software') {
        const inputFile = document.getElementById('edit-input-file-' + index);
        const outputFile = document.getElementById('edit-output-file-' + index);
        if (inputFile) step.input_file = inputFile.value;
        if (outputFile) step.output_file = outputFile.value;
    }

    editingStepIndex = -1;
    renderExperimentSteps();
    updateExperimentJSON();
    setStepStatus(`✅ 已更新步骤: ${step.name}`, false);
}

/** 为 software 步骤选择输入文件。 */
function pickInputFile(index) {
    showFileSelector((path, name) => {
        const display = document.getElementById('edit-input-display-' + index);
        const hidden  = document.getElementById('edit-input-file-' + index);
        if (display) display.textContent = path;
        if (hidden)  hidden.value = path;
    });
}

/** 为 software 步骤选择输出目录。 */
function pickOutputDir(index) {
    showOutputDirSelector((dirPath, label) => {
        const display = document.getElementById('edit-output-display-' + index);
        const hidden  = document.getElementById('edit-output-file-' + index);
        if (display) display.textContent = dirPath;
        if (hidden)  hidden.value = dirPath;
    });
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
        experimentSteps = json.steps.map(step => {
            const s = {
                type:        step.type || 'tool',
                name:        step.name || step.action || '',
                params:      step.params || {},
                description: step.description || ''
            };
            if (step.type === 'software') {
                if (step.input_file)  s.input_file  = step.input_file;
                if (step.output_file) s.output_file = step.output_file;
                if (step.user_params) s.user_params = step.user_params;
            }
            return s;
        });
        renderExperimentSteps();
        updateExperimentJSON();
    } catch (e) {
        alert('加载实验设计失败: ' + e.message);
    }
}

/** 使用系统原生保存对话框将实验设计保存为 JSON 文件。优先使用 File System Access API，不支持时回退到浏览器下载。同时同步保存到服务端会话目录。 */
async function saveExperimentDesign() {
    if (experimentSteps.length === 0) {
        showNotification('请先添加实验步骤', 'warning');
        return;
    }

    updateExperimentJSON();

    const jsonData = {
        experiment_name: experimentName,
        created_at: new Date().toISOString(),
        steps: experimentSteps
    };
    const jsonStr = JSON.stringify(jsonData, null, 2);
    const filename = `${experimentName.replace(/[\\/:*?"<>|]/g, '_')}.json`;

    // 同步保存到服务端会话目录（fire-and-forget）
    fetch('/api/save_experiment_design', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonData)
    }).catch(() => {});

    // 优先使用原生保存对话框
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: filename,
                types: [{ description: 'JSON 文件', accept: { 'application/json': ['.json'] } }]
            });
            const writable = await handle.createWritable();
            await writable.write(jsonStr);
            await writable.close();
            showNotification('实验设计已保存', 'success');
            return;
        } catch (e) {
            if (e.name === 'AbortError') return;
        }
    }

    // 回退：浏览器下载
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    showNotification('实验设计已下载', 'success');
}

/** 打开文件选择器导入实验设计 JSON 文件。优先使用原生对话框，回退到隐藏的 file input。 */
async function importExperimentDesign() {
    if (window.showOpenFilePicker) {
        try {
            const [handle] = await window.showOpenFilePicker({
                types: [{ description: 'JSON 文件', accept: { 'application/json': ['.json'] } }]
            });
            const file = await handle.getFile();
            const text = await file.text();
            const json = JSON.parse(text);
            loadExperimentFromJSON(json);
            showNotification('实验设计已导入', 'success');
        } catch (e) {
            if (e.name === 'AbortError') return;
            showNotification('导入失败: ' + e.message, 'error');
        }
    } else {
        document.getElementById('import-experiment-file').click();
    }
}

/** 处理隐藏 file input 的文件选择（非原生对话框回退方案）。 */
function handleImportExperimentFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function() {
        try {
            const json = JSON.parse(reader.result);
            loadExperimentFromJSON(json);
            showNotification('实验设计已导入', 'success');
        } catch (e) {
            showNotification('JSON 解析失败: ' + e.message, 'error');
        }
    };
    reader.readAsText(file);
    event.target.value = '';
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

/** 将当前实验设计编译为Python代码并显示在聊天区。 */
async function compileExperiment() {
    if (experimentSteps.length === 0) { alert('请先添加实验步骤'); return; }

    appendMessage(`🔧 正在编译实验: ${experimentName}`, 'user');

    try {
        const res = await fetch('/api/compile_experiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment_json: { experiment_name: experimentName, created_at: new Date().toISOString(), steps: experimentSteps }
            })
        });
        const data = await res.json();
        if (data.success) {
            appendMessage(`✅ 编译成功！生成的Python代码：\n\`\`\`python\n${data.code}\n\`\`\``, 'ai');
        } else {
            appendMessage(`❌ 编译失败: ${data.message}`, 'ai');
        }
    } catch (e) {
        appendMessage(`❌ 编译异常: ${e.message}`, 'ai');
    }
}

/** 将当前实验设计编译为Python代码并立即执行。 */
async function compileAndRunExperiment() {
    if (experimentSteps.length === 0) { alert('请先添加实验步骤'); return; }
    if (!confirm(`确定编译并运行实验 "${experimentName}"？\n共 ${experimentSteps.length} 个步骤`)) return;

    appendMessage(`⚡ 正在编译并运行实验: ${experimentName}`, 'user');

    try {
        const res = await fetch('/api/compile_and_run_experiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment_json: { experiment_name: experimentName, created_at: new Date().toISOString(), steps: experimentSteps }
            })
        });
        const data = await res.json();
        if (data.success) {
            let msg = `✅ 执行成功！\n\n**生成的代码：**\n\`\`\`python\n${data.code}\n\`\`\`\n`;
            if (data.output) msg += `\n**执行输出：**\n\`\`\`\n${data.output}\n\`\`\``;
            appendMessage(msg, 'ai');
        } else {
            let msg = `❌ 执行失败: ${data.message}\n`;
            if (data.code) msg += `\n**生成的代码：**\n\`\`\`python\n${data.code}\n\`\`\`\n`;
            if (data.error) msg += `\n**错误信息：**\n\`\`\`\n${data.error}\n\`\`\``;
            appendMessage(msg, 'ai');
        }
    } catch (e) {
        appendMessage(`❌ 执行异常: ${e.message}`, 'ai');
    }
}
