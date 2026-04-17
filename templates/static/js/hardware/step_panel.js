/**
 * step_panel.js — 单步控制面板
 *
 * 提供硬件工具的单步执行界面：加载工具列表、渲染参数表单、
 * 执行单个工具调用，以及在实验设计模式下双击添加步骤。
 *
 * 依赖：state.js, ui/panel.js, ui/menu.js, ui/input_state.js,
 *       experiment/experiment_design.js（addStepToExperiment, renderExperimentSteps）
 */

/** 显示单步控制面板（添加 open class 和 step-control-mode），首次打开时从服务器加载工具列表。 */
async function openStepPanel() {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();

    document.getElementById('step-control-panel').classList.add('open');
    document.getElementById('app-wrapper').classList.add('step-control-mode');
    bringPanelToFront('step-control-panel');

    if (stepPanelTools.length === 0) {
        await fetchStepTools();
    } else {
        renderStepTools();
    }
}

/** 关闭单步控制面板；若当前处于实验设计模式，则转交给 closeExperimentDesignPanel() 统一关闭。 */
function closeStepPanel() {
    const wrapper = document.getElementById('app-wrapper');
    if (wrapper.classList.contains('experiment-design-mode')) {
        closeExperimentDesignPanel();
        return;
    }
    const panel = document.getElementById('step-control-panel');
    panel.classList.remove('open', 'collapsed');
    removePanelFromTracking('step-control-panel');
    wrapper.classList.remove('step-control-mode');
}

/** 请求 /api/hardware_tools 获取工具列表，存入 stepPanelTools 并调用 renderStepTools() 渲染。 */
async function fetchStepTools() {
    const body = document.getElementById('step-panel-body');
    body.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:40px 20px;">加载中...</div>';
    try {
        const res = await fetch('/api/hardware_tools');
        const data = await res.json();
        stepPanelTools = data.tools || [];
        renderStepTools();
    } catch (e) {
        body.innerHTML = '<div style="text-align:center;color:#ef4444;padding:20px;">加载失败，请刷新重试</div>';
    }
}

/** 将 stepPanelTools 渲染为工具行列表 HTML；实验设计模式下双击工具触发 addStepToExperiment，否则触发展开参数。 */
function renderStepTools() {
    const body = document.getElementById('step-panel-body');
    if (stepPanelTools.length === 0) {
        body.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:20px;">没有可用的硬件工具</div>';
        return;
    }

    const isExpDesignMode = document.getElementById('experiment-design-panel').classList.contains('open');
    let html = '';
    for (const tool of stepPanelTools) {
        const isExpanded = stepExpandedTool === tool.name;
        const dblClickHandler = isExpDesignMode
            ? `addStepToExperiment('${tool.name}')`
            : `toggleToolExpand('${tool.name}')`;

        html += `
<div class="tool-row ${isExpanded ? 'expanded' : ''}" id="tool-row-${tool.name}" ondblclick="${dblClickHandler}">
    <span class="tool-arrow" onclick="event.stopPropagation(); toggleToolExpand('${tool.name}')">▶</span>
    <div style="flex:1">
        <div class="tool-name">${tool.name}</div>
        <div class="tool-desc">${tool.description}</div>
    </div>
    <button class="run-btn" id="run-btn-${tool.name}"
            onclick="event.stopPropagation(); runTool('${tool.name}')"
            title="执行 ${tool.name}">▶</button>
</div>
<div class="tool-params-area ${isExpanded ? 'visible' : ''}" id="params-area-${tool.name}">
    ${renderParamFields(tool)}
</div>`;
    }
    body.innerHTML = html;
    if (stepRunning) disableAllRunBtns();
}

/** 为指定工具生成参数输入表单 HTML，每个参数含必填/可选徽章、类型提示和默认值。 */
function renderParamFields(tool) {
    if (Object.keys(tool.params).length === 0) {
        return '<div style="color:#6b7280;font-size:0.85rem;font-style:italic;">此工具无需参数，直接点击 ▶ 执行</div>';
    }
    let html = '';
    for (const [pName, pInfo] of Object.entries(tool.params)) {
        const badge = pInfo.required
            ? '<span class="param-badge-req">必填</span>'
            : '<span class="param-badge-opt">可选</span>';
        const defaultVal = pInfo.default !== null && pInfo.default !== undefined ? pInfo.default : '';
        html += `
<div class="param-field">
    <div class="param-label">${pName} ${badge}</div>
    <div class="param-hint">${pInfo.description}（${pInfo.type}${defaultVal !== '' ? ` 默认:${defaultVal}` : ''}）</div>
    <input class="param-input" id="param-${tool.name}-${pName}" type="text"
           placeholder="${defaultVal}" value="${defaultVal}">
</div>`;
    }
    return html;
}

/** 切换指定工具的参数区域展开/收起状态，同一工具再次点击则收起。 */
function toggleToolExpand(toolName) {
    stepExpandedTool = stepExpandedTool === toolName ? null : toolName;
    renderStepTools();
}

/** 收集指定工具的表单参数，校验必填项，然后调用 /api/chat start_hardware 执行单步硬件操作。 */
async function runTool(toolName) {
    if (stepRunning) return;
    const tool = stepPanelTools.find(t => t.name === toolName);
    if (!tool) return;

    // 收集并类型转换参数（共享逻辑见 state.js collectToolParams）
    const params = collectToolParams(tool);

    // 必填校验
    for (const [pName, pInfo] of Object.entries(tool.params)) {
        if (pInfo.required && (params[pName] === '' || params[pName] === undefined)) {
            setStepStatus(`❌ 参数 "${pName}" 为必填项`, false);
            return;
        }
    }

    stepRunning = true;
    disableAllRunBtns(toolName);
    setStepStatus(`⏳ 正在执行 ${toolName}...`, true);

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start_hardware', tool_calls: [{ name: toolName, params }] })
        });
        const data = await res.json();
        if (data.status === 'success') {
            setStepStatus(`✅ ${toolName} 执行成功`, false);
            appendMessage(`✅ 单步执行完成 [${toolName}]: ${data.reply || ''}`, 'ai');
        } else {
            setStepStatus(`❌ 执行失败: ${data.reply || '未知错误'}`, false);
            appendMessage(`❌ 单步执行失败 [${toolName}]: ${data.reply || data.error || ''}`, 'ai');
        }
    } catch (e) {
        setStepStatus('❌ 网络错误', false);
    } finally {
        stepRunning = false;
        enableAllRunBtns();
    }
}

/** 禁用所有工具的运行按钮，当前执行中的工具按钮显示加载动画。 */
function disableAllRunBtns(activeToolName) {
    for (const tool of stepPanelTools) {
        const btn = document.getElementById(`run-btn-${tool.name}`);
        if (!btn) continue;
        if (tool.name === activeToolName) {
            btn.classList.add('running');
            btn.textContent = '⏳';
        } else {
            btn.disabled = true;
        }
    }
}

/** 恢复所有工具运行按钮为可点击状态，清除加载动画。 */
function enableAllRunBtns() {
    for (const tool of stepPanelTools) {
        const btn = document.getElementById(`run-btn-${tool.name}`);
        if (!btn) continue;
        btn.disabled = false;
        btn.classList.remove('running');
        btn.textContent = '▶';
    }
}

/** 更新底部状态栏文字，isRunning=true 时添加 running 样式（黄色背景）。 */
function setStepStatus(msg, isRunning) {
    const bar = document.getElementById('step-status-bar');
    if (!bar) return;
    bar.textContent = msg;
    bar.className = 'step-status-bar' + (isRunning ? ' running' : '');
}
