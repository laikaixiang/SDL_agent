/**
 * algorithm_panel.js — 左侧算法库面板
 *
 * 管理算法面板的开关、列表渲染、详情展开，
 * 以及通过 LLM 生成新算法的交互流程。
 *
 * 依赖：state.js, ui/panel.js, ui/menu.js, ui/input_state.js,
 *       notification.js, analysis/analysis.js（showFileSelector）,
 *       experiment/experiment_design.js（experimentSteps, renderExperimentSteps）
 */

/** 打开算法面板并加载算法列表。 */
async function openAlgorithmPanel() {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();
    document.getElementById('app-wrapper').classList.add('algorithm-mode');
    await loadAlgorithmList();
}

/** 关闭算法面板。 */
function closeAlgorithmPanel() {
    document.getElementById('app-wrapper').classList.remove('algorithm-mode');
}

/** 从 /api/list_algorithms 加载并渲染算法列表。 */
async function loadAlgorithmList() {
    const container = document.getElementById('algo-list-container');
    container.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:40px 20px;">加载中...</div>';
    try {
        const res = await fetch('/api/list_algorithms');
        const data = await res.json();
        if (data.success && data.algorithms) {
            renderAlgorithmPanelList(data.algorithms);
        } else {
            container.innerHTML = '<div style="text-align:center;color:#ef4444;padding:20px;">加载失败</div>';
        }
    } catch (e) {
        let errMsg = '未知错误';
        if (e instanceof TypeError && e.message.includes('fetch')) errMsg = '网络连接失败（服务器未响应）';
        else if (e instanceof SyntaxError) errMsg = '数据解析错误（响应格式异常）';
        else if (e.name === 'AbortError') errMsg = '请求超时';
        else errMsg = e.message || String(e);
        container.innerHTML = `<div style="text-align:center;color:#ef4444;padding:20px;">错误：${errMsg}</div>`;
    }
}

/** 渲染算法面板列表 HTML。 */
function renderAlgorithmPanelList(algorithms) {
    const container = document.getElementById('algo-list-container');
    if (!algorithms || algorithms.length === 0) {
        container.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:20px;">暂无算法</div>';
        return;
    }

    container.innerHTML = algorithms.map(algo => {
        const tagsHtml = (algo.tags || []).map(t => `<span class="algo-panel-tag">${t}</span>`).join('');
        const safe = s => (s || '').replace(/'/g, "\\'");
        return `
        <div class="algo-panel-item" id="algo-item-${algo.name}"
             onclick="toggleAlgoDetail('${safe(algo.name)}')"
             ondblclick="addAlgoToExperiment('${safe(algo.name)}', '${safe(algo.description || algo.name)}', '${safe(algo.icon || '📊')}')">
            <div class="algo-item-row">
                <div class="algo-panel-item-icon">${algo.icon || '📊'}</div>
                <div class="algo-item-info">
                    <div class="algo-panel-item-name">${algo.description || algo.name}</div>
                    <div class="algo-panel-item-desc">${algo.name}</div>
                </div>
                <button class="algo-arrow-btn" title="选择数据文件"
                    onclick="event.stopPropagation(); selectAlgorithmFromPanel('${safe(algo.name)}', '${safe(algo.description || algo.name)}', '${safe(algo.icon || '📊')}')">›</button>
            </div>
            <div class="algo-item-detail">
                <div class="algo-item-detail-tags">${tagsHtml}</div>
                <div class="algo-item-hint">双击算法可添加至实验设计</div>
            </div>
        </div>`;
    }).join('');
}

/** 切换算法详情展开/收起。 */
function toggleAlgoDetail(algoName) {
    document.getElementById('algo-item-' + algoName)?.classList.toggle('expanded');
}

/** 双击算法时添加到实验设计步骤列表。 */
function addAlgoToExperiment(algoName, description, icon) {
    experimentSteps.push({ type: 'tool', name: algoName, params: {}, description });
    if (typeof renderExperimentSteps === 'function') renderExperimentSteps();
    showNotification(`已添加 ${description} 至实验设计`, 'success');
}

/** 点击算法右侧箭头按钮，打开文件选择器执行算法。 */
function selectAlgorithmFromPanel(algoName) {
    selectedAlgorithm = algoName;
    showFileSelector();
}

/** 打开算法生成器（在聊天区显示描述输入框）。 */
function openAlgorithmGenerator() {
    closeAlgorithmPanel();
    appendMessageHtml(`
    <div class="algorithm-confirm-card">
        <div class="card-header">✨ 生成新算法</div>
        <div style="margin-bottom:16px;color:#6b7280;font-size:0.9rem;">请描述您想要的算法功能，系统将自动生成代码。</div>
        <div style="margin-bottom:16px;">
            <textarea id="algo-gen-input"
                style="width:100%;min-height:120px;padding:10px;border:1px solid #d1d5db;border-radius:8px;font-size:0.9rem;resize:vertical;"
                placeholder="例如：我需要一个对数值列表做移动平均的算法，窗口大小可配置，默认5，输出平滑后的序列和残差"></textarea>
        </div>
        <div class="agent-actions">
            <button class="btn-yes" onclick="generateNewAlgorithm()">✨ 生成算法</button>
            <button class="btn-no" onclick="cancelAlgorithmGeneration()">✗ 取消</button>
        </div>
    </div>`, 'ai');
}

/** 调用 /api/generate_algorithm 生成算法代码，成功后刷新面板列表。 */
async function generateNewAlgorithm() {
    const input = document.getElementById('algo-gen-input');
    if (!input) return;
    const description = input.value.trim();
    if (!description) { appendMessage('请输入算法描述', 'ai'); return; }

    appendMessage(`正在生成算法：${description}`, 'user');
    appendMessage('⏳ 正在调用 LLM 生成算法代码，请稍候...', 'ai');

    try {
        const res = await fetch('/api/generate_algorithm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description })
        });
        const data = await res.json();
        if (data.success) {
            appendMessage(`✅ 算法生成成功！\n算法名称: ${data.name}\n文件路径: ${data.filepath}\n\n${data.message}`, 'ai');
            // 面板打开时刷新列表（算法面板用 algorithm-mode class 标识是否打开）
            if (document.getElementById('app-wrapper').classList.contains('algorithm-mode')) {
                await loadAlgorithmList();
            }
        } else {
            appendMessage(`❌ 算法生成失败: ${data.message}`, 'ai');
        }
    } catch (e) {
        appendMessage(`❌ 网络异常: ${e.message}`, 'ai');
    }
}

/** 取消算法生成。 */
function cancelAlgorithmGeneration() {
    appendMessage('已取消算法生成', 'user');
}
