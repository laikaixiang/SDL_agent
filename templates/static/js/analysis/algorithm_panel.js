/**
 * algorithm_panel.js — 左侧算法库面板
 *
 * 交互模式：
 *   单击标题区 → 选中高亮（selectedAlgorithm 更新）
 *   双击标题区 → 展开/收起详情
 *   详情区包含：输入文件选择 + 输出目录选择
 *
 * 依赖：state.js, ui/panel.js, ui/menu.js, ui/input_state.js,
 *       notification.js, analysis/analysis.js（executeAlgorithm）,
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
        const title = algo.chinese_name || algo.description || algo.name;
        return `
        <div class="algo-panel-item" id="algo-item-${algo.name}">
            <div class="algo-item-row">
                <button class="algo-add-btn" title="添加至实验设计"
                    onclick="addAlgoToExperiment('${safe(algo.name)}', '${safe(algo.description || algo.name)}', '${safe(algo.icon || '📊')}')">+</button>
                <div class="algo-item-info"
                    onclick="_algoSingleClick(event, '${safe(algo.name)}')"
                    ondblclick="_algoDoubleClick(event, '${safe(algo.name)}')">
                    <div class="algo-panel-item-name">${title}</div>
                </div>
                <button class="algo-expand-btn" title="展开详情"
                    onclick="toggleAlgoDetail('${safe(algo.name)}')">›</button>
            </div>
            <div class="algo-item-detail" id="algo-detail-${algo.name}">
                <div class="algo-item-detail-desc">${algo.description || ''}</div>
                <div class="algo-item-detail-tags">${tagsHtml}</div>
                <div class="algo-file-pickers">
                    <div class="algo-picker-row">
                        <span class="algo-picker-label">输入文件</span>
                        <span class="algo-picker-value" id="algo-input-label-${algo.name}">未选择</span>
                        <button class="algo-picker-btn" onclick="_openInputPickerModal('${safe(algo.name)}')">选择</button>
                    </div>
                    <div class="algo-picker-row">
                        <span class="algo-picker-label">输出目录</span>
                        <span class="algo-picker-value" id="algo-output-label-${algo.name}">默认</span>
                        <button class="algo-picker-btn" onclick="_openOutputPickerModal('${safe(algo.name)}')">选择</button>
                    </div>
                </div>
            </div>
        </div>`;
    }).join('');
}

// 单击计时器，用于区分单击和双击
const _algoClickTimers = {};

/** 单击：选中高亮（延迟执行，双击时取消）。 */
function _algoSingleClick(event, algoName) {
    clearTimeout(_algoClickTimers[algoName]);
    _algoClickTimers[algoName] = setTimeout(() => {
        _selectAlgoItem(algoName);
    }, 220);
}

/** 双击：展开/收起详情，取消单击计时。 */
function _algoDoubleClick(event, algoName) {
    clearTimeout(_algoClickTimers[algoName]);
    toggleAlgoDetail(algoName);
}

/** 选中某算法，高亮并更新 selectedAlgorithm。 */
function _selectAlgoItem(algoName) {
    document.querySelectorAll('.algo-panel-item.selected').forEach(el => el.classList.remove('selected'));
    const item = document.getElementById('algo-item-' + algoName);
    if (item) item.classList.add('selected');
    selectedAlgorithm = algoName;
}

/** 切换算法详情展开/收起，同步旋转展开箭头。 */
function toggleAlgoDetail(algoName) {
    const item = document.getElementById('algo-item-' + algoName);
    if (!item) return;
    item.classList.toggle('expanded');
    const btn = item.querySelector('.algo-expand-btn');
    if (btn) btn.classList.toggle('expanded');
    // 展开时初始化 picker 默认值
    if (item.classList.contains('expanded')) _initPickerDefaults(algoName);
}

/** 展开时从后端获取默认输出路径并填入标签。 */
async function _initPickerDefaults(algoName) {
    const outputLabel = document.getElementById('algo-output-label-' + algoName);
    if (!outputLabel || outputLabel.dataset.initialized) return;
    outputLabel.dataset.initialized = '1';
    try {
        const res = await fetch('/api/get_session_path?subdir=results');
        const data = await res.json();
        if (data.success) {
            outputLabel.textContent = data.path.split('/').pop() || data.path;
            outputLabel.title = data.path;
            outputLabel.dataset.value = data.path;
        }
    } catch (_) {}
}

/** 打开输入文件选择器（复用主模态框）。 */
function _openInputPickerModal(algoName) {
    showFileSelector((filePath, fileName) => {
        const label = document.getElementById('algo-input-label-' + algoName);
        if (label) { label.textContent = fileName; label.title = filePath; label.dataset.value = filePath; }
        selectedFilePath = filePath;
    });
}

/** 打开输出目录选择器（复用主模态框）。 */
function _openOutputPickerModal(algoName) {
    showOutputDirSelector((dirPath, dirLabel) => {
        const label = document.getElementById('algo-output-label-' + algoName);
        if (label) { label.textContent = dirLabel; label.title = dirPath; label.dataset.value = dirPath; }
    });
}

/** 双击算法时添加到实验设计步骤列表。 */
function addAlgoToExperiment(algoName, description, icon) {
    experimentSteps.push({ type: 'software', name: algoName, params: {}, description });
    if (typeof renderExperimentSteps === 'function') renderExperimentSteps();
    showNotification(`已添加 ${description} 至实验设计`, 'success');
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
