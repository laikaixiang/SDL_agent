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
    if (typeof updateExperimentJSON === 'function') updateExperimentJSON();
    showNotification(`已添加 ${description} 至实验设计`, 'success');
}

/** 当前引导会话 ID。 */
let _guideSessionId = null;

/** 打开算法生成器 — 调后端取第一个引导问题，激活引导模式。 */
async function openAlgorithmGenerator() {
    closeAlgorithmPanel();
    _guideSessionId = null;

    try {
        var res = await fetch('/api/algorithm_gen/guide', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        var data = await res.json();
        if (data.stage === 'question') {
            _guideSessionId = data.session_id;
            _renderGuideCard(data.reply, data.progress);
            // 激活引导模式：主输入框用于填写答案
            window._guideMode = true;
            userInput.placeholder = '在此输入你的回答（可跳过）';
            userInput.value = '';
            userInput.focus();
        } else {
            appendMessage(data.reply || '启动算法生成失败', 'ai');
        }
    } catch (e) {
        appendMessage('网络异常，请重试', 'ai');
    }
}

/** 渲染引导问题卡片（无 textarea，用户在主输入框打字）。 */
function _renderGuideCard(reply, progress) {
    var pct = progress === 'complete' ? 100 : parseInt(progress) / 4 * 100;
    var progressText = progress === 'complete' ? '完成' : progress;

    appendMessageHtml(
        '<div class="guide-card" id="guide-card">' +
        '<div class="guide-progress-bar"><div class="guide-progress-fill" style="width:' + pct + '%"></div></div>' +
        '<div class="guide-progress-label">' + progressText + '</div>' +
        '<div class="guide-reply">' + _escapeHtml(reply).replace(/\n/g, '<br>') + '</div>' +
        '<div class="agent-actions" style="margin-top:12px;">' +
        '<button class="btn-no" onclick="cancelAlgorithmGeneration()">取消</button>' +
        '<button class="btn-back" onclick="_guideGoBack()">返回</button>' +
        '<button class="btn-yes" onclick="_guideSubmitCurrent()">提交</button>' +
        '</div>' +
        '</div>', 'ai');
}

/** 提交当前主输入框中的答案（由提交按钮触发）。 */
function _guideSubmitCurrent() {
    if (!_guideSessionId) return;
    var text = userInput.value.trim();
    _guideSend(text);
}

/** 由 chat.js 调用：接收主输入框文本，交给引导流程处理。 */
async function handleGuideSend(text) {
    await _guideSend(text);
}

/** 发送答案到引导 API，更新卡片或显示最终结果。 */
async function _guideSend(text) {
    if (!_guideSessionId) return;
    var answer = text;

    // 显示用户回答
    appendMessage(answer || '（跳过）', 'user');
    userInput.value = '';
    _removeGuideCard();

    try {
        var res = await fetch('/api/algorithm_gen/guide', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: _guideSessionId, answer: answer, action: 'answer' }),
        });
        var data = await res.json();

        if (data.stage === 'question') {
            _renderGuideCard(data.reply, data.progress);
            userInput.focus();
        } else if (data.stage === 'done') {
            // 后端返回的完整结果，前端不构造任何 AI 回应文本
            _exitGuideMode();
            appendMessage(data.reply, 'ai');
            if (document.getElementById('app-wrapper').classList.contains('algorithm-mode')) {
                await loadAlgorithmList();
            }
        }
    } catch (e) {
        _exitGuideMode();
        appendMessage('网络异常，请重试', 'ai');
    }
}

/** 返回到上一个问题。 */
async function _guideGoBack() {
    if (!_guideSessionId) return;

    try {
        var res = await fetch('/api/algorithm_gen/guide', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: _guideSessionId, action: 'back' }),
        });
        var data = await res.json();

        if (data.stage === 'question') {
            _removeGuideCard();
            _renderGuideCard(data.reply, data.progress);
            // 恢复之前的答案到输入框，方便用户修改
            if (data.previous_answer) {
                userInput.value = data.previous_answer;
            }
            userInput.focus();
        }
    } catch (e) {
        // 忽略网络错误，不影响当前引导流程
    }
}

/** 从 DOM 中移除引导卡片。 */
function _removeGuideCard() {
    var card = document.getElementById('guide-card');
    if (card) card.remove();
}

/** 退出引导模式，恢复输入框默认状态。 */
function _exitGuideMode() {
    window._guideMode = false;
    _guideSessionId = null;
    userInput.placeholder = '输入问题或指令...';
}

/** 简单 HTML 转义。 */
function _escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/** 取消算法生成 — 通知后端清理，退出引导模式。 */
function cancelAlgorithmGeneration() {
    _removeGuideCard();
    if (_guideSessionId) {
        fetch('/api/algorithm_gen/guide', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: _guideSessionId, action: 'cancel' }),
        }).catch(function () { });
    }
    _exitGuideMode();
    appendMessage('已取消算法生成', 'user');
}
