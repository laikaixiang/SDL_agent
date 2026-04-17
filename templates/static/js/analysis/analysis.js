/**
 * analysis.js — 数据分析与算法交互逻辑
 *
 * 处理数据分析模式：展示算法列表、解析用户输入的算法名、
 * 文件选择器、执行算法，以及显示执行结果。
 *
 * 依赖：state.js, ui/input_state.js, hardware/task_stream.js（startTaskStream）
 */

/**
 * 数据分析模式入口：输入为空时展示算法列表，有输入时调用 /api/parse_algorithm 解析算法名。
 * @param {string} input - 用户输入文字
 */
async function handleAnalyzeMode(input) {
    if (!input || !input.trim()) {
        try {
            const res = await fetch('/api/list_algorithms');
            const data = await res.json();
            if (data.success) displayAlgorithmList(data.algorithms);
            else appendMessage('获取算法列表失败', 'ai');
        } catch (e) {
            appendMessage('网络异常：' + e.message, 'ai');
        }
    } else {
        try {
            const res = await fetch('/api/parse_algorithm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: input })
            });
            const data = await res.json();
            if (data.algorithm_found) {
                displayAlgorithmConfirm(data.algorithm, data.description, data.icon, data.tags);
            } else {
                appendMessage('未识别到算法名称，请从列表中选择：', 'ai');
                displayAlgorithmList(data.available_algorithms);
            }
        } catch (e) {
            appendMessage('网络异常：' + e.message, 'ai');
        }
    }
}

/** 将算法列表渲染为卡片网格，每行显示图标、名称、标签和运行按钮，追加到聊天框。 */
function displayAlgorithmList(algorithms) {
    let html = `<div class="algorithm-list-card">
        <div class="card-header">📊 可用算法列表</div>
        <div class="card-hint">请选择要执行的算法</div>
        <div class="algorithm-grid">`;

    algorithms.forEach(algo => {
        const tagsHtml = (algo.tags || []).map(t => `<span class="algo-tag">${t}</span>`).join('');
        html += `
        <div class="algo-item">
            <div class="algo-icon">${algo.icon || '📊'}</div>
            <div class="algo-content">
                <div class="algo-name">${algo.description || algo.name}</div>
                <div class="algo-desc">${algo.name}</div>
                <div class="algo-tags">${tagsHtml}</div>
            </div>
            <button class="algo-run-btn" onclick="selectAlgorithm('${algo.name}', '${algo.description}', '${algo.icon}')">▶</button>
        </div>`;
    });

    html += `</div></div>`;
    appendMessageHtml(html, 'ai');
}

/** 渲染算法确认卡片，显示已选算法的图标、名称、标签，以及"选择文件并运行"和"取消"按钮。 */
function displayAlgorithmConfirm(algoName, description, icon, tags) {
    const tagsHtml = (tags || []).map(t => `<span class="algo-tag">${t}</span>`).join('');
    const html = `
    <div class="algorithm-confirm-card">
        <div class="card-header">🎯 算法已选择</div>
        <div class="selected-algo">
            <div class="algo-badge">${icon || '📊'} ${description || algoName}</div>
            <div class="algo-description">${algoName}</div>
            <div class="algo-tags" style="margin-top:8px;">${tagsHtml}</div>
        </div>
        <div class="agent-actions">
            <button class="btn-yes" onclick="runAlgorithmWithFile('${algoName}')">📂 选择数据文件并运行</button>
            <button class="btn-no" onclick="cancelAlgorithm()">✗ 取消</button>
        </div>
    </div>`;
    appendMessageHtml(html, 'ai');
}

/** 将选中的算法名存入 selectedAlgorithm，然后打开文件选择器模态框。 */
function selectAlgorithm(algoName, description, icon) {
    selectedAlgorithm = algoName;
    showFileSelector();
}

/** 同 selectAlgorithm，供算法面板调用。 */
function runAlgorithmWithFile(algoName) {
    selectedAlgorithm = algoName;
    showFileSelector();
}

/** 取消算法选择。 */
function cancelAlgorithm() {
    appendMessage('已取消算法执行', 'user');
    selectedAlgorithm = null;
}

/** 显示文件选择器模态框，并异步加载最近使用的文件列表到"最近使用"标签页。 */
async function showFileSelector() {
    document.getElementById('file-selector-modal').style.display = 'flex';
    try {
        const res = await fetch('/api/recent_files');
        const data = await res.json();
        if (data.success) renderRecentFiles(data.files);
    } catch (e) {
        document.getElementById('file-list-recent').innerHTML =
            '<div style="text-align:center;color:#ef4444;padding:20px;">加载失败</div>';
    }
}

/** 将文件列表渲染为可点击的文件行，每行显示文件名、路径和最后修改时间。 */
function renderRecentFiles(files) {
    const container = document.getElementById('file-list-recent');
    if (!files || files.length === 0) {
        container.innerHTML = '<div style="text-align:center;color:#9ca3af;padding:20px;">暂无最近使用的文件</div>';
        return;
    }
    container.innerHTML = files.map(file => `
        <div class="file-item" onclick="selectFile('${file.path}')">
            <span class="file-icon">📄</span>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-path">${file.path}</div>
                <div class="file-meta">最后修改: ${file.modified_str}</div>
            </div>
            <span class="file-check">✓</span>
        </div>`).join('');
}

/** 关闭文件选择器。 */
function closeFileSelector() {
    document.getElementById('file-selector-modal').style.display = 'none';
}

/** 切换文件选择器的标签页（recent / browse / custom），更新按钮激活状态并显示对应内容区。 */
function switchFileTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    ['recent', 'browse', 'custom'].forEach(t => {
        document.getElementById(`file-list-${t}`).style.display = t === tabName ? 'block' : 'none';
    });
}

/** 将选中的文件路径存入 selectedFilePath，关闭选择器，然后立即执行当前选中的算法。 */
function selectFile(filePath) {
    selectedFilePath = filePath;
    closeFileSelector();
    executeAlgorithm(selectedAlgorithm, selectedFilePath, {});
}

/** CSV 文件上传处理（浏览标签页）。 */
function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    appendMessage(`已选择文件: ${file.name}`, 'ai');
    closeFileSelector();
}

/** 确认自定义路径输入。 */
function confirmCustomPath() {
    const path = document.getElementById('custom-path-input').value.trim();
    if (!path) { alert('请输入文件路径'); return; }
    selectFile(path);
}

/** 调用 /api/run_algorithm 提交算法执行请求，成功后启动 SSE 监听任务进度。 */
async function executeAlgorithm(algoName, filePath, params) {
    appendMessage(`正在执行算法 ${algoName}，数据文件: ${filePath}...`, 'ai');
    try {
        const res = await fetch('/api/run_algorithm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ algorithm: algoName, file_path: filePath, params })
        });
        const data = await res.json();
        if (data.success) {
            startTaskStream();
        } else {
            appendMessage(`❌ 执行失败: ${data.message}`, 'ai');
        }
    } catch (e) {
        appendMessage(`❌ 网络异常: ${e.message}`, 'ai');
    }
}

/** 将算法执行结果渲染为分析卡片，递归展开嵌套对象为表格行，追加到聊天框。 */
function displayAnalysisResult(data) {
    const result = data.result || {};
    let rowsHtml = '';

    function flattenObject(obj, prefix = '') {
        for (const [key, value] of Object.entries(obj)) {
            const fullKey = prefix ? `${prefix}.${key}` : key;
            if (value && typeof value === 'object' && !Array.isArray(value)) {
                flattenObject(value, fullKey);
            } else {
                const display = Array.isArray(value)
                    ? `[${value.length} items]`
                    : (typeof value === 'number' ? value.toFixed(4) : String(value));
                rowsHtml += `<tr><td>${fullKey}</td><td>${display}</td></tr>`;
            }
        }
    }
    flattenObject(result);

    appendMessageHtml(`
    <div class="analysis-card">
        <div class="analysis-algo">✅ ${data.algorithm} 执行完成</div>
        <div class="analysis-reason">数据文件: ${data.file_path}</div>
        <table class="analysis-table">${rowsHtml}</table>
        <div class="analysis-filepath">
            📁 最新结果: ${data.output_path_latest}<br>
            📁 存档结果: ${data.output_path_archive}
        </div>
    </div>`, 'ai');
}
