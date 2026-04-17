/**
 * task_stream.js — SSE 任务流监听与中断
 *
 * 文献提取、数据分析、实验执行等长任务通过 SSE 推送进度。
 * startTaskStream() 建立连接并处理各类事件消息。
 *
 * 依赖：state.js, ui/input_state.js, ui/panel.js,
 *       experiment/experiment_confirm.js（renderExperimentConfirm）
 */

/** 建立 SSE 连接到 /api/task_stream，创建任务卡片并监听进度事件，任务完成或出错时关闭连接并恢复 UI。 */
function startTaskStream() {
    setTaskRunningState(true);

    const eventSource = new EventSource('/api/task_stream');

    // 创建任务卡片
    const row = document.createElement('div');
    row.className = 'message-row ai';
    const card = document.createElement('div');
    card.className = 'task-card';
    card.innerHTML = `<div class="task-log" id="current-log">准备连接解析引擎...</div><div id="findings-container"></div>`;
    row.appendChild(card);
    chatBox.appendChild(row);

    const logEl      = card.querySelector('#current-log');
    const findingsEl = card.querySelector('#findings-container');
    scrollToBottom();

    eventSource.onmessage = function(event) {
        const msg = JSON.parse(event.data);

        if (msg.type === 'info' || msg.type === 'progress') {
            logEl.textContent = `> ${msg.data || ''}`;

        } else if (msg.type === 'page_reading' && msg.data) {
            // 文献提取时显示 PDF 预览（实验设计模式下不打开 PDF 面板）
            const wrapper = document.getElementById('app-wrapper');
            if (!wrapper.classList.contains('experiment-design-mode')) {
                wrapper.classList.add('pdf-mode');
                document.getElementById('pdf-viewer').classList.add('is-scanning');
                document.getElementById('pdf-title').textContent =
                    `${msg.data.filename || ''} (第 ${msg.data.page || ''} 页)`;
                bringPanelToFront('pdf-panel');
            }
            const imgEl = document.getElementById('pdf-image');
            if (msg.data.image) {
                imgEl.src = 'data:image/jpeg;base64,' + msg.data.image;
                imgEl.style.opacity = '1';
            }

        } else if (msg.type === 'experiment_confirm') {
            renderExperimentConfirm(msg);

        } else if (msg.type === 'analysis_result' && msg.data) {
            _renderAnalysisResult(msg.data);

        } else if (msg.type === 'finding' && msg.data) {
            _renderFinding(msg.data, findingsEl);

        } else if (msg.type === 'complete') {
            _handleTaskComplete(msg, logEl, eventSource);
        }

        scrollToBottom();
    };

    eventSource.onerror = function() {
        logEl.textContent = '⚠️ 连接断开。';
        eventSource.close();
        setTaskRunningState(false);
    };
}

/** 将 SSE analysis_result 事件数据渲染为分析结果卡片并追加到聊天框。 */
function _renderAnalysisResult(d) {
    const summary = d.result_summary || {};
    let rowsHtml = '';
    for (const [k, v] of Object.entries(summary)) {
        rowsHtml += `<tr><td>${k}</td><td>${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`;
    }
    const card = document.createElement('div');
    card.className = 'message-row ai';
    card.innerHTML = `<div class="analysis-card">
        <div class="analysis-algo">📊 ${d.algorithm}</div>
        <div class="analysis-reason">${d.reasoning}</div>
        <table class="analysis-table">${rowsHtml}</table>
        <div class="analysis-filepath">📁 ${d.filepath}</div>
    </div>`;
    chatBox.appendChild(card);
}

/** 将 SSE finding 事件数据渲染为单条文献发现条目，追加到任务卡片的 findings 容器中。 */
function _renderFinding(d, container) {
    let detailsHtml = '';
    for (const key in d.details) {
        if (key !== '_source_doc') {
            detailsHtml += `<span style="color:#374151"><b>${key}:</b> ${d.details[key]}</span><br>`;
        }
    }
    const div = document.createElement('div');
    div.className = 'finding-item';
    div.innerHTML = `<span class="finding-tag">🎯 新发现 (第${d.page}页)</span><br>
        <div style="font-size:0.85rem; padding-left:10px; border-left:2px solid #e5e7eb; margin-top:5px;">${detailsHtml}</div>`;
    container.appendChild(div);
}

/** 处理 SSE complete 事件：更新日志文字、关闭 SSE 连接、恢复 UI，并根据完成类型显示摘要弹窗或分析结果。 */
function _handleTaskComplete(msg, logEl, eventSource) {
    logEl.textContent = '✅ 任务结束！';
    logEl.style.color = '#047857';
    document.getElementById('pdf-viewer').classList.remove('is-scanning');
    document.getElementById('pdf-title').textContent = '任务已完成';

    eventSource.close();
    setTaskRunningState(false);

    const data = msg.data || {};
    if (data.agent_reply) {
        appendMessage(data.agent_reply, 'ai');
    } else if (data.error) {
        appendMessage(`❌ 任务失败：${data.error}`, 'ai');
    } else if (data.algorithm) {
        displayAnalysisResult(data);
    } else {
        showSummaryModal(data.csv || '', data.count || 0);
    }
}

/** 向 /api/cancel_task 发送 POST 请求，请求中断当前后台任务；后台完成清理后会自动推送 complete 事件关闭 SSE。 */
async function requestCancelTask() {
    sendBtn.innerHTML = '<div class="btn-spinner"></div> 正在中断...';
    sendBtn.disabled = true;
    await fetch('/api/cancel_task', { method: 'POST' });
}
