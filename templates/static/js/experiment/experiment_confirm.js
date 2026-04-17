/**
 * experiment_confirm.js — 实验步骤确认逻辑
 *
 * 当 SSE 推送 experiment_confirm 事件时，渲染参数确认卡片，
 * 用户可确认、修改参数后确认，或跳过该步骤。
 *
 * 依赖：state.js, ui/input_state.js
 */

/**
 * 渲染实验步骤确认卡片（SSE experiment_confirm 事件触发）。
 * @param {Object} data - SSE 消息对象，含 tool/params/request_id/session_id
 */
function renderExperimentConfirm(data) {
    const { tool, params, request_id: requestId, session_id: sessionId } = data;
    let paramsHtml = '';

    if (tool === 'save_experiment_step') {
        paramsHtml = `
        <div class="param-list">
            <div class="param-row"><span>试剂:</span><input type="text" id="param-reagent-${requestId}" value="${params.reagent}" /></div>
            <div class="param-row"><span>体积 (μL):</span><input type="number" id="param-volume-${requestId}" value="${params.volume}" /></div>
            <div class="param-row"><span>转速 (rpm):</span><input type="number" id="param-spin_speed-${requestId}" value="${params.spin_speed}" /></div>
            <div class="param-row"><span>加速度 (rpm/s):</span><input type="number" id="param-spin_acc-${requestId}" value="${params.spin_acc}" /></div>
            <div class="param-row"><span>时长 (ms):</span><input type="number" id="param-spin_dur-${requestId}" value="${params.spin_dur}" /></div>
        </div>`;
    } else if (tool === 'start_experiment') {
        paramsHtml = `<div class="param-hint-text">AI 准备启动已注册的实验序列。请确认是否继续。</div>`;
    } else if (tool === 'read_pdf') {
        paramsHtml = `
        <div class="param-list">
            <div class="param-row"><span>文件路径:</span><span style="font-family:monospace;font-size:0.85rem;">${params.file_path}</span></div>
            <div class="param-row"><span>页码:</span><input type="number" id="param-page_number-${requestId}" value="${params.page_number || ''}" placeholder="留空读取全部" /></div>
        </div>`;
    }

    appendMessageHtml(`
    <div class="experiment-confirm-card">
        <div class="card-header">🧪 实验步骤待确认</div>
        <div class="card-tool-name">工具: ${tool}</div>
        ${paramsHtml}
        <div class="agent-actions">
            <button class="btn-yes" onclick="confirmExperiment('${requestId}', '${sessionId}', '${tool}', this)">✓ 确认</button>
            <button class="btn-edit" onclick="modifyExperiment('${requestId}', '${sessionId}', '${tool}', this)">✏️ 修改并确认</button>
            <button class="btn-no" onclick="skipExperiment('${requestId}', '${sessionId}', this)">✗ 跳过</button>
        </div>
    </div>`, 'ai');
}

/** 用户直接确认，不修改参数。 */
async function confirmExperiment(requestId, sessionId, tool, btnElement) {
    btnElement.parentElement.innerHTML = '<i>(用户已确认)</i>';
    appendMessageHtml('✅ 确认，请继续执行。', 'user');
    await _sendExperimentConfirm(requestId, sessionId, 'confirm', {});
}

/** 用户修改参数后确认，从表单读取最新值。 */
async function modifyExperiment(requestId, sessionId, tool, btnElement) {
    const params = {};
    if (tool === 'save_experiment_step') {
        params.reagent    = document.getElementById(`param-reagent-${requestId}`).value;
        params.volume     = parseInt(document.getElementById(`param-volume-${requestId}`).value);
        params.spin_speed = parseInt(document.getElementById(`param-spin_speed-${requestId}`).value);
        params.spin_acc   = parseInt(document.getElementById(`param-spin_acc-${requestId}`).value);
        params.spin_dur   = parseInt(document.getElementById(`param-spin_dur-${requestId}`).value);
    } else if (tool === 'read_pdf') {
        const pageNum = document.getElementById(`param-page_number-${requestId}`).value;
        params.page_number = pageNum ? parseInt(pageNum) : null;
    }

    btnElement.parentElement.innerHTML = '<i>(用户已修改并确认)</i>';
    appendMessageHtml('✏️ 已修改参数并确认。', 'user');
    await _sendExperimentConfirm(requestId, sessionId, 'confirm', params);
}

/** 用户跳过当前步骤。 */
async function skipExperiment(requestId, sessionId, btnElement) {
    btnElement.parentElement.innerHTML = '<i>(用户已跳过)</i>';
    appendMessageHtml('✗ 跳过此步骤。', 'user');
    await _sendExperimentConfirm(requestId, sessionId, 'skip', {});
}

/** 发送实验确认请求到后端。 */
async function _sendExperimentConfirm(requestId, sessionId, action, params) {
    await fetch('/api/experiment_confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: requestId, session_id: sessionId, action, params })
    });
}
