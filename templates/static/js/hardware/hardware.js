/**
 * hardware.js — 硬件操作确认逻辑
 *
 * 当 AI 解析出硬件工具调用后，渲染确认卡片让用户
 * 确认/取消/补充，然后调用 /api/chat start_hardware 执行。
 *
 * 依赖：state.js, ui/input_state.js, ui/menu.js
 */

/**
 * 渲染硬件操作确认卡片。
 * @param {Array}  toolCalls - 工具调用列表
 * @param {string} replyMsg  - AI 的说明文字
 */
function renderHardwareConfirm(toolCalls, replyMsg) {
    const html = `<div>
${replyMsg.replace(/\n/g, '<br>')}
<div class="agent-actions">
    <button class="btn-yes" onclick='confirmHardware(${JSON.stringify(toolCalls)}, this)'>✅ 确认执行</button>
    <button class="btn-no" onclick='cancelHardware(this)'>❌ 取消</button>
    <button class="btn-no" onclick='supplementHardware(this)'>💬 补充</button>
</div>
</div>`;
    appendMessageHtml(html, 'ai');
}

/** 用户确认后发送硬件执行请求，执行期间锁定 UI。 */
async function confirmHardware(toolCalls, btnElement) {
    if (hardwareRunning || btnElement.disabled) return;
    btnElement.disabled = true;
    btnElement.parentElement.innerHTML = '<i>(用户已确认)</i>';
    appendMessageHtml('✅ 确认，请执行硬件操作。', 'user');

    setHardwareRunningState(true);
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start_hardware', tool_calls: toolCalls })
        });
        const data = await res.json();
        if (data.status === 'success') {
            appendMessage('✅ 硬件操作执行完成。\n' + (data.reply || ''), 'ai');
        } else {
            appendMessage('❌ 硬件操作执行失败: ' + (data.reply || data.error || '未知错误'), 'ai');
        }
    } catch (e) {
        appendMessage('❌ 硬件通信异常: ' + e.message, 'ai');
    }
    setHardwareRunningState(false);
}

/** 用户取消硬件操作。 */
function cancelHardware(btnElement) {
    btnElement.parentElement.innerHTML = '<i>(用户已取消)</i>';
    appendMessageHtml('❌ 已取消硬件操作。', 'user');
}

/** 用户要求补充说明，切换到硬件控制模式。 */
function supplementHardware(btnElement) {
    btnElement.parentElement.innerHTML = '<i>(用户要求补充)</i>';
    appendMessageHtml('💬 我需要补充说明。', 'user');
    setMode('hardware_single', '硬件控制：', '🔧 单步控制');
    userInput.placeholder = '请输入补充说明，例如：修改转速为4000rpm...';
    userInput.focus();
}
