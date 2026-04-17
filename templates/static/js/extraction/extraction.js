/**
 * extraction.js — 文献提取字段确认逻辑
 *
 * 当后端返回 field_confirm 类型时，渲染确认卡片让用户
 * 确认或修改提取字段，然后触发实际提取任务。
 *
 * 依赖：state.js, ui/input_state.js, ui/menu.js,
 *       hardware/task_stream.js（startTaskStream）
 */

/**
 * 渲染字段确认卡片（AI 推断出提取字段后调用）。
 * @param {string} taskDesc - 任务描述
 * @param {Array}  fields   - 推断出的字段列表
 * @param {string} replyMsg - AI 的说明文字
 */
function renderFieldConfirm(taskDesc, fields, replyMsg) {
    const html = `<div>
${replyMsg.replace(/\n/g, '<br>')}
<div class="agent-actions">
    <button class="btn-yes" onclick='confirmExtraction(${JSON.stringify(taskDesc)}, ${JSON.stringify(fields)}, this)'>✅ 确认使用上述字段提取</button>
    <button class="btn-no" onclick='modifyExtraction(this)'>❌ 补充修改要求</button>
</div>
</div>`;
    appendMessageHtml(html, 'ai');
}

/** 用户确认字段后，发送 start_extraction 请求并启动 SSE 任务流。 */
async function confirmExtraction(taskDesc, fields, btnElement) {
    btnElement.parentElement.innerHTML = '<i>(用户已确认)</i>';
    appendMessageHtml('✅ 是的，请按此计划开始执行。', 'user');

    setNormalLoadingState(true);
    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start_extraction', task_desc: taskDesc, fields })
    });
    setNormalLoadingState(false);

    const data = await res.json();
    if (data.type === 'task_trigger') {
        appendMessage(data.reply, 'ai');
        startTaskStream();
    }
}

/** 用户要求修改字段，切换回提取模式让用户补充说明。 */
function modifyExtraction(btnElement) {
    btnElement.parentElement.innerHTML = '<i>(用户要求修改)</i>';
    appendMessageHtml('❌ 否，我需要修改要求。', 'user');
    setMode('extract', '帮我搜寻：', '📄 文献提取');
    userInput.placeholder = '请直接输入你的补充要求，例如：增加某某字段...';
    userInput.focus();
}
