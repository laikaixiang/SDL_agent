/**
 * history.js — 对话历史持久化
 *
 * Monkey-patch appendMessage / appendMessageHtml，在消息渲染时自动记录到
 * messageHistory[]，每 5 条触发批量保存到后端 /api/history/save_batch。
 * 页面关闭时通过 sendBeacon 兜底保存。
 *
 * 加载顺序：必须在 input_state.js 之后，notification.js 之前。
 * 依赖：state.js（messageHistory, historyLastSavedIndex, currentMode）
 */

/* ── 保存原始函数引用 ── */
const _originalAppendMessage = appendMessage;
const _originalAppendMessageHtml = appendMessageHtml;

let _historySaveTimer = null;
const HISTORY_SAVE_INTERVAL = 500;   // debounce 毫秒

/* ── 工具函数 ── */

/** 从 HTML 字符串提取纯文本。 */
function _stripHtml(html) {
    const div = document.createElement('div');
    div.innerHTML = html;
    return div.textContent || div.innerText || '';
}

/** 从 currentMode.id 映射到历史记录中的 mode 值。 */
function _currentModeId() {
    // currentMode.id 可能是 'hardware_single' / 'hardware_design' 等
    if (currentMode.id === 'hardware_single') return 'hardware_single';
    if (currentMode.id === 'hardware_design') return 'experiment_design';
    return currentMode.id;
}

/* ── 消息记录 ── */

/**
 * 将一条消息 push 到 messageHistory，自动附加时间戳和模式上下文。
 * @param {'user'|'ai'} role
 * @param {string} content - 纯文本内容
 * @param {Object} [extra] - 额外字段 (prefix, response_type)
 */
function recordMessage(role, content, extra = {}) {
    const record = {
        role: role,
        content: content,
        timestamp: new Date().toISOString(),
        mode: _currentModeId()
    };
    if (extra.prefix)        record.prefix        = extra.prefix;
    if (extra.response_type) record.response_type = extra.response_type;
    messageHistory.push(record);
}

/* ── Monkey-patch ── */

/** 替换全局 appendMessage：先调用原始函数渲染，再记录到 messageHistory。 */
appendMessage = function(text, sender, responseType) {
    _originalAppendMessage(text, sender);
    const role = sender === 'user' ? 'user' : 'ai';
    const extra = {};
    if (role === 'user' && currentMode.prefix) extra.prefix = currentMode.prefix;
    if (role === 'ai' && responseType)          extra.response_type = responseType;
    recordMessage(role, text, extra);
    scheduleHistorySave();
};

/** 替换全局 appendMessageHtml：先调用原始函数渲染，再提取纯文本记录。 */
appendMessageHtml = function(html, sender, responseType) {
    _originalAppendMessageHtml(html, sender);
    const role = sender === 'user' ? 'user' : 'ai';
    const extra = {};
    if (role === 'user' && currentMode.prefix) extra.prefix = currentMode.prefix;
    if (role === 'ai' && responseType)          extra.response_type = responseType;
    recordMessage(role, _stripHtml(html), extra);
    scheduleHistorySave();
};

/* ── 流式消息记录 ── */

/**
 * 供 chat.js 在 _readStreamResponse 结束后调用，记录完整的流式响应文本。
 * 如果流式响应之前已经通过 _dispatchJsonResponse 记录了（type != streaming），
 * 则跳过，避免重复记录。
 * @param {string} text - 完整的流式响应文本
 * @param {'ai'} sender
 */
function recordStreamingMessage(text, sender) {
    // 检查最近一条 ai 消息是否已经通过 appendMessage 记录了（非流式分发路径）
    // 如果最后一条 ai 记录的内容和当前文本相同，跳过
    const role = sender === 'user' ? 'user' : 'ai';
    if (role === 'ai') {
        // 流式路径：_dispatchJsonResponse 中 type='task_trigger'/'experiment_design_mode'
        // 等已经通过 appendMessage(data.reply) 记录了。检查是否有未记录的流式响应。
        const lastRecord = messageHistory[messageHistory.length - 1];
        if (lastRecord && lastRecord.role === 'ai' && lastRecord.content === text) {
            return; // 已通过 appendMessage 记录
        }
    }
    recordMessage(role, text, { response_type: 'streaming' });
    scheduleHistorySave();
}

/* ── 逐条保存 ── */

/**
 * 每条消息记录后立刻异步保存到后端（不等待，fire-and-forget）。
 * 同时清空 debounce timer，避免之前的延迟保存重复触发。
 */
function scheduleHistorySave() {
    if (_historySaveTimer) clearTimeout(_historySaveTimer);
    _historySaveTimer = setTimeout(saveChatHistory, HISTORY_SAVE_INTERVAL);
}

/** 将 messageHistory 全量 POST 到 /api/history/save_batch。 */
async function saveChatHistory() {
    const unsaved = messageHistory.length - historyLastSavedIndex;
    if (unsaved === 0) return;
    historyLastSavedIndex = messageHistory.length;
    try {
        await fetch('/api/history/save_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: messageHistory })
        });
    } catch (e) {
        historyLastSavedIndex -= unsaved;
    }
}

/* ── 页面关闭兜底 ── */

window.addEventListener('beforeunload', () => {
    const unsaved = messageHistory.slice(historyLastSavedIndex);
    if (unsaved.length === 0) return;
    const payload = JSON.stringify({ messages: messageHistory });
    navigator.sendBeacon('/api/history/save_batch', payload);
});
