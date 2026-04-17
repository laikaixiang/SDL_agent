/**
 * chat.js — 聊天核心逻辑
 *
 * 处理用户消息发送、流式响应读取，以及根据后端返回的 type 字段
 * 分发到对应的确认渲染函数（硬件/提取/实验设计）。
 *
 * 依赖：state.js, ui/input_state.js, ui/menu.js,
 *       extraction/extraction.js, hardware/hardware.js,
 *       hardware/task_stream.js, experiment/experiment_chat.js
 */

// 绑定发送按钮点击和回车键
sendBtn.addEventListener('click', handleSendClick);
userInput.addEventListener('keypress', e => { if (e.key === 'Enter') handleSendClick(); });

/** 点击发送按钮时的入口：流式输出中则中断，否则发送消息。 */
function handleSendClick() {
    chatStreaming ? stopChatStream() : sendMessage();
}

/** 通过 AbortController 中断当前流式响应。 */
function stopChatStream() {
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
}

/** 读取输入框内容，拼接模式前缀后发送到 /api/chat，根据响应类型走 JSON 分发或流式读取。 */
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text && currentMode.id === 'normal') return;

    const finalPayload = currentMode.prefix + text;
    const displayHtml = currentMode.id !== 'normal'
        ? `<span style="background:rgba(255,255,255,0.2); padding:2px 6px; border-radius:6px; font-size:0.85em; margin-right:5px;">${currentMode.label}</span> ${text || '专门用于 FAPbI3 钙钛矿体系的钝化剂(Passivator)'}`
        : text;

    appendMessageHtml(displayHtml, 'user');
    userInput.value = '';
    setNormalLoadingState(true);

    // 数据分析模式由 analysis.js 单独处理，不走 /api/chat
    if (currentMode.id === 'analyze') {
        await handleAnalyzeMode(text);
        setNormalLoadingState(false);
        return;
    }

    abortController = new AbortController();
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'chat', message: finalPayload }),
            signal: abortController.signal
        });

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            // JSON 响应：解析后按 type 字段分发
            setNormalLoadingState(false);
            _dispatchJsonResponse(await response.json());
        } else {
            // 流式文本响应：逐块追加到消息气泡
            await _readStreamResponse(response);
            setNormalLoadingState(false);
        }
    } catch (error) {
        if (error.name !== 'AbortError') appendMessage('网络连接异常。', 'ai');
        setNormalLoadingState(false);
        setChatStreamingState(false);
    }
    abortController = null;
}

/** 根据后端返回的 type 字段，将响应分发到对应的 UI 渲染函数。 */
function _dispatchJsonResponse(data) {
    if (data.type === 'hardware_confirm') {
        // AI 解析出硬件工具调用，显示确认卡片
        renderHardwareConfirm(data.tool_calls, data.reply);
    } else if (data.type === 'field_confirm') {
        // AI 推断出提取字段，硬件控制走硬件确认，其余走字段确认
        data.task_desc === '硬件控制'
            ? renderHardwareConfirm(data.fields, data.reply)
            : renderFieldConfirm(data.task_desc, data.fields, data.reply);
    } else if (data.type === 'experiment_design_mode') {
        // 进入实验设计模式，启动 Agent 对话
        appendMessage(data.reply, 'ai');
        startExperimentChat(data.command);
    } else if (data.type === 'task_trigger') {
        // 触发后台长任务，打开 SSE 监听
        appendMessage(data.reply, 'ai');
        startTaskStream();
    } else {
        appendMessage(data.reply || '', 'ai');
    }
}

/** 读取流式响应，将每个文本块追加到消息气泡；用户中断时追加"已停止生成"提示。 */
async function _readStreamResponse(response) {
    setChatStreamingState(true);
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const msgDiv = createMessageDiv('ai');
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            msgDiv.textContent += decoder.decode(value, { stream: true });
            scrollToBottom();
        }
    } catch (readErr) {
        if (readErr.name !== 'AbortError') throw readErr;
        msgDiv.textContent += '\n(已停止生成)';
    }
    setChatStreamingState(false);
}
