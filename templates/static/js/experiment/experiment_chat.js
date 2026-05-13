/**
 * experiment_chat.js — 实验设计 Agent 对话启动（SSE 流式）
 *
 * 用户选择"实验设计"模式后，通过 SSE 流式接收 LLM 实时输出，
 * 完成后解析 JSON 并推送到实验设计画布。
 *
 * 依赖：state.js, ui/input_state.js, notification.js,
 *       experiment/experiment_design.js（loadExperimentFromJSON）
 */

/**
 * 启动实验设计 Agent 对话（SSE 流式）。
 * @param {string} command - 用户的实验描述指令
 */
async function startExperimentChat(command) {
    const sessionId = 'exp_' + Date.now();
    window.currentExperimentSession = sessionId;

    // 创建流式消息气泡
    const msgDiv = createMessageDiv('ai');
    msgDiv.textContent = '⏳ AI 正在分析实验需求...';
    scrollToBottom();

    const controller = new AbortController();
    const timeoutId = setTimeout(function () {
        controller.abort();
    }, 240000);

    try {
        const res = await fetch('/api/experiment_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: command, stream: true }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!res.ok) {
            var text = await res.text().catch(function () { return ''; });
            throw new Error(text || 'HTTP ' + res.status);
        }

        // 读取 SSE 流
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';
        var streamedText = '';

        while (true) {
            var _a = await reader.read(), done = _a.done, value = _a.value;
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop() || '';

            var dataLine = '';
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i];
                if (line.indexOf('data: ') === 0) {
                    dataLine = line.slice(6);
                } else if (line === '' && dataLine) {
                    try {
                        var msg = JSON.parse(dataLine);
                        switch (msg.type) {
                            case 'chunk':
                                streamedText += msg.data;
                                // 每收集50个字符更新一次，显示进度而非原始JSON
                                if (streamedText.length % 50 < msg.data.length || streamedText.length < 50) {
                                    msgDiv.textContent = '⏳ AI 正在生成实验方案...\n\n```json\n' + streamedText + '\n```';
                                }
                                scrollToBottom();
                                break;
                            case 'complete':
                                msgDiv.textContent = msg.data.reply;
                                recordStreamingMessage(msg.data.reply, 'ai');
                                if (msg.data.experiment_json) {
                                    loadExperimentFromJSON(msg.data.experiment_json);
                                    showNotification('✅ 实验设计已生成并推送到画布', 'success');
                                }
                                return;
                            case 'error':
                                msgDiv.textContent = '❌ ' + msg.data;
                                recordStreamingMessage('❌ ' + msg.data, 'ai');
                                return;
                        }
                    } catch (e) {
                        // JSON 解析错误，跳过
                    }
                    dataLine = '';
                }
            }
        }
        // 流意外结束
        msgDiv.textContent = '⚠️ 流式响应意外结束';
        recordStreamingMessage('⚠️ 流式响应意外结束', 'ai');
    } catch (e) {
        clearTimeout(timeoutId);
        if (e.name === 'AbortError') {
            msgDiv.textContent = '⚠️ 实验设计生成超时（>4分钟），请重试或简化需求描述';
            recordStreamingMessage(msgDiv.textContent, 'ai');
        } else {
            msgDiv.textContent = '实验设计通信异常: ' + e.message;
            recordStreamingMessage(msgDiv.textContent, 'ai');
        }
    }
}
