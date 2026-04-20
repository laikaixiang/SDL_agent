/**
 * experiment_chat.js — 实验设计 Agent 对话启动
 *
 * 用户选择"实验设计"模式后，发送命令到 /api/experiment_chat，
 * 后端返回 JSON 实验方案后推送到实验设计画布。
 *
 * 依赖：state.js, ui/input_state.js, notification.js,
 *       hardware/task_stream.js（startTaskStream）,
 *       experiment/experiment_design.js（loadExperimentFromJSON）
 */

/**
 * 启动实验设计 Agent 对话。
 * @param {string} command - 用户的实验描述指令
 */
async function startExperimentChat(command) {
    const sessionId = 'exp_' + Date.now();
    window.currentExperimentSession = sessionId;

    setNormalLoadingState(true);
    try {
        // 设置30秒超时，因为LLM生成实验设计需要10-15秒
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const res = await fetch('/api/experiment_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: command }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        const data = await res.json();
        setNormalLoadingState(false);

        if (data.type === 'task_trigger') {
            appendMessage(data.reply, 'ai');
            startTaskStream();
        } else if (data.type === 'experiment_design') {
            appendMessage(data.reply, 'ai');
            if (data.experiment_json) {
                loadExperimentFromJSON(data.experiment_json);
                showNotification('✅ 实验设计已生成并推送到画布', 'success');
            }
        } else if (data.type === 'error') {
            appendMessage(data.reply, 'ai');
        }
    } catch (e) {
        setNormalLoadingState(false);
        if (e.name === 'AbortError') {
            appendMessage('⚠️ 实验设计生成超时（>30秒），请重试或简化需求描述', 'ai');
        } else {
            appendMessage('实验设计通信异常: ' + e.message, 'ai');
        }
    }
}
