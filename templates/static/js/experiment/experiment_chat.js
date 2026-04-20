/**
 * experiment_chat.js — 实验设计 Agent 对话启动
 *
 * 流程：
 * 1. 用户输入实验描述 → 发送到 /api/experiment_chat
 * 2. 后端 LLM 生成 JSON → ExperimentManager 解析
 * 3. 前端接收 experiment_json → 更新实验设计面板和 JSON 显示
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
    console.log('[ExperimentChat] ========== 开始 ==========');
    console.log('[ExperimentChat] command:', command);

    const sessionId = 'exp_' + Date.now();
    window.currentExperimentSession = sessionId;

    console.log('[ExperimentChat] sessionId:', sessionId);

    setNormalLoadingState(true);
    try {
        console.log('[ExperimentChat] 发送 POST 请求到 /api/experiment_chat');
        console.log('[ExperimentChat] 请求体:', { session_id: sessionId, message: command });

        const res = await fetch('/api/experiment_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: command })
        });

        console.log('[ExperimentChat] HTTP 状态码:', res.status);
        console.log('[ExperimentChat] HTTP 状态文本:', res.statusText);

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        console.log('[ExperimentChat] ========== 响应数据 ==========');
        console.log('[ExperimentChat] 完整响应:', JSON.stringify(data, null, 2));
        console.log('[ExperimentChat] data.type:', data.type);
        console.log('[ExperimentChat] data.reply:', data.reply);
        console.log('[ExperimentChat] =====================================');

        setNormalLoadingState(false);

        // 处理不同类型的响应
        if (data.type === 'experiment_design') {
            console.log('[ExperimentChat] ✅ 类型: experiment_design');
            appendMessage(data.reply, 'ai');

            if (data.experiment_json) {
                console.log('[ExperimentChat] 收到实验设计 JSON');
                console.log('[ExperimentChat] 实验名称:', data.experiment_json.experiment_name);
                console.log('[ExperimentChat] 步骤数量:', data.experiment_json.steps?.length || 0);

                // 加载实验设计到画布和 JSON 编辑器
                loadExperimentFromJSON(data.experiment_json);

                // 显示成功通知
                const stepCount = data.experiment_json.steps?.length || 0;
                const expName = data.experiment_json.experiment_name || '未命名实验';
                showNotification(`✅ 实验设计已生成：${expName}（${stepCount} 个步骤）`, 'success');

                console.log('[ExperimentChat] 实验设计已加载到画布');
            } else {
                console.warn('[ExperimentChat] ⚠️ 响应中没有 experiment_json');
                showNotification('⚠️ 实验设计生成成功，但未返回 JSON 数据', 'warning');
            }
        } else if (data.type === 'error') {
            console.error('[ExperimentChat] ❌ 类型: error');
            appendMessage(data.reply, 'ai');
            showNotification('❌ 实验设计生成失败', 'error');
        } else if (data.type === 'task_trigger') {
            console.log('[ExperimentChat] 类型: task_trigger');
            appendMessage(data.reply, 'ai');
            startTaskStream();
        } else {
            console.error('[ExperimentChat] ❌ 未知响应类型:', data.type);
            console.error('[ExperimentChat] 完整数据:', data);
            appendMessage(data.reply || '未知响应类型', 'ai');
            showNotification('⚠️ 收到未知响应类型: ' + (data.type || 'undefined'), 'warning');
        }
    } catch (e) {
        console.error('[ExperimentChat] ========== 异常 ==========');
        console.error('[ExperimentChat] 错误类型:', e.name);
        console.error('[ExperimentChat] 错误消息:', e.message);
        console.error('[ExperimentChat] 错误堆栈:', e.stack);
        console.error('[ExperimentChat] =====================================');

        setNormalLoadingState(false);
        appendMessage('实验设计通信异常: ' + e.message, 'ai');
        showNotification('❌ 通信异常: ' + e.message, 'error');
    }
}
