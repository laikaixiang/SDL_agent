/**
 * state.js — 全局状态与 DOM 引用
 *
 * 所有模块共享的变量在此统一声明，避免各文件重复查询 DOM。
 * 加载顺序：必须是第一个 <script>，其他模块依赖此文件。
 */

/* ── DOM 引用（页面加载后立即可用） ── */
const chatBox        = document.getElementById('chat-box');
const userInput      = document.getElementById('user-input');
const sendBtn        = document.getElementById('send-btn');
const toolBtn        = document.getElementById('tool-btn');
const modeMenu       = document.getElementById('mode-menu');
const modeBadge      = document.getElementById('mode-badge');
const inputContainer = document.getElementById('input-container');

/* ── 聊天状态 ── */
let currentMode      = { id: 'normal', prefix: '', label: '' };
let abortController  = null;   // 用于中断流式响应
let chatStreaming     = false;  // 是否正在流式输出
let hardwareRunning  = false;  // 硬件执行中（不可中断）

/* ── 算法交互状态 ── */
let selectedAlgorithm = null;
let selectedFilePath  = null;
let algorithmParams   = {};

/* ── 单步控制面板状态 ── */
let stepPanelTools    = [];
let stepRunning       = false;
let stepExpandedTool  = null;
let stepPanelCollapsed = false;

/* ── 实验设计面板状态 ── */
let experimentSteps   = [];
let experimentName    = '未命名实验';
let draggedStepIndex  = null;
let expCodeViewMode   = 'json';
let expCodeJSON       = '';
let expPythonCode     = '';

/* ── 面板 z-index 管理 ── */
let panelZIndexCounter = 100;       // 每次打开面板递增，确保新面板在最上层
const activePanels     = new Set(); // 当前已打开的面板 ID 集合

/**
 * 从工具参数表单中收集并类型转换参数值。
 * step_panel.js 和 experiment_design.js 共用此逻辑。
 * @param {Object} tool - 工具对象（含 name 和 params 定义）
 * @returns {Object} 收集到的参数键值对
 */
function collectToolParams(tool) {
    const params = {};
    for (const [pName, pInfo] of Object.entries(tool.params)) {
        const inputEl = document.getElementById(`param-${tool.name}-${pName}`);
        let val = inputEl ? inputEl.value.trim() : '';
        if (val === '' && pInfo.default !== null && pInfo.default !== undefined) val = String(pInfo.default);
        if (pInfo.type === 'int')        params[pName] = parseInt(val) || 0;
        else if (pInfo.type === 'float') params[pName] = parseFloat(val) || 0.0;
        else params[pName] = val;
    }
    return params;
}
