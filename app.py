"""
Flask应用入口 - 简洁的Web服务入口

职责：
- Flask应用初始化和配置
- 路由定义和请求处理
- 响应格式化和错误处理
- 核心业务逻辑通过core模块调用
"""

import sys
# Windows: prevent UnicodeEncodeError when print() outputs Chinese characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from flask import Flask, request, jsonify, Response, send_from_directory
import threading
import os
import json
import queue
import uuid
import asyncio
import atexit
import signal
import webbrowser
import re
import requests
from threading import Timer
from datetime import datetime

# 导入核心模块
from core import (
    Config,
    LLMClient,
    FieldInference,
    AlgorithmParser,
    HardwareController,
    TaskManager,
    # ExperimentDesignAgent,  # Deprecated PydanticAI version, now using Approach 2 in field_inference.py
    SoftwareManager,
    AdaptiveStreamHandler,
)
from utils.sse import sse_response
from utils.i18n import i18n
from utils.stream_adapter import StreamAdapter
from core.extract_manager import PDFProcessor, ExtractionEngine, AlgorithmGuide
from extract.embedding_service import create_embedding_service
from extract.vector_store import ChromaVectorStore
from extract.semantic_search import SemanticSearch
from extract.literature_indexer import LiteratureIndexer
from utils import CSVWriter
from prompts.api import prompts_bp
from core.agent_tools import create_main_executor, AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentLoop, AgentOrchestrator
import queue as queue_module

# 初始化Flask应用，static 文件夹指向 Vue 前端构建产物
app = Flask(__name__, static_folder='frontend/dist', static_url_path='/static')
app.secret_key = os.urandom(24)  # 用于session管理
app.register_blueprint(prompts_bp)

# 初始化核心组件
config = Config()

# 初始化 PromptManager（全局单例，各模块通过 create_prompt_manager() 获取）
from prompts import create_prompt_manager as _init_prompt_manager
_init_prompt_manager()
_talk_extra = config.get_extra_body("TALK")
llm_client = LLMClient(api_key=config.TALK_API_KEY, api_url=config.TALK_API_URL, extra_body=_talk_extra)
pdf_processor = PDFProcessor()
field_inference = FieldInference()
algorithm_parser = AlgorithmParser(llm_client)    # 算法解析器
hardware_controller = HardwareController()
task_manager = TaskManager()

# Agent engine（Phase 1 — 启动时初始化）
_agent_executor = None
_agent_orchestrator = None
_agent_ask_queues: dict = {}  # session_id -> queue.Queue
if config.AGENT_ENABLED:
    try:
        _agent_executor = create_main_executor()
        _agent_orchestrator = AgentOrchestrator(executor=_agent_executor)
        print(f"[Agent]   Tools: {len(_agent_executor.names)} registered")
        print(f"[Agent]   Templates: {_agent_orchestrator.list_templates()}")
    except Exception as e:
        print(f"[Agent] 初始化失败: {e}，Agent 功能不可用")
        _agent_executor = None
        _agent_orchestrator = None

# Phase 3: 语义搜索基础设施（提前初始化，供 ExtractionEngine 复用）
_semantic_search_instance = None
try:
    _embedding_service = create_embedding_service()
    _vector_store = ChromaVectorStore(persist_dir=config.CHROMADB_PERSIST_DIR)
    _sqlite_path = os.path.join(config.CHROMADB_PERSIST_DIR, "page_metadata.db")
    _semantic_search_instance = SemanticSearch(_embedding_service, _vector_store, _sqlite_path)
    print(f"[语义搜索] 初始化成功，已索引 {_vector_store.count()} 个页面向量")
except Exception as e:
    print(f"[语义搜索] 初始化失败: {e}，搜索功能不可用")

# 初始化 ExtractionEngine，注入已创建的 embedding/vector_store
extraction_engine = ExtractionEngine(task_manager)
if _semantic_search_instance is not None:
    extraction_engine.embedding_service = _embedding_service
    extraction_engine.vector_store = _vector_store

csv_writer = CSVWriter()
# experiment_agent = ExperimentDesignAgent()  # Deprecated PydanticAI version, now using Approach 2 in field_inference.py
software_manager = SoftwareManager()        # 软件算法管理器
adaptive_handler = AdaptiveStreamHandler(config, llm_client)  # 自适应流式响应处理器
literature_indexer = LiteratureIndexer()     # 文献库索引器（注册表查询）

# =============================================================================
# 会话管理系统
# =============================================================================

# 全局会话时间戳（应用启动时创建）
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 全局 temporal 目录（所有会话共享，不在 history 下）
GLOBAL_TEMPORAL_DIR = os.path.join(os.path.dirname(config.DIALOGUE_DATA_DIR), "temporal")
os.makedirs(GLOBAL_TEMPORAL_DIR, exist_ok=True)

# 创建会话专属文件夹（位于 history/<timestamp>/）
SESSION_BASE_PATH = os.path.join(config.DIALOGUE_DATA_DIR, SESSION_TIMESTAMP)
os.makedirs(os.path.join(SESSION_BASE_PATH, "extract"), exist_ok=True)
os.makedirs(os.path.join(SESSION_BASE_PATH, "results"), exist_ok=True)
os.makedirs(os.path.join(SESSION_BASE_PATH, "experiment_designs"), exist_ok=True)

print(f"[会话管理] 应用启动，会话时间戳: {SESSION_TIMESTAMP}")
print(f"[会话管理] 数据保存路径: {SESSION_BASE_PATH}")
print(f"[会话管理] 全局 temporal: {GLOBAL_TEMPORAL_DIR}")

# 初始化引导式算法生成（依赖 SESSION_BASE_PATH 做持久化）
algorithm_guide = AlgorithmGuide(session_path=SESSION_BASE_PATH)

# =============================================================================
# 平台启动检测：流式能力 + 统一诊断
# =============================================================================
from platform_init.check_stream_capability import ensure_capability_check

_stream_capabilities = {"default": {"streaming_enabled": False}}
try:
    _stream_capabilities = ensure_capability_check(config)
except Exception as e:
    print(f"[platform_init] 流式检测失败（不影响正常功能）: {e}")

# 统一平台诊断（仅本地检查，< 1.5s）
from platform_init.platform_init import run_checks as _run_platform_checks

_platform_status = {"summary": {"total": 0, "pass": 0, "fail": 0, "warn": 0, "skip": 0}}
try:
    _platform_status = _run_platform_checks(scope="local")
    s = _platform_status["summary"]
    print(f"[platform_init] 平台检查完成: {s['pass']} pass, {s['fail']} fail, "
          f"{s['warn']} warn, {s['skip']} skip (共{s['total']}项)")
except Exception as e:
    print(f"[platform_init] 平台检查失败（不影响正常功能）: {e}")

def _update_session_index(session_info):
    """更新 sessions_index.json，upsert 当前会话条目。"""
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"sessions": []}

    ts = session_info["timestamp"]
    for entry in data["sessions"]:
        if entry["timestamp"] == ts:
            entry.update(session_info)
            break
    else:
        data["sessions"].append(session_info)

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

FOLDERS_PATH = os.path.join(config.DIALOGUE_DATA_DIR, "folders.json")

def _read_folders():
    """读取 folders.json，不存在则返回空列表。"""
    if os.path.exists(FOLDERS_PATH):
        with open(FOLDERS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f).get("folders", [])
    return []

def _write_folders(folders: list):
    """写入 folders.json。"""
    with open(FOLDERS_PATH, 'w', encoding='utf-8') as f:
        json.dump({"folders": folders}, f, ensure_ascii=False, indent=2)

def _generate_title(messages, lang: str = 'zh'):
    """取前2条用户消息调用 LLM 生成会话标题。"""
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    if len(user_msgs) < 2:
        return None
    lines = "\n".join(f"{i+1}. {user_msgs[i]}" for i in range(min(3, len(user_msgs))))
    from prompts import create_prompt_manager
    pm = create_prompt_manager(lang=lang)
    prompt = pm.get("misc_session_title", lines=lines)
    try:
        result = adaptive_handler.generate_non_streaming_response(
            prompt, model=config.MODEL_NAME_TALK
        )
        result = result.strip().strip('"''「」『』\n')
        if len(result) > 50:
            result = result[:50]
        return result if result else None
    except Exception as e:
        print(f"[历史] 标题生成失败: {e}")
        return None

def _scan_session_outputs():
    """扫描当前会话子目录和全局 temporal，返回 outputs 字典。"""
    outputs = {}
    for subdir in ["extract", "results", "experiment_designs"]:
        dir_path = os.path.join(SESSION_BASE_PATH, subdir)
        if os.path.isdir(dir_path):
            files = sorted(os.listdir(dir_path))
            outputs[subdir] = files
        else:
            outputs[subdir] = []
    # temporal 是全局共享目录，不在会话文件夹下
    if os.path.isdir(GLOBAL_TEMPORAL_DIR):
        outputs["temporal"] = sorted(os.listdir(GLOBAL_TEMPORAL_DIR))
    else:
        outputs["temporal"] = []
    return outputs

def _on_shutdown():
    """服务关闭时更新 sessions_index.json 的 saved_at。"""
    try:
        _update_session_index({
            "timestamp": SESSION_TIMESTAMP,
            "started_at": SESSION_TIMESTAMP,
            "saved_at": datetime.now().isoformat(),
            "path": SESSION_TIMESTAMP
        })
    except Exception:
        pass  # 静默失败，避免阻塞关闭

atexit.register(_on_shutdown)
signal.signal(signal.SIGINT, lambda s, f: (_on_shutdown(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (_on_shutdown(), sys.exit(0)))

# 启动时注册当前会话到索引
_update_session_index({
    "timestamp": SESSION_TIMESTAMP,
    "started_at": datetime.now().isoformat(),
    "saved_at": datetime.now().isoformat(),
    "message_count": 0,
    "title": None,
    "path": SESSION_TIMESTAMP
})

def get_session_path(subdir=""):
    """
    获取当前会话的数据路径

    Args:
        subdir: 子目录名称，如 "extract", "temporal", "results", "experiment_designs"
                "temporal" 返回全局共享的 dialogue data/temporal/ 路径

    Returns:
        str: 完整路径
    """
    if subdir == "temporal":
        return GLOBAL_TEMPORAL_DIR
    if subdir:
        return os.path.join(SESSION_BASE_PATH, subdir)
    return SESSION_BASE_PATH

def _resolve_pdf_path(doc_name: str) -> str:
    """Resolve a PDF filename to its full path.

    Searches in: dialogue data/PDF_TARGET/, then project root.
    """
    import glob
    # Try PDF_FOLDER directory
    pdf_dir = config.PDF_FOLDER if hasattr(config, 'PDF_FOLDER') else os.path.join(config.DIALOGUE_DATA_DIR, '..', 'PDF_TARGET')
    pdf_dir = os.path.normpath(pdf_dir)
    if os.path.isdir(pdf_dir):
        for ext in ['.pdf', '.PDF']:
            candidate = os.path.join(pdf_dir, doc_name if doc_name.endswith('.pdf') else doc_name + ext)
            if os.path.isfile(candidate):
                return candidate
    # Try direct path
    if os.path.isfile(doc_name):
        return doc_name
    # Try with .pdf extension
    if not doc_name.endswith('.pdf') and os.path.isfile(doc_name + '.pdf'):
        return doc_name + '.pdf'
    return ""


# 重新初始化需要会话路径的组件
extraction_engine = ExtractionEngine(task_manager, session_path=SESSION_BASE_PATH, temporal_dir=GLOBAL_TEMPORAL_DIR)
csv_writer = CSVWriter(session_path=SESSION_BASE_PATH, temporal_dir=GLOBAL_TEMPORAL_DIR)
software_manager = SoftwareManager(
    temporal_dir=GLOBAL_TEMPORAL_DIR,
    results_dir=get_session_path("results")
)


def open_browser():
    """打开浏览器 — 默认打开 Vue 前端"""
    webbrowser.open("http://127.0.0.1:5000/")


# =============================================================================
# Agent engine helpers (Phase 1)
# =============================================================================

def _spawn_agent_impl(template: str, task: str, context: dict = None, mode: str = "single", siblings: list = None) -> dict:
    """
    Implementation of the spawn_agent tool.
    Called by AgentLoop when the LLM decides to spawn a sub-agent.

    Returns:
        {"result": "summary string"}
    """
    if _agent_orchestrator is None:
        return {"result": "错误: Agent 引擎未启用"}

    if mode == "parallel" and siblings:
        tasks = [s.get("task", "") for s in siblings]
        results = _agent_orchestrator.spawn_parallel(template, tasks)
        summary_parts = []
        for i, r in enumerate(results):
            if r and not r.get("error"):
                fm = r.get("final_message", {})
                content = fm.get("content", "") if fm else str(r)
                summary_parts.append(f"[{template}_{i}]: {str(content)[:200]}")
        return {"result": "并行执行完成:\n" + "\n".join(summary_parts) if summary_parts else "并行执行完成"}

    # mode == "single"
    result = _agent_orchestrator.spawn(template, task, context)
    if result.get("error"):
        return {"result": f"子 Agent 执行失败: {result['error']}"}
    fm = result.get("final_message", {})
    content = fm.get("content", "") if fm else str(result)
    return {"result": str(content)}


def _make_session_executor() -> UnifiedToolExecutor:
    """Create a per-session executor with spawn_agent added."""
    if _agent_executor is None:
        return UnifiedToolExecutor([])

    # Get available templates
    templates_list = _agent_orchestrator.list_templates() if _agent_orchestrator else []

    # Build spawn_agent tool
    spawn_tool = AgentTool(
        name="spawn_agent",
        description=(
            "创建一个子 Agent 来执行专门任务。子 Agent 有自己的工具集，独立于主对话历史运行，完成后返回结果摘要。"
            "用于需要大量上下文或长时间运行的任务（如文献检索、数据提取、实验设计）。"
            "mode='single' 创建单个子 Agent，'parallel' 并行创建多个。"
            f"可用模板: {templates_list}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": f"Agent 模板名。可用模板: {templates_list}",
                    "enum": templates_list if templates_list else ["literature_searcher"]
                },
                "task": {
                    "type": "string",
                    "description": "子 Agent 的任务描述"
                },
                "context": {
                    "type": "object",
                    "description": "可选的上下文数据字典"
                },
                "mode": {
                    "type": "string",
                    "description": "执行模式: single=单个子Agent, parallel=并行多个",
                    "enum": ["single", "parallel"]
                },
                "siblings": {
                    "type": "array",
                    "description": "并行模式下额外的任务列表 [{'task': '...'}, ...]"
                },
            },
            "required": ["template", "task"],
        },
        required=["template", "task"],
        func=lambda args: _spawn_agent_impl(**args),
        category="builtin",
    )

    # Merge with main executor's tools
    all_tools = list(_agent_executor._tools.values()) + [spawn_tool]
    return UnifiedToolExecutor(all_tools)


@app.route('/')
def home():
    """主页路由 — 返回 Vue SPA"""
    return send_from_directory('frontend/dist', 'index.html')


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Vite 构建产物 — JS/CSS/字体等"""
    return send_from_directory('frontend/dist/assets', filename)


@app.route('/extraction_mode')
def extraction_mode_page():
    """提取模式设置页面 — 返回 Vue SPA"""
    return send_from_directory('frontend/dist', 'index.html')


# ── 旧路由兼容（/v2、/v2/*、/v2-static/* 重定向到新入口） ──
@app.route('/v2')
@app.route('/v2/')
@app.route('/v2/<path:path>')
def serve_v2_redirect(path: str = None):
    """旧 /v2/* 路由 — 返回 Vue SPA"""
    return send_from_directory('frontend/dist', 'index.html')


@app.route('/v2-static/<path:filename>')
def serve_v2_static(filename: str):
    """旧静态资源路由 — 兼容旧版 URL 路径"""
    return send_from_directory('frontend/dist', filename)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    文件上传路由
    处理PDF上传（到PDF文件夹）和CSV上传（到全局temporal目录）
    支持 form key: 'files' 或 'file'
    """
    files = request.files.getlist('files') or request.files.getlist('file')
    if not files:
        lang = i18n.get_lang(request)
        return jsonify({'error': i18n.get('errors.noFileReceived', lang)}), 400

    saved_files = []

    for file in files:
        if not file.filename:
            continue
        fname_lower = file.filename.lower()
        if fname_lower.endswith('.pdf'):
            os.makedirs(config.PDF_FOLDER, exist_ok=True)
            path = os.path.join(config.PDF_FOLDER, file.filename)
        elif fname_lower.endswith('.csv') or fname_lower.endswith('.xlsx') or fname_lower.endswith('.xls'):
            os.makedirs(GLOBAL_TEMPORAL_DIR, exist_ok=True)
            path = os.path.join(GLOBAL_TEMPORAL_DIR, file.filename)
        else:
            continue
        file.save(path)
        saved_files.append({'name': file.filename, 'path': path.replace('\\', '/')})

    if saved_files:
        return jsonify({
            'status': 'success',
            'saved': [s['name'] for s in saved_files],
            'filename': saved_files[0]['path'],
            'files': saved_files,
        })
    return jsonify({'status': 'success', 'saved': []})


@app.route('/api/task_stream')
def task_stream():
    """
    任务流路由
    提供Server-Sent Events接口，实时推送任务进度
    """
    def event_stream():
        while True:
            try:
                msg = task_manager.get_task_message(timeout=2)
                if msg:
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                    if msg.get("type") == "complete":
                        break
            except queue.Empty:
                if not task_manager.task_running:
                    break
                yield ": heartbeat\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/api/cancel_task', methods=['POST'])
def cancel_task():
    """
    取消任务路由
    中断当前正在执行的提取任务（硬件执行中不可中断）
    """
    if hardware_controller.is_hardware_running():
        lang = i18n.get_lang(request)
        return jsonify({"status": "rejected", "reason": i18n.get('errors.hardwareRunningCannotInterrupt', lang)})
    task_manager.cancel_task()
    return jsonify({"status": "stopping"})


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    聊天和命令处理路由
    处理用户输入，支持聊天、提取任务、硬件控制等多种功能
    """
    data = request.json
    user_message = data.get('message', '').strip()
    action = data.get('action', 'chat')  # 用于区分普通对话还是特殊指令
    history = data.get('history', [])    # 前端传来的对话历史

    # 特殊流程：用户已确认数据分析参数，正式开始分析
    if action == 'start_data_analysis':
        return handle_data_analysis_execute(data)

    # 特殊流程：用户已确认字段，正式开始提取
    if action == 'start_extraction':
        return handle_extraction_start(data)

    # 特殊流程：用户已确认硬件操作，正式执行
    if action == 'start_hardware':
        return handle_hardware_execute(data)

    # 拦截提取指令：Agentic 判断与 Schema 生成
    if user_message.startswith("帮我搜寻："):
        return handle_extraction_request(user_message, history)

    # 硬件控制（根据前端选择的模式直接分发）
    if user_message.startswith("硬件控制：") or user_message.startswith("实验设计："):
        return handle_hardware_request(user_message)

    # 数据分析（智能交互模式）
    if user_message.startswith("数据分析"):
        return handle_data_analysis(user_message)

    # 算法生成
    if user_message.startswith("生成算法："):
        return handle_generate_algorithm(user_message)

    # 普通聊天流式输出
    return handle_normal_chat(user_message, history)


def handle_extraction_start(data: dict) -> Response:
    """
    处理提取任务开始

    Args:
        data: 请求数据

    Returns:
        JSON响应
    """
    task_desc = data.get('task_desc')
    fields = data.get('fields')
    lang = i18n.get_lang(request)

    # 清空任务队列
    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    # 生成任务ID
    task_id = task_manager.generate_task_id()

    # 启动提取任务线程
    threading.Thread(
        target=extraction_engine.process_pdf_library,
        args=(task_id, task_desc, fields)
    ).start()

    return jsonify({
        'type': 'task_trigger',
        'reply': i18n.get('status.taskDispatched', lang)
    })


def handle_extraction_request(user_message: str, history: list = None) -> Response:
    """
    处理提取请求

    Args:
        user_message: 用户消息
        history: 对话历史 [{role: str, content: str}, ...]

    Returns:
        JSON响应
    """
    lang = i18n.get_lang(request)

    # 检查是否有任务正在运行
    if task_manager.task_running:
        return jsonify({
            'type': 'system',
            'reply': i18n.get('status.extractionTaskRunning', lang)
        })

    # 提取任务描述
    task_desc = user_message.replace("帮我搜寻：", "").strip()

    # 场景1：如果没有输入，使用默认值
    if not task_desc:
        task_desc = "专门用于 FAPbI3 钙钛矿体系的钝化剂(Passivator)"
        default_fields = ["钝化剂名称", "原文原句", "作用机理", "文献来源"]

        # 清空任务队列
        while not task_manager.is_queue_empty():
            task_manager.get_task_message()

        # 生成任务ID
        task_id = task_manager.generate_task_id()

        # 启动提取任务线程
        threading.Thread(
            target=extraction_engine.process_pdf_library,
            args=(task_id, task_desc, default_fields)
        ).start()

        return jsonify({
            'type': 'task_trigger',
            'reply': i18n.get('status.extractionStarted', lang)
        })

    # 场景2：自定义输入，去LLM询问字段
    else:
        success, fields = extraction_engine.infer_fields(task_desc, history)

        if not success:
            # 若失败，返回错误
            return jsonify({
                'type': 'system',
                'reply': i18n.get('errors.fieldInferenceFailed', lang).format(error=fields)
            })

        confirm_msg = i18n.get('status.fieldConfirmPrompt', lang).format(task=task_desc, fields=', '.join(fields))

        return jsonify({
            'type': 'field_confirm',
            'task_desc': task_desc,
            'fields': fields,
            'reply': confirm_msg
        })


def handle_hardware_request(user_message: str) -> Response:
    """
    处理硬件请求：根据前端选择的模式直接分发

    Args:
        user_message: 用户消息（带前缀："硬件控制：" 或 "实验设计："）

    Returns:
        JSON响应
    """
    lang = i18n.get_lang(request)

    # 判断模式
    if user_message.startswith("实验设计："):
        mode = "design"
        cmd_text = user_message.replace("实验设计：", "").strip()
    else:
        mode = "single"
        cmd_text = user_message.replace("硬件控制：", "").strip()

    if not cmd_text:
        return jsonify({
            'type': 'system',
            'reply': i18n.get('info.hardwarePromptHint', lang)
        })

    # 实验设计模式
    if mode == "design":
        return jsonify({
            'type': 'experiment_design_mode',
            'command': cmd_text,
            'reply': i18n.get('status.experimentDesignMode', lang).format(command=cmd_text)
        })

    # 单步控制模式
    else:
        success, tool_calls = hardware_controller.agent.parse_complex_command(cmd_text)

        if not success or not tool_calls:
            return jsonify({
                'type': 'system',
                'reply': i18n.get('errors.hardwareCommandParseFailed', lang).format(command=cmd_text)
            })

        # 生成确认信息
        confirmation_msg = hardware_controller.ask_for_experiment_confirmation(tool_calls)
        confirmation_msg = i18n.get('status.singleStepControlMode', lang).format(confirmation=confirmation_msg)

        return jsonify({
            'type': 'hardware_confirm',
            'task_desc': "硬件控制",
            'tool_calls': tool_calls,
            'reply': confirmation_msg
        })


def handle_hardware_control(user_message: str) -> Response:
    """
    处理硬件控制

    Args:
        user_message: 用户消息

    Returns:
        JSON响应
    """
    cmd_text = user_message.replace("硬件控制：", "").strip()

    # 解析硬件命令
    success, tool_calls = hardware_controller.agent.parse_complex_command(cmd_text)

    if not success or not tool_calls:
        lang = i18n.get_lang(request)
        return jsonify({
            'type': 'system',
            'reply': i18n.get('errors.hardwareCommandParseFailedShort', lang)
        })

    # 生成确认信息
    confirmation_msg = hardware_controller.ask_for_experiment_confirmation(tool_calls)

    return jsonify({
        'type': 'field_confirm',
        'task_desc': "硬件控制",
        'fields': tool_calls,
        'reply': confirmation_msg
    })


def handle_hardware_execute(data: dict) -> Response:
    """
    执行已确认的硬件操作

    Args:
        data: 请求数据，包含 tool_calls

    Returns:
        JSON响应
    """
    lang = i18n.get_lang(request)
    tool_calls = data.get('tool_calls', [])
    if not tool_calls:
        return jsonify({'status': 'error', 'reply': i18n.get('errors.noExecutableHardwareOps', lang)})

    try:
        success, result = hardware_controller.execute_tool_calls(tool_calls)
        if success:
            return jsonify({'status': 'success', 'reply': i18n.get('success.allHardwareOpsExecuted', lang), 'result': result})
        else:
            msg = result.get('message', '') if isinstance(result, dict) else str(result)
            return jsonify({'status': 'error', 'reply': i18n.get('errors.partialOpFailed', lang).format(msg=msg), 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'reply': i18n.get('errors.hardwareExecutionError', lang).format(error=str(e))})


def handle_data_analysis(user_message: str) -> Response:
    """
    处理数据分析请求（智能交互模式）

    支持三种格式：
    1. "数据分析" - 触发交互式选择器（推荐）
    2. "数据分析：<算法名称>" - 使用指定算法分析默认文件
    3. "数据分析：<算法名称> <csv_path>" - 使用指定算法分析指定文件

    Args:
        user_message: 用户消息

    Returns:
        JSON响应
    """
    lang = i18n.get_lang(request)
    content = user_message.replace("数据分析：", "").replace("数据分析", "").strip()

    # 场景1：用户只输入"数据分析"，触发交互式选择器
    if not content:
        available_algorithms = software_manager.list_algorithms()

        # 获取可用的CSV文件列表
        csv_files = []
        for folder in ["temporal", "extract"]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(folder, file))

        algo_list = '\n'.join([f'  • {algo["name"]}: {algo["description"]}' for algo in available_algorithms])
        file_list = ('\n'.join([f'  • {f}' for f in csv_files]) if csv_files else i18n.get('status.dataAnalysisModeNoFiles', lang))

        return jsonify({
            'type': 'data_analysis_selector',
            'algorithms': available_algorithms,
            'csv_files': csv_files,
            'reply': i18n.get('status.dataAnalysisMode', lang).format(algorithms=algo_list, files=file_list)
        })

    # 场景2和3：解析算法名称和CSV路径
    parts = content.split(maxsplit=1)
    algorithm_name = parts[0] if parts else ""
    csv_path = parts[1] if len(parts) > 1 else os.path.join(get_session_path("temporal"), "extraction.csv")

    if not algorithm_name:
        return jsonify({
            'type': 'system',
            'reply': i18n.get('errors.specifyAlgorithmName', lang)
        })

    # 检查算法是否存在
    available_algorithms = software_manager.list_algorithms()
    algorithm_exists = any(algo['name'] == algorithm_name for algo in available_algorithms)

    if not algorithm_exists:
        # 算法不存在，询问用户是否需要生成
        algo_list = '\n'.join([f'  • {algo["name"]}: {algo["description"]}' for algo in available_algorithms])
        return jsonify({
            'type': 'algorithm_not_found',
            'algorithm_name': algorithm_name,
            'reply': i18n.get('errors.algorithmNotFound', lang).format(name=algorithm_name, list=algo_list)
        })

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        return jsonify({
            'type': 'system',
            'reply': i18n.get('errors.csvFileNotFound', lang).format(path=csv_path)
        })

    # 算法和文件都存在，执行分析
    if task_manager.task_running:
        return jsonify({'type': 'system', 'reply': i18n.get('errors.taskAlreadyRunning', lang)})

    # 清空任务队列
    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    # 启动任务状态
    task_id = task_manager.generate_task_id()
    task_manager.current_task_id = task_id
    task_manager.task_running = True

    # 在后台线程中运行分析
    threading.Thread(
        target=software_manager.run_algorithm_on_csv,
        args=(algorithm_name, csv_path, task_manager)
    ).start()

    return jsonify({
        'type': 'task_trigger',
        'reply': i18n.get('status.algorithmAnalyzing', lang).format(name=algorithm_name, path=csv_path)
    })


def handle_data_analysis_execute(data: dict) -> Response:
    """
    执行用户确认的数据分析任务

    Args:
        data: 请求数据，包含 algorithm_name 和 csv_path

    Returns:
        JSON响应
    """
    lang = i18n.get_lang(request)
    algorithm_name = data.get('algorithm_name', '').strip()
    csv_path = data.get('csv_path', os.path.join(get_session_path("temporal"), "extraction.csv")).strip()

    if not algorithm_name:
        return jsonify({'status': 'error', 'reply': i18n.get('errors.missingAlgorithmName', lang)})

    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'reply': i18n.get('errors.fileNotExist', lang).format(path=csv_path)})

    if task_manager.task_running:
        return jsonify({'status': 'error', 'reply': i18n.get('errors.taskRunning', lang)})

    # 清空任务队列
    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    # 启动任务
    task_id = task_manager.generate_task_id()
    task_manager.current_task_id = task_id
    task_manager.task_running = True

    # 在后台线程中运行分析
    threading.Thread(
        target=software_manager.run_algorithm_on_csv,
        args=(algorithm_name, csv_path, task_manager)
    ).start()

    return jsonify({
        'type': 'task_trigger',
        'reply': i18n.get('status.algorithmAnalyzing', lang).format(name=algorithm_name, path=csv_path)
    })


def handle_auto_analyze(user_message: str) -> Response:
    """
    处理自动数据分析请求（旧版本，保留兼容性）

    用户消息格式："数据分析：<csv_path>"
    csv_path 为空时默认使用当前会话的 temporal/extraction.csv

    Args:
        user_message: 用户消息

    Returns:
        JSON响应（task_trigger 类型，触发前端 SSE 监听）
    """
    lang = i18n.get_lang(request)
    if task_manager.task_running:
        return jsonify({'type': 'system', 'reply': i18n.get('errors.taskAlreadyRunning', lang)})

    csv_path = user_message.replace("数据分析：", "").strip()
    if not csv_path:
        csv_path = os.path.join(get_session_path("temporal"), "extraction.csv")

    # 清空任务队列
    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    # 启动任务状态
    task_id = task_manager.generate_task_id()
    task_manager.current_task_id = task_id
    task_manager.task_running = True

    # 在后台线程中运行分析流水线
    threading.Thread(
        target=software_manager.auto_analyze,
        args=(csv_path, task_manager)
    ).start()

    return jsonify({
        'type' : 'task_trigger',
        'reply': i18n.get('status.analyzing', lang).format(path=csv_path)
    })


def handle_normal_chat(user_message: str, history: list = None) -> Response:
    """
    处理普通聊天 — SSE 流式输出，支持思考和正文分离。

    Args:
        user_message: 用户消息
        history: 前端传来的对话历史 [{role, content, ...}]

    Returns:
        SSE 流式响应 (text/event-stream)

    TODO: /api/chat_with_tools 路由 — 使用 llm_client.run_with_tools() 处理 tool-use 对话
    TODO: SSE 事件类型扩展 — 新增 tool_call_start / tool_call_result / tool_call_end
    """
    model = config.MODEL_NAME_TALK

    # Build messages from history
    messages = []
    if history:
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            api_role = "assistant" if role == "ai" else "user"
            msg = {"role": api_role, "content": content}
            # 保留 reasoning_content 以支持 DeepSeek 多轮对话
            if role == "ai" and m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    def raw_lines():
        """Yield raw SSE lines from the LLM API."""
        try:
            yield from llm_client.stream_raw(model, messages)
        except Exception as e:
            # If the streaming request itself fails, emit an error line
            # that the StreamAdapter will pick up.
            pass  # StreamAdapter will handle the empty stream case

    adapter = StreamAdapter()
    return sse_response(adapter.adapt(raw_lines()))


def handle_generate_algorithm(user_message: str) -> Response:
    """
    处理算法生成请求

    用户消息格式："生成算法：<算法描述>"

    Args:
        user_message: 用户消息

    Returns:
        JSON响应
    """
    lang = i18n.get_lang(request)
    description = user_message.replace("生成算法：", "").strip()

    if not description:
        return jsonify({
            'type': 'system',
            'reply': i18n.get('info.algorithmDescriptionHint', lang)
        })

    try:
        result = software_manager.generate_algorithm(description)

        if result.get("success"):
            reply = i18n.get('success.algorithmGenerated', lang).format(
                name=result['name'],
                filepath=result['filepath'],
                message=result.get('message', '')
            )
        else:
            reply = i18n.get('errors.algorithmGenerationFailed', lang).format(
                message=result.get('message', i18n.get('errors.unknownError', lang))
            )

        return jsonify({'type': 'system', 'reply': reply})

    except Exception as e:
        return jsonify({
            'type': 'system',
            'reply': i18n.get('errors.algorithmGenerationException', lang).format(error=str(e))
        })


@app.route('/api/hardware_status', methods=['GET'])
def hardware_status():
    """
    获取硬件状态
    """
    status = hardware_controller.get_hardware_status()
    return jsonify(status)


@app.route('/api/hardware_tools', methods=['GET'])
def get_hardware_tools():
    """返回所有注册的硬件工具及其参数 Schema，供单步控制面板使用"""
    tools_data = []
    for tool in hardware_controller.agent.hardware_tools:
        tools_data.append({
            "name": tool.name,
            "description": tool.description,
            "params": {
                param_name: {
                    "type": param_info.get("type"),
                    "description": param_info.get("description"),
                    "required": param_info.get("required", False),
                    "default": param_info.get("default"),
                }
                for param_name, param_info in tool.params.items()
            }
        })
    return jsonify({"tools": tools_data})


@app.route('/api/available_hardware', methods=['GET'])
def available_hardware():
    """
    获取可用硬件列表
    """
    hardware_list = hardware_controller.get_available_hardware()
    return jsonify({"hardware": hardware_list})


@app.route('/api/streaming_status', methods=['GET'])
def streaming_status():
    """
    获取流式响应状态
    """
    status = adaptive_handler.get_status()
    return jsonify(status)


@app.route('/api/extraction_mode', methods=['GET'])
def get_extraction_mode():
    """
    获取当前PDF提取模式
    """
    lang = i18n.get_lang(request)
    return jsonify({
        "mode": config.EXTRACTION_MODE,
        "available_modes": {
            "vision": i18n.get('info.modeVisionDesc', lang),
            "text": i18n.get('info.modeTextDesc', lang),
            "hybrid": i18n.get('info.modeHybridDesc', lang)
        }
    })


@app.route('/api/extraction_mode', methods=['POST'])
def set_extraction_mode():
    """
    设置PDF提取模式
    """
    data = request.json
    mode = data.get('mode', '').strip()

    valid_modes = ['vision', 'text', 'hybrid']
    if mode not in valid_modes:
        lang = i18n.get_lang(request)
        return jsonify({
            'success': False,
            'message': i18n.get('errors.invalidExtractionMode', lang).format(modes=", ".join(valid_modes))
        }), 400

    config.EXTRACTION_MODE = mode

    lang = i18n.get_lang(request)
    mode_names = {
        'vision': i18n.get('info.modeVision', lang),
        'text': i18n.get('info.modeText', lang),
        'hybrid': i18n.get('info.modeHybrid', lang)
    }

    return jsonify({
        'success': True,
        'mode': mode,
        'message': i18n.get('success.extractionModeChanged', lang).format(mode=mode_names[mode])
    })


@app.route('/api/streaming_recheck', methods=['POST'])
def streaming_recheck():
    """
    强制重新检测流式支持
    """
    result = adaptive_handler.force_recheck()
    lang = i18n.get_lang(request)
    return jsonify({
        "supports_streaming": result,
        "message": i18n.get('status.streamingRechecked', lang) if result else i18n.get('errors.streamingNotSupported', lang)
    })


@app.route('/api/platform_check', methods=['GET'])
def platform_check():
    """平台诊断 — 本地快速检查（无网络，< 1.5s）"""
    global _platform_status
    return jsonify(_platform_status)


@app.route('/api/platform_check/full', methods=['POST'])
def platform_check_full():
    """平台诊断 — 全量检查（含网络，15-30s）"""
    try:
        result = _run_platform_checks(scope="full")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """
    404错误处理
    """
    lang = i18n.get_lang(request)
    return jsonify({'error': i18n.get('errors.endpointNotFound', lang)}), 404


@app.errorhandler(500)
def internal_error(error):
    """
    500错误处理
    """
    import traceback
    traceback.print_exc()
    lang = i18n.get_lang(request)
    return jsonify({'error': i18n.get('errors.internalServerError', lang)}), 500


# =============================================================================
# 实验设计路由（基于 PydanticAI Agent 原生 tool-use）
# =============================================================================

@app.route('/api/experiment_chat', methods=['POST'])
def experiment_chat():
    """
    实验设计对话 - 使用自然语言生成实验设计JSON

    支持两种模式：
    - 非流式（默认）：POST body 不含 stream 或 stream=false
      返回 JSON: {"type": "experiment_design", "experiment_json": {...}, ...}
    - 流式：POST body 含 stream=true
      返回 SSE 流: chunk 事件 → complete 事件
    """
    lang = i18n.get_lang(request)
    data = request.json
    if data is None:
        return jsonify({'type': 'error', 'reply': i18n.get('errors.emptyOrInvalidJson', lang)}), 400

    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'type': 'error', 'reply': i18n.get('errors.messageEmpty', lang)})

    use_stream = data.get('stream', False)

    try:
        from core.field_inference import ExperimentDesignAgent
        from experiment.format import ExperimentFormatConverter

        if use_stream:
            # ---- SSE 流式模式 ----
            agent = ExperimentDesignAgent()
            def event_stream():
                for event_str in agent.parse_experiment_design_stream(user_message):
                    yield event_str
            return Response(event_stream(), mimetype="text/event-stream")

        # ---- 非流式模式（保持向后兼容） ----
        agent = ExperimentDesignAgent()
        converter = ExperimentFormatConverter()

        success, result = agent.parse_experiment_design(user_message)

        if success:
            import datetime
            result['created_at'] = datetime.datetime.now().isoformat()
            visual_data = converter.json_to_visual(result)
            var_count = len(result.get('variables', {}))
            var_hint = ""
            exp_name = result.get('experiment_name', i18n.get('experiment.unnamed', lang))
            exp_desc = result.get('description', '')
            steps_count = len(result.get('steps', []))
            if var_count:
                var_hint = "\n\n" + i18n.get('info.variableCountHint', lang).format(count=var_count)
            return jsonify({
                'type': 'experiment_design',
                'experiment_json': result,
                'visual_data': visual_data,
                'reply': i18n.get('success.experimentDesignGenerated', lang).format(
                    name=exp_name, description=exp_desc, steps=steps_count, var_hint=var_hint
                )
            })
        else:
            return jsonify({
                'type': 'error',
                'reply': i18n.get('errors.experimentDesignFailed', lang).format(error=result)
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'type': 'error',
            'reply': i18n.get('errors.internalServerErrorWithMsg', lang).format(msg=str(e))
        }), 500


@app.route('/api/experiment_chat_stream', methods=['GET'])
def experiment_chat_stream_get():
    """GET 方式流式实验设计（供 platform_init 测试脚本和直接调试使用）"""
    user_message = request.args.get('message', '').strip()
    if not user_message:
        lang = i18n.get_lang(request)
        return jsonify({'type': 'error', 'reply': i18n.get('errors.messageParamEmpty', lang)}), 400

    from core.field_inference import ExperimentDesignAgent
    agent = ExperimentDesignAgent()
    def event_stream():
        for event_str in agent.parse_experiment_design_stream(user_message):
            yield event_str
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/api/capabilities', methods=['GET'])
def get_capabilities():
    """返回平台能力配置（模型流式支持等），前端据此选择请求方式"""
    return jsonify(_stream_capabilities)


@app.route('/api/experiment_upload', methods=['POST'])
def experiment_upload():
    """
    实验设计模式的 PDF 上传（方案2暂不支持PDF读取）

    TODO: 方案2不支持交互式PDF读取，如需此功能请使用方案1（PydanticAI）
    """
    session_id = request.form.get('session_id', 'default')
    if 'file' not in request.files:
        lang = i18n.get_lang(request)
        return jsonify({'error': i18n.get('errors.noFileReceived', lang)}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        lang = i18n.get_lang(request)
        return jsonify({'error': i18n.get('errors.onlyPdfAllowed', lang)}), 400

    os.makedirs('./pdf_cache', exist_ok=True)
    safe_name = f"{session_id}_{uuid.uuid4().hex}.pdf"
    path = os.path.join('./pdf_cache', safe_name)
    file.save(path)

    # TODO: 方案2不支持交互式PDF读取，保留路径供未来扩展
    # experiment_agent.set_pdf_path(session_id, path)
    return jsonify({'filename': safe_name, 'path': path})


@app.route('/api/experiment_confirm', methods=['POST'])
def experiment_confirm():
    """
    处理实验确认响应（方案2暂不支持交互式确认）

    TODO: 方案2不支持交互式确认，如需此功能请使用方案1（PydanticAI）

    请求体：
    {
        "request_id": "uuid",
        "session_id": "session_id",
        "action": "confirm" | "skip" | "cancel",
        "params": {...}  # Modified parameters (optional)
    }
    """
    data = request.json
    request_id = data.get('request_id')
    session_id = data.get('session_id')
    action = data.get('action')
    params = data.get('params', {})

    if not request_id or not session_id:
        return jsonify({'error': 'Missing request_id or session_id'}), 400

    # TODO: 方案2不支持交互式确认，保留接口供未来扩展
    # Submit response to the agent's queue
    # response = {
    #     "action": action,
    #     "params": params
    # }
    # experiment_agent.submit_response(request_id, response)

    lang = i18n.get_lang(request)
    return jsonify({'status': 'success', 'message': i18n.get('info.scheme2NotSupportInteractive', lang)})


# =============================================================================
# 软件算法路由（SoftwareManager 统一管理）
# =============================================================================

@app.route('/api/software/algorithms', methods=['GET'])
def software_algorithms():
    """
    获取所有可用软件算法列表
    返回每个算法的名称、描述和参数规格
    """
    algorithms = software_manager.list_algorithms()
    return jsonify({"algorithms": algorithms})


@app.route('/api/software/run', methods=['POST'])
def software_run():
    """
    运行指定算法

    请求体：
        {
            "algorithm": "data_statistics",
            "data"     : {"col_a": [1, 2, 3], "col_b": [4, 5, 6]},
            "params"   : {"include_correlation": true}
        }
    """
    data = request.json
    algorithm_name = data.get('algorithm', '').strip()
    input_data     = data.get('data')
    params         = data.get('params', {})

    lang = i18n.get_lang(request)
    if not algorithm_name:
        return jsonify({'success': False, 'message': i18n.get('errors.missingAlgorithmField', lang)}), 400
    if input_data is None:
        return jsonify({'success': False, 'message': i18n.get('errors.missingDataField', lang)}), 400

    result = software_manager.run_algorithm(algorithm_name, input_data, params)
    return jsonify(result)


@app.route('/api/software/run_on_csv', methods=['POST'])
def software_run_on_csv():
    """
    对当前会话的 temporal/extraction.csv 中的数值列运行算法（提取任务完成后可直接使用）

    请求体：
        {
            "algorithm": "data_statistics",
            "params"   : {"include_correlation": true}
        }
    """
    data           = request.json
    algorithm_name = data.get('algorithm', '').strip()
    params         = data.get('params', {})

    if not algorithm_name:
        lang = i18n.get_lang(request)
        return jsonify({'success': False, 'message': i18n.get('errors.missingAlgorithmField', lang)}), 400

    result = software_manager.run_on_csv(algorithm_name, params)
    return jsonify(result)


@app.route('/api/software/generate_algorithm', methods=['POST'])
def software_generate_algorithm():
    """
    使用 LLM 根据自然语言描述自动生成新算法并保存到项目

    请求体：
        {
            "description": "我需要一个对光谱数据做高斯平滑的算法，输入是 wavelength 和 intensity 列表，窗口大小可配置"
        }

    生成成功后调用 /api/software/reload 使新算法立即生效。
    """
    data        = request.json
    description = data.get('description', '').strip()

    if not description:
        lang = i18n.get_lang(request)
        return jsonify({'success': False, 'message': i18n.get('errors.missingDescriptionField', lang)}), 400

    result = software_manager.generate_algorithm(description)
    return jsonify(result)


@app.route('/api/software/reload', methods=['POST'])
def software_reload():
    """
    重新扫描并注册算法（生成新算法后调用，使其立即可用）
    """
    algorithms = software_manager.reload_algorithms()
    lang = i18n.get_lang(request)
    return jsonify({
        'success'   : True,
        'count'     : len(algorithms),
        'algorithms': algorithms,
        'message'   : i18n.get('success.algorithmsReloaded', lang).format(count=len(algorithms)),
    })


# =============================================================================
# 算法交互式选择路由
# =============================================================================

@app.route('/api/list_algorithms', methods=['GET'])
def list_algorithms():
    """
    获取算法列表（带标签和图标信息）
    """
    algorithms = software_manager.list_algorithms()

    # 为每个算法添加标签和图标
    for algo in algorithms:
        algo['tags'] = algorithm_parser.get_tags(algo['name'])
        algo['icon'] = algorithm_parser.get_icon(algo['name'])

    return jsonify({
        "success": True,
        "algorithms": algorithms
    })


@app.route('/api/parse_algorithm', methods=['POST'])
def parse_algorithm():
    """
    解析用户输入，判断是否指定了算法名称

    输入: {"user_input": "使用数据统计分析"}
    输出: {"algorithm_found": true, "algorithm": "data_statistics", ...}
    """
    data = request.json
    user_input = data.get('user_input', '').strip()

    # 获取可用算法列表
    available_algorithms = software_manager.list_algorithms()

    # 使用算法解析器解析
    result = algorithm_parser.parse(user_input, available_algorithms)

    return jsonify(result)


@app.route('/api/recent_files', methods=['GET'])
def get_recent_files():
    """返回推荐文件列表：全局 temporal + 当前会话 extract/temporal + 最近会话的 extract"""
    import glob
    import time

    files = []
    seen = set()  # 去重

    # 1. 全局 temporal（dialogue data/temporal/）— 暂存数据，优先
    if os.path.isdir(GLOBAL_TEMPORAL_DIR):
        for fp in glob.glob(os.path.join(GLOBAL_TEMPORAL_DIR, "*.csv")):
            try:
                stat = os.stat(fp)
                path = fp.replace('\\', '/')
                if path not in seen:
                    seen.add(path)
                    files.append({
                        'path': path,
                        'name': os.path.basename(fp),
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'modified_str': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime)),
                        'source': 'temporal',
                    })
            except Exception:
                continue

    # 2. 当前会话的 temporal/ 和 extract/
    session_temporal = get_session_path("temporal")
    session_extract = get_session_path("extract")

    for folder in [session_temporal, session_extract]:
        if not os.path.isdir(folder):
            continue
        for fp in glob.glob(os.path.join(folder, "*.csv")):
            try:
                stat = os.stat(fp)
                path = fp.replace('\\', '/')
                if path not in seen:
                    seen.add(path)
                    files.append({
                        'path': path,
                        'name': os.path.basename(fp),
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'modified_str': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime)),
                        'source': 'session',
                    })
            except Exception:
                continue

    # 3. 所有历史会话的 extract/ 和 temporal/ 目录
    history_dir = config.DIALOGUE_DATA_DIR
    if os.path.isdir(history_dir):
        try:
            sessions = sorted(
                [d for d in os.listdir(history_dir) if os.path.isdir(os.path.join(history_dir, d))],
                reverse=True
            )
            for sess in sessions:
                for sub in ["extract", "temporal"]:
                    sub_dir = os.path.join(history_dir, sess, sub)
                    if not os.path.isdir(sub_dir):
                        continue
                    for fp in glob.glob(os.path.join(sub_dir, "*.csv")):
                        try:
                            stat = os.stat(fp)
                            path = fp.replace('\\', '/')
                            if path not in seen:
                                seen.add(path)
                                files.append({
                                    'path': path,
                                    'name': f"[{sess[:8]}] {os.path.basename(fp)}",
                                    'size': stat.st_size,
                                    'modified': stat.st_mtime,
                                    'modified_str': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime)),
                                    'source': 'history',
                                })
                        except Exception:
                            continue
        except Exception:
            pass

    # 按修改时间倒序排序，返回前20个
    files.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify({
        "success": True,
        "files": files[:20]
    })


@app.route('/api/browse_csv', methods=['GET'])
def browse_csv():
    """列出可用的输入 CSV 文件：全局 temporal/ + 当前会话 extract/ + const_data/extract/"""
    import glob
    files = []
    paths_to_scan = [
        get_session_path("extract"),
        get_session_path("temporal"),
        os.path.join(config.DIALOGUE_DATA_DIR, "const_data", "extract"),
    ]
    for folder in paths_to_scan:
        for fp in glob.glob(os.path.join(folder, "*.csv")):
            files.append({
                'path': fp.replace('\\', '/'),
                'name': os.path.basename(fp),
                'folder': folder.replace('\\', '/'),
            })
    return jsonify({"success": True, "files": files})


@app.route('/api/browse_output_dirs', methods=['GET'])
def browse_output_dirs():
    """列出可用的输出目录：当前会话 results/ 以及 const_data/result/ 下的子文件夹"""
    dirs = []
    session_results = get_session_path("results")
    lang = i18n.get_lang(request)
    dirs.append({'path': session_results.replace('\\', '/'), 'label': i18n.get('info.currentSessionResults', lang), 'is_default': True})
    const_result = os.path.join(config.DIALOGUE_DATA_DIR, "const_data", "result")
    if os.path.isdir(const_result):
        for name in sorted(os.listdir(const_result)):
            full = os.path.join(const_result, name)
            if os.path.isdir(full):
                dirs.append({'path': full.replace('\\', '/'), 'label': name, 'is_default': False})
    return jsonify({"success": True, "dirs": dirs})


@app.route('/api/generate_algorithm', methods=['POST'])
def generate_algorithm_alias():
    """
    生成新算法的简化接口（前端调用）

    请求体：
        {
            "description": "算法描述"
        }
    """
    data = request.json
    description = data.get('description', '').strip()

    if not description:
        lang = i18n.get_lang(request)
        return jsonify({'success': False, 'message': i18n.get('errors.missingAlgorithmDescription', lang)}), 400

    # 调用软件管理器生成算法
    result = software_manager.generate_algorithm(description)

    # 如果生成成功，自动重新加载算法列表
    if result.get('success'):
        software_manager.reload_algorithms()

    return jsonify(result)


@app.route('/api/algorithm_gen/guide', methods=['POST'])
def algorithm_gen_guide():
    """逐步引导式算法生成。"""
    data = request.json or {}
    action = data.get('action', 'answer')

    resp = algorithm_guide.handle(
        session_id=data.get('session_id'),
        answer=data.get('answer'),
        action=action,
    )

    if resp.get('stage') == 'cancelled':
        return jsonify(resp)

    if resp.get('stage') == 'ready':
        combined = resp['combined_prompt']
        result = software_manager.generate_algorithm(combined)
        if result.get('success'):
            software_manager.reload_algorithms()
        resp = algorithm_guide.finish(resp['session_id'], result)

    return jsonify(resp)


@app.route('/api/run_algorithm', methods=['POST'])
def run_algorithm_with_file():
    """
    执行指定算法（用户选择文件后）

    输入: {
        "algorithm": "data_statistics",
        "file_path": "temporal/extraction.csv",
        "params": {"include_correlation": true}
    }
    """
    data = request.json
    algo_name = data.get('algorithm', '').strip()
    file_path = data.get('file_path', '').strip()
    params = data.get('params', {})

    lang = i18n.get_lang(request)
    if not algo_name:
        return jsonify({
            "success": False,
            "message": i18n.get('errors.missingAlgorithmName', lang)
        }), 400

    if not file_path:
        return jsonify({
            "success": False,
            "message": i18n.get('errors.missingFilePath', lang)
        }), 400

    # 验证文件存在
    if not os.path.exists(file_path):
        return jsonify({
            "success": False,
            "message": i18n.get('errors.fileNotExist', lang).format(path=file_path)
        }), 404

    # 检查是否有任务正在运行
    if task_manager.task_running:
        return jsonify({
            "success": False,
            "message": i18n.get('errors.taskAlreadyRunning', lang)
        }), 409

    # 清空任务队列
    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    # 启动任务
    task_id = task_manager.generate_task_id()
    task_manager.current_task_id = task_id
    task_manager.task_running = True

    # 在后台线程中运行算法
    threading.Thread(
        target=software_manager.run_algorithm_on_csv,
        args=(algo_name, file_path, params, task_manager)
    ).start()

    return jsonify({
        "success": True,
        "task_id": task_id,
        "message": i18n.get('status.algorithmExecuting', lang).format(name=algo_name)
    })


# =============================================================================
# 实验设计画布路由
# =============================================================================

@app.route('/api/save_experiment_design', methods=['POST'])
def save_experiment_design():
    """
    保存实验设计JSON到指定路径或当前会话的文件夹

    请求体：
    {
        "experiment_name": "旋涂实验_v1",
        "created_at": "2026-04-17T...",
        "steps": [...],
        "save_path": "可选，完整保存路径（含文件名）"
    }
    """
    data = request.json
    lang = i18n.get_lang(request)
    experiment_name = data.get('experiment_name', i18n.get('experiment.unnamed', lang))
    custom_path = data.get('save_path')

    # 如果提供了自定义路径，使用自定义路径
    if custom_path:
        filepath = custom_path
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    else:
        # 使用当前会话的实验设计保存目录
        design_folder = get_session_path('experiment_designs')
        os.makedirs(design_folder, exist_ok=True)

        # 生成文件名（带时间戳避免覆盖）
        timestamp = json.dumps(data.get('created_at', '')).strip('"').replace(':', '-').replace('.', '-')[:19]
        safe_name = experiment_name.replace(' ', '_').replace('/', '_')
        filename = f"{safe_name}_{timestamp}.json"
        filepath = os.path.join(design_folder, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'filepath': filepath,
            'message': i18n.get('success.experimentDesignSaved', lang).format(filepath=filepath)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': i18n.get('errors.saveFailed', lang).format(error=str(e))
        }), 500


@app.route('/api/execute_experiment_design', methods=['POST'])
def execute_experiment_design():
    """
    执行实验设计JSON中的步骤序列，委托给 ExperimentManager.execute_plan。

    请求体：
    {
        "experiment_name": "旋涂实验_v1",
        "steps": [
            {"type": "tool",     "name": "spin_coating",    "params": {...}},
            {"type": "helper",   "name": "WAIT",            "params": {"duration": 5000}},
            {"type": "software", "name": "data_statistics", "params": {}, "input_file": "...", "output_file": "..."}
        ]
    }
    """
    from experiment.executor import ExperimentExecutor

    data = request.json
    lang = i18n.get_lang(request)
    experiment_name = data.get('experiment_name', i18n.get('experiment.unnamed', lang))
    steps = data.get('steps', [])

    if not steps:
        return jsonify({'type': 'error', 'reply': i18n.get('errors.experimentDesignNoSteps', lang)}), 400

    if task_manager.task_running:
        return jsonify({'type': 'error', 'reply': i18n.get('errors.taskAlreadyRunning', lang)}), 409

    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    task_id = task_manager.generate_task_id()
    task_manager.current_task_id = task_id
    task_manager.task_running = True

    def _run():
        try:
            executor = ExperimentExecutor(software_manager=software_manager, hardware_agent=hardware_controller.agent)
            total = len(steps)

            def on_progress(step_num, status, message):
                msg_type = "info" if status in ("running", "completed") else "error"
                task_manager.put_task_message({"type": msg_type, "data": message})

            task_manager.put_task_message({"type": "info", "data": i18n.get('status.experimentExecutionStarted', lang).format(name=experiment_name, total=total)})
            result = executor.execute_plan(data, progress_callback=on_progress)

            if result["success"]:
                task_manager.put_task_message({"type": "complete", "data": {"message": i18n.get('success.experimentExecutionCompleted', lang).format(name=experiment_name)}})
            else:
                err = result.get("error") or i18n.get('errors.unknownError', lang)
                task_manager.put_task_message({"type": "complete", "data": {"error": err}})
        except Exception as e:
            task_manager.put_task_message({"type": "complete", "data": {"error": i18n.get('errors.hardwareExecutionError', lang).format(error=str(e))}})
        finally:
            task_manager.task_running = False

    threading.Thread(target=_run, daemon=True).start()

    return jsonify({
        'type': 'task_trigger',
        'reply': i18n.get('status.experimentExecutingDesign', lang).format(name=experiment_name, steps=len(steps))
    })


@app.route('/api/variables/import_csv', methods=['POST'])
def import_variables_csv():
    """
    导入CSV生成变量定义和批量数据

    请求体：
    {
        "csv_content": "name,value\\nspeed,3000\\n..."
    }

    返回：
    {
        "type": "variables_csv",
        "variables": {"speed": {"type": "int", "default_value": 3000, "constraints": {}}},
        "batch_data": [{"speed": 3000}, ...],
        "reply": "✅ CSV 解析完成..."
    }
    """
    lang = i18n.get_lang(request)
    data = request.json
    if data is None:
        return jsonify({'type': 'error', 'reply': i18n.get('errors.emptyOrInvalidJson', lang)}), 400

    csv_content = data.get('csv_content', '')
    if not csv_content or not csv_content.strip():
        return jsonify({'type': 'error', 'reply': i18n.get('errors.csvContentEmpty', lang)}), 400

    try:
        from core.variable_resolver import VariableResolver
        variables, batch_data, error = VariableResolver.parse_csv(csv_content)
        if error:
            return jsonify({'type': 'error', 'reply': i18n.get('errors.csvParseFailed', lang).format(error=error)}), 400

        var_count = len(variables)
        row_count = len(batch_data)
        var_names = ", ".join(variables.keys())
        reply = i18n.get('success.csvParsed', lang).format(var_count=var_count, row_count=row_count, var_names=var_names)

        return jsonify({
            'type': 'variables_csv',
            'variables': variables,
            'batch_data': batch_data,
            'reply': reply
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'type': 'error', 'reply': i18n.get('errors.csvImportFailed', lang).format(error=str(e))}), 500


@app.route('/api/export_experiment_json', methods=['POST'])
def export_experiment_json():
    """
    导出实验设计JSON到指定路径

    请求体：
    {
        "json_data": {...},
        "filepath": "hardware/design_of_experiments/experiment_name.json"
    }
    """
    data = request.json
    lang = i18n.get_lang(request)
    json_data = data.get('json_data')
    filepath = data.get('filepath', '').strip()

    if not json_data:
        return jsonify({
            'success': False,
            'message': i18n.get('errors.missingJsonData', lang)
        }), 400

    if not filepath:
        return jsonify({
            'success': False,
            'message': i18n.get('errors.missingFilePath', lang)
        }), 400

    try:
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 写入JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'filepath': filepath,
            'message': i18n.get('success.experimentDesignExported', lang).format(filepath=filepath)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': i18n.get('errors.exportFailed', lang).format(error=str(e))
        }), 500


@app.route('/api/compile_experiment', methods=['POST'])
def compile_experiment():
    """
    将实验设计JSON编译为Python代码

    请求体：
    {
        "experiment_json": {...}
    }

    返回：
    {
        "success": true,
        "code": "# Python代码..."
    }
    """
    data = request.json
    experiment_json = data.get('experiment_json')

    if not experiment_json:
        lang = i18n.get_lang(request)
        return jsonify({
            'success': False,
            'message': i18n.get('errors.missingExperimentJson', lang)
        }), 400

    try:
        from experiment.compiler import ExperimentCompiler
        compiler = ExperimentCompiler()
        python_code = compiler.compile_to_python(experiment_json)

        return jsonify({
            'success': True,
            'code': python_code
        })
    except Exception as e:
        lang = i18n.get_lang(request)
        return jsonify({
            'success': False,
            'message': i18n.get('errors.compileFailed', lang).format(error=str(e))
        }), 500


@app.route('/api/compile_and_run_experiment', methods=['POST'])
def compile_and_run_experiment():
    """
    将实验设计JSON编译为Python代码并执行

    请求体：
    {
        "experiment_json": {...}
    }

    返回：
    {
        "success": true,
        "code": "# Python代码...",
        "output": "执行输出...",
        "error": ""
    }
    """
    data = request.json
    experiment_json = data.get('experiment_json')

    if not experiment_json:
        lang = i18n.get_lang(request)
        return jsonify({
            'success': False,
            'message': i18n.get('errors.missingExperimentJson', lang)
        }), 400

    try:
        from experiment.compiler import ExperimentCompiler
        compiler = ExperimentCompiler()
        result = compiler.compile_and_run(experiment_json)

        return jsonify(result)
    except Exception as e:
        lang = i18n.get_lang(request)
        return jsonify({
            'success': False,
            'message': i18n.get('errors.compileOrRunFailed', lang).format(error=str(e))
        }), 500


@app.route('/api/get_session_path', methods=['GET'])
def get_session_path_api():
    """
    获取当前会话的数据路径

    查询参数：
        subdir: 子目录名称（可选），如 "extract", "temporal", "results", "experiment_designs"

    返回：
        {
            "success": true,
            "path": "dialogue data/20260417_152030/experiment_designs",
            "timestamp": "20260417_152030"
        }
    """
    subdir = request.args.get('subdir', '')
    path = get_session_path(subdir)

    return jsonify({
        'success': True,
        'path': path,
        'timestamp': SESSION_TIMESTAMP
    })


# =============================================================================
# 对话历史持久化路由
# =============================================================================

@app.route('/api/history/save_batch', methods=['POST'])
def history_save_batch():
    """
    批量保存对话历史到当前会话的 chat_history.json。
    前端每 5 条消息触发一次，页面关闭时通过 sendBeacon 触发。

    请求体：
    {
        "messages": [
            {"role": "user", "content": "...", "timestamp": "...", "mode": "normal", ...},
            {"role": "ai",   "content": "...", "timestamp": "...", "mode": "normal", ...}
        ]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "message": i18n.get('errors.requestBodyEmpty', lang)}), 400

    messages = data.get('messages', [])

    # 读取已有历史文件（保留已生成的 title）
    history_path = os.path.join(SESSION_BASE_PATH, "chat_history.json")
    existing_title = None
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                existing_title = existing.get("title")
        except Exception:
            pass

    # 标题：已有则复用，否则尝试 LLM 生成
    title = existing_title
    if not title and len(messages) >= 3:
        title = _generate_title(messages, lang=i18n.get_lang(request))
    if not title:
        title = "未命名会话"

    session_info = {
        "timestamp": SESSION_TIMESTAMP,
        "started_at": datetime.now().isoformat(),
        "saved_at": datetime.now().isoformat(),
        "message_count": len(messages)
    }

    history_data = {
        "title": title,
        "session": session_info,
        "outputs": _scan_session_outputs(),
        "messages": messages
    }

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    # 只索引已拟定标题的会话（未命名会话不显示在历史列表中）
    if title and title != "未命名会话":
        _update_session_index({
            "timestamp": SESSION_TIMESTAMP,
            "started_at": session_info["started_at"],
            "saved_at": session_info["saved_at"],
            "message_count": len(messages),
            "title": title,
            "path": SESSION_TIMESTAMP
        })

    return jsonify({"success": True, "saved_count": len(messages)})


@app.route('/api/history/clear_cache', methods=['POST'])
def history_clear_cache():
    """
    清除所有未拟定标题的历史对话文件夹。

    返回: { success: true, deleted_count: int, deleted_folders: [...] }
    """
    from utils.cache_cleaner import clear_untitled_sessions
    result = clear_untitled_sessions(config.DIALOGUE_DATA_DIR)
    return jsonify({"success": True, **result})


@app.route('/api/history/sessions', methods=['GET'])
def history_sessions():
    """返回所有已拟定标题的历史会话索引列表（过滤掉未命名会话）。"""
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"sessions": []}
    # 只返回已拟定标题的会话
    data["sessions"] = [
        s for s in data.get("sessions", [])
        if s.get("title") and s["title"] != "未命名会话"
    ]
    return jsonify(data)


@app.route('/api/history/session/<timestamp>', methods=['GET'])
def history_load_session(timestamp: str):
    """
    加载指定会话的 chat_history.json。

    URL: /api/history/session/20260507_190432
    返回: { success, data: { title, messages, outputs } } 或 { success: false, error }
    """
    # 安全检查：timestamp 只能包含数字和下划线
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.invalidTimestampFormat', lang)}), 400

    history_path = os.path.join(config.DIALOGUE_DATA_DIR, timestamp, "chat_history.json")
    if not os.path.exists(history_path):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.sessionNotFound', lang)}), 404

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.readFailed', lang).format(error=str(e))}), 500

    return jsonify({
        "success": True,
        "data": {
            "title": data.get("title"),
            "messages": data.get("messages", []),
            "outputs": data.get("outputs", {}),
        }
    })


@app.route('/api/history/session/<timestamp>', methods=['DELETE'])
def history_delete_session(timestamp: str):
    """
    软删除会话：从 sessions_index.json 移除条目，保留文件夹和 chat_history.json。

    返回: { success: true } 或 { success: false, error }
    """
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.invalidTimestampFormat', lang)}), 400

    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if not os.path.exists(index_path):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.indexFileNotFound', lang)}), 404

    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_len = len(data.get("sessions", []))
    data["sessions"] = [s for s in data.get("sessions", []) if s.get("timestamp") != timestamp]

    if len(data["sessions"]) == original_len:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.sessionNotFound', lang)}), 404

    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.writeFailed', lang).format(error=str(e))}), 500

    return jsonify({"success": True})


@app.route('/api/history/session/<timestamp>/title', methods=['PUT'])
def history_update_title(timestamp: str):
    """
    更新会话标题，同步写入 sessions_index.json 和 chat_history.json。

    Body: {"title": "新标题"}
    返回: { success: true, title }
    """
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.invalidTimestampFormat', lang)}), 400

    data = request.get_json(force=True, silent=True)
    if not data or "title" not in data:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.missingTitleField', lang)}), 400

    new_title = data["title"].strip()
    if not new_title or len(new_title) > 100:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.titleLengthInvalid', lang)}), 400

    # 验证会话存在
    history_path = os.path.join(config.DIALOGUE_DATA_DIR, timestamp, "chat_history.json")
    if not os.path.exists(history_path):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.sessionNotFound', lang)}), 404

    # 1. 更新 sessions_index.json
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            for s in index_data.get("sessions", []):
                if s.get("timestamp") == timestamp:
                    s["title"] = new_title
                    break
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            lang = i18n.get_lang(request)
            return jsonify({"success": False, "error": i18n.get('errors.writeIndexFailed', lang).format(error=str(e))}), 500

    # 2. 更新 chat_history.json
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            hist_data = json.load(f)
        hist_data["title"] = new_title
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(hist_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.writeSessionFailed', lang).format(error=str(e))}), 500

    return jsonify({"success": True, "title": new_title})


@app.route('/api/history/folders', methods=['GET'])
def history_list_folders():
    """返回所有文件夹列表。"""
    return jsonify({"success": True, "folders": _read_folders()})


@app.route('/api/history/folders', methods=['POST'])
def history_create_folder():
    """
    创建文件夹。

    Body: {"name": "钙钛矿实验"}
    返回: { success: true, folder: { id, name, created_at } }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.requestBodyEmpty', lang)}), 400

    name = (data.get("name") or "").strip()
    if not name or len(name) > 50:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.folderNameLengthInvalid', lang)}), 400

    folders = _read_folders()
    folder = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "created_at": datetime.now().isoformat()
    }
    folders.append(folder)
    _write_folders(folders)

    return jsonify({"success": True, "folder": folder})


@app.route('/api/history/folders/<folder_id>', methods=['PUT'])
def history_rename_folder(folder_id: str):
    """
    重命名文件夹。

    Body: {"name": "新名称"}
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.requestBodyEmpty', lang)}), 400

    name = (data.get("name") or "").strip()
    if not name or len(name) > 50:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.folderNameLengthInvalid', lang)}), 400

    folders = _read_folders()
    for f in folders:
        if f["id"] == folder_id:
            f["name"] = name
            _write_folders(folders)
            return jsonify({"success": True, "folder": f})

    lang = i18n.get_lang(request)
    return jsonify({"success": False, "error": i18n.get('errors.folderNotFound', lang)}), 404


@app.route('/api/history/folders/<folder_id>', methods=['DELETE'])
def history_delete_folder(folder_id: str):
    """
    删除文件夹，该文件夹下的所有会话变为未分类（移除 folder_id）。
    """
    folders = _read_folders()
    original_len = len(folders)
    folders = [f for f in folders if f["id"] != folder_id]
    if len(folders) == original_len:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.folderNotFound', lang)}), 404
    _write_folders(folders)

    # 清除 sessions_index 中该文件夹的关联
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        for s in index_data.get("sessions", []):
            if s.get("folder_id") == folder_id:
                s.pop("folder_id", None)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True})


@app.route('/api/history/session/<timestamp>/move', methods=['PUT'])
def history_move_session(timestamp: str):
    """
    移动会话到指定文件夹（或移除文件夹关联）。

    Body: {"folder_id": "a1b2c3d4"}  或  {"folder_id": null}  移除关联
    返回: { success: true }
    """
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.invalidTimestampFormat', lang)}), 400

    data = request.get_json(force=True, silent=True)
    if not data or "folder_id" not in data:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.missingFolderIdField', lang)}), 400

    folder_id = data["folder_id"]  # None / null means remove from folder

    # 如果指定了 folder_id，验证文件夹存在
    if folder_id is not None:
        folders = _read_folders()
        if not any(f["id"] == folder_id for f in folders):
            lang = i18n.get_lang(request)
            return jsonify({"success": False, "error": i18n.get('errors.folderNotFound', lang)}), 404

    # 更新 sessions_index.json
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if not os.path.exists(index_path):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.indexFileNotFound', lang)}), 404

    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    found = False
    for s in index_data.get("sessions", []):
        if s.get("timestamp") == timestamp:
            found = True
            if folder_id is None:
                s.pop("folder_id", None)
            else:
                s["folder_id"] = folder_id
            break
    if not found:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.sessionNotFound', lang)}), 404
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True})


# =============================================================================
# Phase 3: 语义搜索 API
# =============================================================================

@app.route('/api/semantic_search', methods=['POST'])
def semantic_search():
    """
    语义搜索全文献库

    POST body: {"query": "钙钛矿钝化剂效率对比", "top_k": 10}
    返回匹配的页面列表，含文本片段和相似度
    """
    global _semantic_search_instance
    if _semantic_search_instance is None:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.semanticSearchNotInitialized', lang)}), 503

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    top_k = data.get("top_k", 10)

    if not query:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.queryEmpty', lang)}), 400

    if top_k < 1 or top_k > 100:
        top_k = 10

    results = _semantic_search_instance.search(query, top_k=top_k)
    total_pages = _semantic_search_instance.get_total_pages()

    return jsonify({
        "success": True,
        "query": query,
        "total_pages_indexed": total_pages,
        "result_count": len(results),
        "results": results,
    })


@app.route('/api/page_image', methods=['POST'])
def page_image():
    """
    获取 PDF 指定页面的图片（base64）

    POST body: {"pdf_path": "dialogue data/PDF_TARGET/xxx.pdf", "page_num": 2}
    返回 base64 编码的 JPEG 图片
    """
    data = request.get_json(silent=True) or {}
    pdf_path = (data.get("pdf_path") or "").strip()
    page_num = data.get("page_num", -1)

    if not pdf_path:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.pdfPathEmpty', lang)}), 400

    if page_num < 0:
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.pageNumInvalid', lang)}), 400

    if not os.path.isfile(pdf_path):
        lang = i18n.get_lang(request)
        return jsonify({"success": False, "error": i18n.get('errors.pdfFileNotFound', lang).format(path=pdf_path)}), 404

    try:
        img_base64 = pdf_processor.pdf_page_to_image(pdf_path, page_num)
        if not img_base64:
            lang = i18n.get_lang(request)
            return jsonify({"success": False, "error": i18n.get('errors.pageConversionFailed', lang).format(page=page_num + 1)}), 500

        return jsonify({
            "success": True,
            "pdf_path": pdf_path,
            "page_num": page_num,
            "image_base64": img_base64,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# PDF 预览路由
# =============================================================================

@app.route('/api/page_preview', methods=['GET'])
def page_preview():
    """PDF page preview with optional keyword highlighting.

    Query params:
        doc:   PDF filename
        page:  page number (1-based)
        query: optional, comma-separated keywords to highlight
        mode:  "image" (default) or "text"

    Returns JSON:
        {doc, page, total_pages, image_base64, text, highlights: [{keyword, line, context}]}
    """
    if not config.PDF_PREVIEW_ENABLED:
        return jsonify({"error": "PDF preview not enabled"}), 404

    doc = request.args.get("doc", "")
    page = int(request.args.get("page", 1))
    query = request.args.get("query", "")

    pdf_path = _resolve_pdf_path(doc)
    if not pdf_path:
        return jsonify({"error": f"Document not found: {doc}"}), 404

    try:
        markdown_text, img_base64, _use_vision = pdf_processor.extract_page_content(pdf_path, page - 1)
    except Exception as e:
        return jsonify({"error": f"Failed to extract page: {e}"}), 500

    # Fallback: extract_page_content may not return an image in text mode
    if not img_base64:
        try:
            img_base64 = pdf_processor.pdf_page_to_image(pdf_path, page - 1)
        except Exception:
            pass

    text = markdown_text or ""

    total_pages = 0
    try:
        info = pdf_processor.get_pdf_info(pdf_path)
        if info:
            total_pages = info.get('num_pages', 0)
    except Exception:
        pass

    highlights = []
    if query:
        keywords = [k.strip() for k in query.split(",") if k.strip()]
        text_lines = text.split("\n")
        for kw in keywords:
            for line_idx, line in enumerate(text_lines):
                if kw in line:
                    highlights.append({
                        "keyword": kw,
                        "line": line_idx + 1,
                        "context": line.strip()[:300],
                    })

    return jsonify({
        "success": True,
        "data": {
            "doc": doc,
            "page": page,
            "total_pages": total_pages,
            "image_base64": f"data:image/jpeg;base64,{img_base64}" if img_base64 else "",
            "text": text,
            "highlights": highlights[:20],
        }
    })


@app.route('/api/page_context', methods=['POST'])
def page_context():
    """Batch read page context for agent verification.

    Request body:
        {results: [{doc, page, query}, ...]}

    Returns:
        {contexts: [{doc, page, text, matches: [{line, text}]}]}
    """
    if not config.PDF_PREVIEW_ENABLED:
        return jsonify({"error": "PDF preview not enabled"}), 404

    data = request.get_json(silent=True) or {}
    contexts = []

    for item in data.get("results", [])[:20]:  # max 20 per request
        doc = item.get("doc", "")
        page = item.get("page", 1)
        query = item.get("query", "")

        pdf_path = _resolve_pdf_path(doc)
        if not pdf_path:
            contexts.append({"doc": doc, "page": page, "error": "not found"})
            continue

        try:
            markdown_text, _img_base64, _use_vision = pdf_processor.extract_page_content(pdf_path, page - 1)
            text = markdown_text or ""
        except Exception as e:
            contexts.append({"doc": doc, "page": page, "error": str(e)})
            continue

        matches = []
        if query and text:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if query in line:
                    matches.append({"line": i + 1, "text": line.strip()[:300]})

        contexts.append({
            "doc": doc,
            "page": page,
            "text": text[:3000],
            "matches": matches[:10],
        })

    return jsonify({"success": True, "data": {"contexts": contexts}})


@app.route('/api/literature/list', methods=['GET'])
def api_literature_list():
    """
    分页查询文献注册表，返回所有已索引PDF的标题、作者、摘要
    参数: page(默认1), limit(默认50)
    """
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        result = literature_indexer.query_registry(page=page, limit=limit)
        pdf_dir = config.PDF_FOLDER if hasattr(config, 'PDF_FOLDER') else os.path.join(config.DIALOGUE_DATA_DIR, '..', 'PDF_TARGET')
        for entry in result.get('entries', []):
            entry['pdf_path'] = os.path.join(pdf_dir, entry.get('current_filename', '')).replace('\\', '/')
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/literature/detail/<unique_id>', methods=['GET'])
def api_literature_detail(unique_id):
    """
    查询单篇文献完整详情（含摘要、创新点）
    """
    try:
        detail = literature_indexer.get_detail(unique_id)
        if detail is None:
            lang = i18n.get_lang(request)
            return jsonify({"success": False, "error": i18n.get('errors.recordNotFound', lang)}), 404
        pdf_dir = config.PDF_FOLDER if hasattr(config, 'PDF_FOLDER') else os.path.join(config.DIALOGUE_DATA_DIR, '..', 'PDF_TARGET')
        detail['pdf_path'] = os.path.join(pdf_dir, detail.get('current_filename', '')).replace('\\', '/')
        return jsonify({"success": True, "entry": detail})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Agent tool-use API (Phase 1)
# =============================================================================

@app.route('/api/chat/agent', methods=['POST'])
def chat_agent():
    """Agent tool-use 对话端点 — SSE 流式响应"""
    if _agent_executor is None:
        return jsonify({"type": "error", "reply": "Agent 模式未启用"}), 400

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    history = data.get('history', [])

    if not user_message:
        return jsonify({"type": "error", "reply": "消息不能为空"}), 400

    # Build messages from history
    messages = []
    if history:
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            api_role = "assistant" if role == "ai" else "user"
            msg = {"role": api_role, "content": content}
            if role == "ai" and m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    # Create per-session executor with spawn_agent
    session_executor = _make_session_executor()

    # Create ask_user queue for this session
    ask_queue = queue_module.Queue()
    _agent_ask_queues[session_id] = ask_queue

    # Create LLM client
    _talk_extra = config.get_extra_body("TALK")
    agent_llm = LLMClient(
        api_key=config.TALK_API_KEY,
        api_url=config.TALK_API_URL,
        extra_body=_talk_extra,
    )

    # Create AgentLoop
    loop = AgentLoop(
        llm=agent_llm,
        executor=session_executor,
        model=config.MODEL_NAME_TALK,
        max_turns=config.AGENT_MAX_TURNS,
        extra_body=_talk_extra,
    )

    # Event queue for SSE streaming
    event_queue = []
    loop_done = threading.Event()
    result_container = {}

    def event_callback(event: dict):
        event_queue.append(event)

    def run_loop():
        try:
            result = loop.run(
                messages=messages,
                event_callback=event_callback,
                ask_user_queue=ask_queue,
            )
            result_container["result"] = result
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_container["error"] = str(e)
        finally:
            loop_done.set()

    # Run agent in daemon thread
    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    def agent_sse_stream():
        idx = 0
        while not loop_done.is_set() or idx < len(event_queue):
            while idx < len(event_queue):
                event = event_queue[idx]
                idx += 1
                if event.get("type") in ("done", "error"):
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if not loop_done.is_set() and idx >= len(event_queue):
                import time
                time.sleep(0.05)

        # Cleanup
        _agent_ask_queues.pop(session_id, None)

        # Emit final event
        result = result_container.get("result", {})
        if result.get("error"):
            error_event = {"type": "agent_error", "message": result["error"]}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'agent_done'}, ensure_ascii=False)}\n\n"

    return Response(
        agent_sse_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route('/api/chat/agent/respond', methods=['POST'])
def chat_agent_respond():
    """用户响应 ask_user 问题"""
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', 'default')
    answer = data.get('answer', '').strip()

    if not answer:
        return jsonify({"type": "error", "reply": "回答不能为空"}), 400

    ask_queue = _agent_ask_queues.get(session_id)
    if ask_queue is None:
        return jsonify({"type": "error", "reply": "没有等待中的问题，或会话已过期"}), 404

    ask_queue.put(answer)
    return jsonify({"type": "ok", "reply": "回答已接收"})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SDL Agent - AI-driven lab automation')
    parser.add_argument('--default-language', choices=['zh', 'en'], default='zh',
                        help='Default language for the UI (default: zh)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    args = parser.parse_args()

    from utils.i18n import init_i18n
    init_i18n(default_lang=args.default_language)
    print(f"[i18n] Default language: {args.default_language}")

    # ---- Initialize Agent Engine (Phase 1) ----
    if config.AGENT_ENABLED:
        print("[Agent] Initializing agent toolkit...")
        _agent_executor = create_main_executor()
        _agent_orchestrator = AgentOrchestrator(executor=_agent_executor)
        print(f"[Agent]   Tools: {len(_agent_executor.names)} registered")
        print(f"[Agent]   Templates: {_agent_orchestrator.list_templates()}")
    else:
        _agent_executor = None
        _agent_orchestrator = None
        print("[Agent] Agent mode disabled (AGENT_ENABLED=false)")

    print("服务即将启动...")
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=args.port, threaded=True)