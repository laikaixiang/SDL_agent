"""
Flask应用入口 - 简洁的Web服务入口

职责：
- Flask应用初始化和配置
- 路由定义和请求处理
- 响应格式化和错误处理
- 核心业务逻辑通过core模块调用
"""

from flask import Flask, request, jsonify, render_template, Response, session, send_from_directory
import threading
import os
import json
import queue
import uuid
import asyncio
import atexit
import signal
import sys
import webbrowser
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
from core.extract_manager import PDFProcessor, ExtractionEngine
from extract.embedding_service import create_embedding_service
from extract.vector_store import ChromaVectorStore
from extract.semantic_search import SemanticSearch
from utils import CSVWriter
from prompts.api import prompts_bp

# 初始化Flask应用，static 文件夹已移入 templates/static
app = Flask(__name__, static_folder='templates/static', static_url_path='/static')
app.secret_key = os.urandom(24)  # 用于session管理
app.register_blueprint(prompts_bp)

# 初始化核心组件
config = Config()

# 初始化 PromptManager（全局单例，各模块通过 create_prompt_manager() 获取）
from prompts import create_prompt_manager as _init_prompt_manager
_init_prompt_manager()
llm_client = LLMClient()
pdf_processor = PDFProcessor()
field_inference = FieldInference()
algorithm_parser = AlgorithmParser(llm_client)    # 算法解析器
hardware_controller = HardwareController()
task_manager = TaskManager()

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

# =============================================================================
# 会话管理系统
# =============================================================================

# 全局会话时间戳（应用启动时创建）
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建会话专属文件夹
SESSION_BASE_PATH = os.path.join(config.DIALOGUE_DATA_DIR, SESSION_TIMESTAMP)
os.makedirs(os.path.join(SESSION_BASE_PATH, "extract"), exist_ok=True)
os.makedirs(os.path.join(SESSION_BASE_PATH, "temporal"), exist_ok=True)
os.makedirs(os.path.join(SESSION_BASE_PATH, "results"), exist_ok=True)
os.makedirs(os.path.join(SESSION_BASE_PATH, "experiment_designs"), exist_ok=True)

print(f"[会话管理] 应用启动，会话时间戳: {SESSION_TIMESTAMP}")
print(f"[会话管理] 数据保存路径: {SESSION_BASE_PATH}")

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

def _generate_title(messages):
    """取前2条用户消息调用 LLM 生成会话标题。"""
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    if len(user_msgs) < 2:
        return None
    lines = "\n".join(f"{i+1}. {user_msgs[i]}" for i in range(min(3, len(user_msgs))))
    from prompts import create_prompt_manager
    pm = create_prompt_manager()
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
    """扫描当前会话子目录，返回 outputs 字典。"""
    outputs = {}
    for subdir in ["extract", "temporal", "results", "experiment_designs"]:
        dir_path = os.path.join(SESSION_BASE_PATH, subdir)
        if os.path.isdir(dir_path):
            files = sorted(os.listdir(dir_path))
            outputs[subdir] = files
        else:
            outputs[subdir] = []
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

    Returns:
        str: 完整路径
    """
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
extraction_engine = ExtractionEngine(task_manager, session_path=SESSION_BASE_PATH)
csv_writer = CSVWriter(session_path=SESSION_BASE_PATH)
software_manager = SoftwareManager(
    temporal_dir=get_session_path("temporal"),
    results_dir=get_session_path("results")
)


def open_browser():
    """打开浏览器 — 默认打开新版 Vue 前端"""
    webbrowser.open("http://127.0.0.1:5000/v2")


@app.route('/')
def home():
    """
    主页路由
    返回主界面模板
    """
    return render_template('index.html')


@app.route('/extraction_mode')
def extraction_mode_page():
    """
    提取模式设置页面
    """
    return render_template('extraction_mode.html')


# ── V2 新版前端 (Vue SPA) ──
@app.route('/v2')
@app.route('/v2/')
@app.route('/v2/<path:path>')
def serve_v2_frontend(path: str = None):
    """新版 Vue SPA 入口 — 所有 /v2/* 路由返回同一个 index.html"""
    return send_from_directory('frontend/dist', 'index.html')


@app.route('/v2-static/<path:filename>')
def serve_v2_static(filename: str):
    """新版静态资源 — JS/CSS/图片等"""
    return send_from_directory('frontend/dist', filename)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    文件上传路由
    处理PDF文件上传，保存到配置的PDF文件夹
    """
    if 'files' not in request.files:
        return jsonify({'error': '没有收到文件'}), 400

    files = request.files.getlist('files')

    # 确保PDF文件夹存在
    os.makedirs(config.PDF_FOLDER, exist_ok=True)
    saved_files = []

    for file in files:
        if file.filename.lower().endswith('.pdf'):
            path = os.path.join(config.PDF_FOLDER, file.filename)
            file.save(path)
            saved_files.append(file.filename)

    return jsonify({'status': 'success', 'saved': saved_files})


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
        return jsonify({"status": "rejected", "reason": "硬件正在执行中，无法中断"})
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
        'reply': "指令确认！正在调度解析引擎，实时进度见下方..."
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
    # 检查是否有任务正在运行
    if task_manager.task_running:
        return jsonify({
            'type': 'system',
            'reply': "⚠️ 当前已有一个提取任务正在运行。"
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
            'reply': f"已启动 FAPbI3 钝化剂解析..."
        })

    # 场景2：自定义输入，去LLM询问字段
    else:
        success, fields = extraction_engine.infer_fields(task_desc, history)

        if not success:
            # 若失败，返回错误
            return jsonify({
                'type': 'system',
                'reply': f"❌ **动态字段推断失败**\n\n系统已重试3次但均未成功。\n**底层原因**：{fields}\n\n建议您重新发送指令，或检查 API 网络状态。"
            })

        confirm_msg = f"我分析了你的需求，为了完美完成【{task_desc}】的提取，我为你规划了以下输出表格列名：\n\n`{', '.join(fields)}`\n\n请问是否确认使用这些字段进行解析？"

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
            'reply': '请描述你的需求，例如：\n'
                     '  • "设计一个旋涂实验，转速3000rpm"\n'
                     '  • "移动到位置A并测量光谱"\n'
                     '  • "设置加热台温度为150度"'
        })

    # 实验设计模式
    if mode == "design":
        return jsonify({
            'type': 'experiment_design_mode',
            'command': cmd_text,
            'reply': f'🔬 **实验设计模式**\n\n'
                     f'我将使用 AI 自主规划实验流程来完成你的需求：\n'
                     f'"{cmd_text}"\n\n'
                     f'AI 将自动选择合适的工具和参数，规划完整的实验步骤。\n'
                     f'实验流程规划完成后，我会推送给你确认。'
        })

    # 单步控制模式
    else:
        success, tool_calls = hardware_controller.agent.parse_complex_command(cmd_text)

        if not success or not tool_calls:
            return jsonify({
                'type': 'system',
                'reply': f'❌ 硬件指令解析失败\n\n'
                         f'无法理解命令："{cmd_text}"\n\n'
                         f'请检查指令格式，或使用"实验设计：<需求描述>"让 AI 自动规划。'
            })

        # 生成确认信息
        confirmation_msg = hardware_controller.ask_for_experiment_confirmation(tool_calls)
        confirmation_msg = f'⚙️ **单步控制模式**\n\n' + confirmation_msg

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
        return jsonify({
            'type': 'system',
            'reply': "❌ 硬件指令解析失败，请检查指令格式"
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
    tool_calls = data.get('tool_calls', [])
    if not tool_calls:
        return jsonify({'status': 'error', 'reply': '没有可执行的硬件操作'})

    try:
        success, result = hardware_controller.execute_tool_calls(tool_calls)
        if success:
            return jsonify({'status': 'success', 'reply': '所有硬件操作已成功执行', 'result': result})
        else:
            msg = result.get('message', '') if isinstance(result, dict) else str(result)
            return jsonify({'status': 'error', 'reply': f'部分操作失败: {msg}', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'reply': f'硬件执行异常: {str(e)}'})


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

        return jsonify({
            'type': 'data_analysis_selector',
            'algorithms': available_algorithms,
            'csv_files': csv_files,
            'reply': '📊 **数据分析模式**\n\n'
                     '请选择要使用的算法和目标CSV文件：\n\n'
                     '**可用算法：**\n' +
                     '\n'.join([f'  • {algo["name"]}: {algo["description"]}' for algo in available_algorithms]) +
                     '\n\n**可用数据文件：**\n' +
                     ('\n'.join([f'  • {f}' for f in csv_files]) if csv_files else '  （暂无CSV文件）') +
                     '\n\n💡 你也可以直接输入：\n'
                     '  "数据分析：算法名称 文件路径"\n'
                     '  例如："数据分析：data_statistics temporal/extraction.csv"'
        })

    # 场景2和3：解析算法名称和CSV路径
    parts = content.split(maxsplit=1)
    algorithm_name = parts[0] if parts else ""
    csv_path = parts[1] if len(parts) > 1 else os.path.join(get_session_path("temporal"), "extraction.csv")

    if not algorithm_name:
        return jsonify({
            'type': 'system',
            'reply': f'❌ 请指定要使用的算法名称，例如：\n"数据分析：data_statistics {os.path.join(get_session_path("temporal"), "extraction.csv")}"'
        })

    # 检查算法是否存在
    available_algorithms = software_manager.list_algorithms()
    algorithm_exists = any(algo['name'] == algorithm_name for algo in available_algorithms)

    if not algorithm_exists:
        # 算法不存在，询问用户是否需要生成
        return jsonify({
            'type': 'algorithm_not_found',
            'algorithm_name': algorithm_name,
            'reply': f'❌ 算法 "{algorithm_name}" 不存在。\n\n'
                     f'**当前可用的算法：**\n' +
                     '\n'.join([f'  • {algo["name"]}: {algo["description"]}' for algo in available_algorithms]) +
                     f'\n\n💡 是否需要生成新算法 "{algorithm_name}"？\n'
                     f'请使用：生成算法：<算法描述>'
        })

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        return jsonify({
            'type': 'system',
            'reply': f'❌ 文件 "{csv_path}" 不存在。\n\n'
                     f'请检查文件路径是否正确，或使用"数据分析"命令查看可用文件。'
        })

    # 算法和文件都存在，执行分析
    if task_manager.task_running:
        return jsonify({'type': 'system', 'reply': '⚠️ 当前已有任务正在运行，请等待完成后再试。'})

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
        'reply': f'✅ 正在使用算法 **{algorithm_name}** 分析文件 `{csv_path}`\n\n实时进度见下方...'
    })


def handle_data_analysis_execute(data: dict) -> Response:
    """
    执行用户确认的数据分析任务

    Args:
        data: 请求数据，包含 algorithm_name 和 csv_path

    Returns:
        JSON响应
    """
    algorithm_name = data.get('algorithm_name', '').strip()
    csv_path = data.get('csv_path', os.path.join(get_session_path("temporal"), "extraction.csv")).strip()

    if not algorithm_name:
        return jsonify({'status': 'error', 'reply': '缺少算法名称'})

    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'reply': f'文件 "{csv_path}" 不存在'})

    if task_manager.task_running:
        return jsonify({'status': 'error', 'reply': '当前已有任务正在运行'})

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
        'reply': f'✅ 正在使用算法 **{algorithm_name}** 分析文件 `{csv_path}`\n\n实时进度见下方...'
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
    if task_manager.task_running:
        return jsonify({'type': 'system', 'reply': '⚠️ 当前已有任务正在运行，请等待完成后再试。'})

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
        'reply': f'正在分析 `{csv_path}`，实时进度见下方...'
    })


def handle_normal_chat(user_message: str, history: list = None) -> Response:
    """
    处理普通聊天

    Args:
        user_message: 用户消息
        history: 前端传来的对话历史 [{role, content, timestamp, mode, ...}]

    Returns:
        流式响应（自适应：支持流式则使用流式，否则使用非流式模拟）
    """
    return adaptive_handler.generate_response(user_message, history=history)


def handle_generate_algorithm(user_message: str) -> Response:
    """
    处理算法生成请求

    用户消息格式："生成算法：<算法描述>"

    Args:
        user_message: 用户消息

    Returns:
        JSON响应
    """
    description = user_message.replace("生成算法：", "").strip()

    if not description:
        return jsonify({
            'type': 'system',
            'reply': '请描述你需要的算法功能，例如：\n"对数值列表做移动平均，窗口大小可配置"'
        })

    try:
        result = software_manager.generate_algorithm(description)

        if result.get("success"):
            reply = f"✅ 算法生成成功！\n\n"
            reply += f"算法名称: {result['name']}\n"
            reply += f"文件路径: {result['filepath']}\n\n"
            reply += result.get('message', '')
            reply += f"\n\n你现在可以在数据分析模式中使用这个算法了。"
        else:
            reply = f"❌ 算法生成失败\n\n{result.get('message', '未知错误')}"

        return jsonify({'type': 'system', 'reply': reply})

    except Exception as e:
        return jsonify({
            'type': 'system',
            'reply': f'❌ 算法生成异常: {str(e)}'
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
    return jsonify({
        "mode": config.EXTRACTION_MODE,
        "available_modes": {
            "vision": "纯视觉模式（准确但贵）",
            "text": "纯文本模式（快速便宜）",
            "hybrid": "混合模式（推荐）"
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
        return jsonify({
            'success': False,
            'message': f'无效的模式，请选择: {", ".join(valid_modes)}'
        }), 400

    config.EXTRACTION_MODE = mode

    mode_names = {
        'vision': '纯视觉模式',
        'text': '纯文本模式',
        'hybrid': '混合模式'
    }

    return jsonify({
        'success': True,
        'mode': mode,
        'message': f'已切换到 {mode_names[mode]}'
    })


@app.route('/api/streaming_recheck', methods=['POST'])
def streaming_recheck():
    """
    强制重新检测流式支持
    """
    result = adaptive_handler.force_recheck()
    return jsonify({
        "supports_streaming": result,
        "message": "流式支持已重新检测" if result else "API不支持流式响应"
    })


@app.errorhandler(404)
def not_found(error):
    """
    404错误处理
    """
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    """
    500错误处理
    """
    return jsonify({'error': '服务器内部错误'}), 500


# =============================================================================
# 实验设计路由（基于 PydanticAI Agent 原生 tool-use）
# =============================================================================

@app.route('/api/experiment_chat', methods=['POST'])
def experiment_chat():
    """
    实验设计对话 - 使用自然语言生成实验设计JSON

    直接生成统一格式的实验设计JSON，打印到控制台并推送到前端

    返回格式：
    {
        "type": "experiment_design",
        "experiment_json": {...},  # 统一格式的实验设计JSON
        "visual_data": {...},      # 前端可视化格式
        "reply": "AI的解释说明"
    }
    """
    data = request.json
    session_id = data.get('session_id', 'default')
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'type': 'error', 'reply': '消息不能为空'})

    # 使用ExperimentDesignAgent生成JSON
    from core.field_inference import ExperimentDesignAgent
    from experiment.format import ExperimentFormatConverter

    agent = ExperimentDesignAgent()
    converter = ExperimentFormatConverter()

    print(f"\n{'='*60}")
    print(f"[实验设计] 开始生成实验方案")
    print(f"[实验设计] 用户需求: {user_message}")
    print(f"{'='*60}\n")

    success, result = agent.parse_experiment_design(user_message)

    if success:
        # 添加时间戳
        import datetime
        result['created_at'] = datetime.datetime.now().isoformat()

        # 打印生成的JSON到控制台
        print(f"\n{'='*60}")
        print(f"[实验设计] ✅ 生成成功")
        print(f"[实验设计] 实验名称: {result.get('experiment_name', '未命名实验')}")
        print(f"[实验设计] 步骤数量: {len(result.get('steps', []))}")
        print(f"\n[实验设计] 完整JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{'='*60}\n")

        # 转换为前端可视化格式
        visual_data = converter.json_to_visual(result)

        print(f"[实验设计] 已转换为前端可视化格式")
        print(f"[实验设计] 节点数量: {len(visual_data.get('nodes', []))}")
        print(f"[实验设计] 边数量: {len(visual_data.get('edges', []))}\n")

        return jsonify({
            'type': 'experiment_design',
            'experiment_json': result,
            'visual_data': visual_data,
            'reply': f"✅ 已生成实验设计方案：{result.get('experiment_name', '未命名实验')}\n\n{result.get('description', '')}\n\n共 {len(result.get('steps', []))} 个步骤，已推送到实验流程画布。"
        })
    else:
        print(f"\n{'='*60}")
        print(f"[实验设计] ❌ 生成失败")
        print(f"[实验设计] 错误信息: {result}")
        print(f"{'='*60}\n")

        return jsonify({
            'type': 'error',
            'reply': f"❌ 实验设计生成失败：{result}"
        })


@app.route('/api/experiment_upload', methods=['POST'])
def experiment_upload():
    """
    实验设计模式的 PDF 上传（方案2暂不支持PDF读取）

    TODO: 方案2不支持交互式PDF读取，如需此功能请使用方案1（PydanticAI）
    """
    session_id = request.form.get('session_id', 'default')
    if 'file' not in request.files:
        return jsonify({'error': '没有收到文件'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': '仅支持 PDF 文件'}), 400

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

    return jsonify({'status': 'success', 'message': '方案2暂不支持交互式确认'})


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

    if not algorithm_name:
        return jsonify({'success': False, 'message': '缺少 algorithm 字段'}), 400
    if input_data is None:
        return jsonify({'success': False, 'message': '缺少 data 字段'}), 400

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
        return jsonify({'success': False, 'message': '缺少 algorithm 字段'}), 400

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
        return jsonify({'success': False, 'message': '缺少 description 字段'}), 400

    result = software_manager.generate_algorithm(description)
    return jsonify(result)


@app.route('/api/software/reload', methods=['POST'])
def software_reload():
    """
    重新扫描并注册算法（生成新算法后调用，使其立即可用）
    """
    algorithms = software_manager.reload_algorithms()
    return jsonify({
        'success'   : True,
        'count'     : len(algorithms),
        'algorithms': algorithms,
        'message'   : f'已重新加载，共注册 {len(algorithms)} 个算法',
    })


# =============================================================================
# 新增：算法交互式选择路由
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
    """返回当前会话最近使用的CSV文件列表"""
    import glob
    import time

    files = []

    # 扫描当前会话的 temporal/ 和 extract/ 目录
    session_temporal = get_session_path("temporal")
    session_extract = get_session_path("extract")

    for pattern in [
        os.path.join(session_temporal, "*.csv"),
        os.path.join(session_extract, "*.csv")
    ]:
        for filepath in glob.glob(pattern):
            try:
                stat = os.stat(filepath)
                files.append({
                    'path': filepath.replace('\\', '/'),
                    'name': os.path.basename(filepath),
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'modified_str': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))
                })
            except Exception:
                continue

    # 按修改时间倒序排序
    files.sort(key=lambda x: x['modified'], reverse=True)

    # 只返回最近10个
    return jsonify({
        "success": True,
        "files": files[:10]
    })


@app.route('/api/browse_csv', methods=['GET'])
def browse_csv():
    """列出可用的输入 CSV 文件：当前会话 extract/ + const_data/extract/"""
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
    dirs.append({'path': session_results.replace('\\', '/'), 'label': '当前会话 results（默认）', 'is_default': True})
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
        return jsonify({'success': False, 'message': '缺少算法描述'}), 400

    # 调用软件管理器生成算法
    result = software_manager.generate_algorithm(description)

    # 如果生成成功，自动重新加载算法列表
    if result.get('success'):
        software_manager.reload_algorithms()

    return jsonify(result)


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

    if not algo_name:
        return jsonify({
            "success": False,
            "message": "缺少算法名称"
        }), 400

    if not file_path:
        return jsonify({
            "success": False,
            "message": "缺少文件路径"
        }), 400

    # 验证文件存在
    if not os.path.exists(file_path):
        return jsonify({
            "success": False,
            "message": f"文件不存在: {file_path}"
        }), 404

    # 检查是否有任务正在运行
    if task_manager.task_running:
        return jsonify({
            "success": False,
            "message": "当前已有任务正在运行，请等待完成后再试"
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
        "message": f"正在执行算法 {algo_name}..."
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
    experiment_name = data.get('experiment_name', '未命名实验')
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
            'message': f'实验设计已保存到 {filepath}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'保存失败: {str(e)}'
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
    experiment_name = data.get('experiment_name', '未命名实验')
    steps = data.get('steps', [])

    if not steps:
        return jsonify({'type': 'error', 'reply': '实验设计中没有步骤'}), 400

    if task_manager.task_running:
        return jsonify({'type': 'error', 'reply': '当前已有任务正在运行，请等待完成后再试'}), 409

    while not task_manager.is_queue_empty():
        task_manager.get_task_message()

    task_id = task_manager.generate_task_id()
    task_manager.current_task_id = task_id
    task_manager.task_running = True

    def _run():
        try:
            executor = ExperimentExecutor(software_manager=software_manager)
            total = len(steps)

            def on_progress(step_num, status, message):
                msg_type = "info" if status in ("running", "completed") else "error"
                task_manager.put_task_message({"type": msg_type, "data": message})

            task_manager.put_task_message({"type": "info", "data": f"开始执行实验: {experiment_name}，共 {total} 步"})
            result = executor.execute_plan(data, progress_callback=on_progress)

            if result["success"]:
                task_manager.put_task_message({"type": "complete", "data": {"message": f"✅ 实验 {experiment_name} 执行完成！"}})
            else:
                err = result.get("error") or "部分步骤失败"
                task_manager.put_task_message({"type": "complete", "data": {"error": err}})
        except Exception as e:
            task_manager.put_task_message({"type": "complete", "data": {"error": f"实验执行异常: {str(e)}"}})
        finally:
            task_manager.task_running = False

    threading.Thread(target=_run, daemon=True).start()

    return jsonify({
        'type': 'task_trigger',
        'reply': f'🚀 开始执行实验设计: {experiment_name}\n共 {len(steps)} 个步骤，实时进度见下方...'
    })


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
    json_data = data.get('json_data')
    filepath = data.get('filepath', '').strip()

    if not json_data:
        return jsonify({
            'success': False,
            'message': '缺少JSON数据'
        }), 400

    if not filepath:
        return jsonify({
            'success': False,
            'message': '缺少文件路径'
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
            'message': f'实验设计已导出到 {filepath}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'导出失败: {str(e)}'
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
        return jsonify({
            'success': False,
            'message': '缺少实验JSON数据'
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
        return jsonify({
            'success': False,
            'message': f'编译失败: {str(e)}'
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
        return jsonify({
            'success': False,
            'message': '缺少实验JSON数据'
        }), 400

    try:
        from experiment.compiler import ExperimentCompiler
        compiler = ExperimentCompiler()
        result = compiler.compile_and_run(experiment_json)

        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'编译或执行失败: {str(e)}'
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
        return jsonify({"success": False, "message": "请求体为空"}), 400

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
    if not title and len(messages) >= 4:
        title = _generate_title(messages)
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

    _update_session_index({
        "timestamp": SESSION_TIMESTAMP,
        "started_at": session_info["started_at"],
        "saved_at": session_info["saved_at"],
        "message_count": len(messages),
        "title": title,
        "path": SESSION_TIMESTAMP
    })

    return jsonify({"success": True, "saved_count": len(messages)})


@app.route('/api/history/sessions', methods=['GET'])
def history_sessions():
    """返回所有历史会话的索引列表。"""
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"sessions": []}
    return jsonify(data)


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
        return jsonify({"success": False, "error": "语义搜索服务未初始化，请检查 EMBEDDING_API_KEY 配置"}), 503

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    top_k = data.get("top_k", 10)

    if not query:
        return jsonify({"success": False, "error": "query 不能为空"}), 400

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
        return jsonify({"success": False, "error": "pdf_path 不能为空"}), 400

    if page_num < 0:
        return jsonify({"success": False, "error": "page_num 必须 >= 0"}), 400

    if not os.path.isfile(pdf_path):
        return jsonify({"success": False, "error": f"PDF 文件不存在: {pdf_path}"}), 404

    try:
        img_base64 = pdf_processor.pdf_page_to_image(pdf_path, page_num)
        if not img_base64:
            return jsonify({"success": False, "error": f"无法转换第 {page_num + 1} 页"}), 500

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
            total_pages = info.get('total_pages', 0)
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


if __name__ == '__main__':
    print("服务即将启动...")
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000, threaded=True)