"""
Flask应用入口 - 简洁的Web服务入口

职责：
- Flask应用初始化和配置
- 路由定义和请求处理
- 响应格式化和错误处理
- 核心业务逻辑通过core模块调用
"""

from flask import Flask, request, jsonify, render_template, Response
import threading
import os
import json
import queue
import uuid
import asyncio
import webbrowser
import requests
from threading import Timer

# 导入核心模块
from core import (
    Config,
    LLMClient,
    PDFProcessor,
    FieldInference,
    HardwareController,
    TaskManager,
    ExtractionEngine,
    CSVWriter,
    ExperimentDesignAgent,
    SoftwareManager,
    AdaptiveStreamHandler,
)

# 初始化Flask应用
app = Flask(__name__)

# 初始化核心组件
config = Config()
llm_client = LLMClient()
pdf_processor = PDFProcessor()
field_inference = FieldInference()
hardware_controller = HardwareController()
task_manager = TaskManager()
extraction_engine = ExtractionEngine(task_manager)
csv_writer = CSVWriter()
experiment_agent = ExperimentDesignAgent()  # 实验设计智能体（PydanticAI 原生 tool-use）
software_manager = SoftwareManager()        # 软件算法管理器
adaptive_handler = AdaptiveStreamHandler(config, llm_client)  # 自适应流式响应处理器


def open_browser():
    """打开浏览器"""
    webbrowser.open("http://127.0.0.1:5000")


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

    # 特殊流程：用户已确认字段，正式开始提取
    if action == 'start_extraction':
        return handle_extraction_start(data)

    # 特殊流程：用户已确认硬件操作，正式执行
    if action == 'start_hardware':
        return handle_hardware_execute(data)

    # 拦截提取指令：Agentic 判断与 Schema 生成
    if user_message.startswith("帮我搜寻："):
        return handle_extraction_request(user_message)

    # 硬件控制
    if user_message.startswith("硬件控制："):
        return handle_hardware_control(user_message)

    # 自动数据分析
    if user_message.startswith("数据分析："):
        return handle_auto_analyze(user_message)

    # 算法生成
    if user_message.startswith("生成算法："):
        return handle_generate_algorithm(user_message)

    # 普通聊天流式输出
    return handle_normal_chat(user_message)


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


def handle_extraction_request(user_message: str) -> Response:
    """
    处理提取请求

    Args:
        user_message: 用户消息

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
        success, fields = extraction_engine.infer_fields(task_desc)

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


def handle_auto_analyze(user_message: str) -> Response:
    """
    处理自动数据分析请求

    用户消息格式："数据分析：<csv_path>"
    csv_path 为空时默认使用 temporal/extraction.csv

    Args:
        user_message: 用户消息

    Returns:
        JSON响应（task_trigger 类型，触发前端 SSE 监听）
    """
    if task_manager.task_running:
        return jsonify({'type': 'system', 'reply': '⚠️ 当前已有任务正在运行，请等待完成后再试。'})

    csv_path = user_message.replace("数据分析：", "").strip()
    if not csv_path:
        csv_path = "temporal/extraction.csv"

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


def handle_normal_chat(user_message: str) -> Response:
    """
    处理普通聊天

    Args:
        user_message: 用户消息

    Returns:
        流式响应（自适应：支持流式则使用流式，否则使用非流式模拟）
    """
    # 使用自适应流式处理器
    return adaptive_handler.generate_response(user_message)


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
    实验设计对话 - AI 自主选择工具规划实验流程
    """
    data = request.json
    session_id = data.get('session_id', 'default')
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'type': 'error', 'reply': '消息不能为空'})

    # 将异步 send_event 桥接到 task_manager 消息队列
    async def send_event_async(event):
        task_manager.put_task_message(event)

    # 在同步 Flask 中运行异步 Agent
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            experiment_agent.run(session_id, user_message, send_event_async)
        )
        return jsonify({'type': 'assistant_response', 'reply': result})
    except Exception as e:
        return jsonify({'type': 'error', 'reply': f'实验设计Agent错误: {str(e)}'})
    finally:
        loop.close()


@app.route('/api/experiment_upload', methods=['POST'])
def experiment_upload():
    """
    实验设计模式的 PDF 上传，上传后 AI 可通过 read_pdf 工具读取
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

    experiment_agent.set_pdf_path(session_id, path)
    return jsonify({'filename': safe_name, 'path': path})


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
    对 temporal/extraction.csv 中的数值列运行算法（提取任务完成后可直接使用）

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


if __name__ == '__main__':
    print("服务即将启动...")
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000, threaded=True)