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


def handle_normal_chat(user_message: str) -> Response:
    """
    处理普通聊天

    Args:
        user_message: 用户消息

    Returns:
        流式响应
    """
    def generate_chat():
        headers = llm_client.get_default_headers()
        payload = {
            "model": config.MODEL_NAME_TALK,
            "messages": [{"role": "user", "content": user_message}],
            "stream": True
        }

        try:
            response = requests.post(
                config.API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except:
                            pass
        except Exception as e:
            yield f"\n[请求失败: {str(e)}]"

    return Response(generate_chat(), content_type='text/plain; charset=utf-8')


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


if __name__ == '__main__':
    print("服务即将启动...")
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000, threaded=True)