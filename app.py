from flask import Flask, request, jsonify, render_template, Response
import requests
import threading
import os
import base64
import json
import time
import csv
import io
import queue
import webbrowser
from threading import Timer
import asyncio
import re
import sys

# ==========================================
# 🔧 路径配置与依赖注入
# ==========================================
# 将 hardware 文件夹加入系统路径，确保硬件模块间相互导入正常 (如 tool.py 导入 agent_client.py)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hardware'))

import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field, create_model
from typing import Literal, Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI  # 🌟 新增：用于底层的原生多模态视觉 API 调用

# 引入硬件控制模块中的具体执行函数
from hardware.tools import execute_spin_coating, execute_set_temperature, execute_move_robot_arm

app = Flask(__name__)

# ==========================================
# ⚙️ 核心配置参数
# ==========================================
SILICONFLOW_API_KEY = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"  # ⚠️ 请填入真实的API KEY
PDF_FOLDER = r"test"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

# 模型路径定义
Base_URL = "https://api.siliconflow.cn/v1"
CHAT_API_URL = f"{Base_URL}/chat/completions"

# 🌟 全局模型与客户端实例
# 1. 提供给 Pydantic-AI 的 Provider
custom_provider = OpenAIProvider(base_url=Base_URL, api_key=SILICONFLOW_API_KEY)
ai_model = OpenAIChatModel(MODEL_NAME, provider=custom_provider)

# 2. 原生 AsyncOpenAI 客户端（解决 'OpenAIProvider' 没有 'chat' 属性的问题）
async_openai_client = AsyncOpenAI(api_key=SILICONFLOW_API_KEY, base_url=Base_URL)

task_queue = queue.Queue()
task_running = False
cancel_requested = False


# ==========================================
# 🛠️ 辅助函数
# ==========================================
def pdf_page_to_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ==========================================
# 🧠 核心：异步动态文献提取 (基于 Pydantic-AI)
# ==========================================
async def async_process_pdf_library(task_description: str, fields: list):
    global task_running, cancel_requested

    # 1. 动态生成 Pydantic Schema
    fields_def = {f: (str, Field(description=f"提取并高度凝练：{f}")) for f in fields}
    DynamicRecord = create_model('DynamicRecord', **fields_def)

    save_dir = "extract"
    os.makedirs(save_dir, exist_ok=True)
    all_extracted_data = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)

    for file_idx, filename in enumerate(pdf_files):
        if cancel_requested: break

        pdf_path = os.path.join(PDF_FOLDER, filename)
        doc_id = os.path.splitext(filename)[0]
        try:
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            task_queue.put({"type": "progress", "message": f"正在处理第 {file_idx + 1}/{total_files} 篇: {filename}"})

            for page_num in range(num_pages):
                if cancel_requested: break

                img_base64 = pdf_page_to_image(pdf_path, page_num)
                task_queue.put(
                    {"type": "page_reading", "data": {"filename": filename, "page": page_num + 1, "image": img_base64}})

                # 3. 🌟 使用原生 AsyncOpenAI 客户端处理图片
                try:
                    response = await async_openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system",
                             "content": f"你是一个严谨的文献数据清洗专家。当前任务：【{task_description}】\n请务必准确提取信息，并严格输出一个 JSON 格式的数组。数组的每个元素必须包含以下字段：{fields}。如果没有找到符合的内容，请输出空数组 []。绝对不要输出 markdown 标记。"},
                            {"role": "user", "content": [
                                {"type": "text", "text": "提取这张文献页面中的信息："},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                            ]}
                        ],
                        temperature=0.1
                    )

                    # 剥离大模型可能的 markdown 残留
                    content = response.choices[0].message.content.strip()
                    clean_json = re.sub(r'```json\n|\n```|```', '', content).strip()

                    try:
                        extracted_list = json.loads(clean_json)
                    except json.JSONDecodeError:
                        extracted_list = []

                    # 4. 结合 Pydantic 动态模型进行严谨校验与清洗
                    for item in extracted_list:
                        if not isinstance(item, dict): continue

                        try:
                            # 使用前面定义的 DynamicRecord 进行校验，过滤幻觉字段
                            record = DynamicRecord(**item)
                            record_dict = record.model_dump()
                            record_dict['_source_doc'] = doc_id
                            all_extracted_data.append(record_dict)

                            task_queue.put({"type": "finding",
                                            "data": {"page": page_num + 1, "filename": filename,
                                                     "details": record_dict}})
                        except Exception as validation_err:
                            print(f"Pydantic校验跳过不合规数据: {validation_err}")

                except Exception as e:
                    print(f"该页提取异常: {e}")
                    task_queue.put({"type": "warning", "message": f"第 {page_num + 1} 页提取异常: {str(e)}"})

                time.sleep(2.0)
        except Exception as e:
            task_queue.put({"type": "error", "message": f"处理 {filename} 失败: {str(e)}"})

    # 导出 CSV
    os.makedirs("extract", exist_ok=True)
    csv_filename = os.path.join(save_dir, f"Extraction_{time.strftime('%Y%m%d-%H%M%S')}.csv")

    # 🌟 扩展：同时保存归档文件与临时文件
    temp_csv = os.path.join(save_dir, "extraction.csv")

    all_keys = list(fields) + ['_source_doc']

    for target_file in [csv_filename, temp_csv]:
        with open(target_file, 'w', newline='', encoding='utf-8') as csvfile:
            if all_extracted_data:
                writer = csv.DictWriter(csvfile, fieldnames=all_keys)
                writer.writeheader()
                for row in all_extracted_data:
                    writer.writerow({k: row.get(k, '') for k in all_keys})
            else:
                csvfile.write(",".join(fields))

    task_queue.put({"type": "complete", "csv": csv_filename, "count": len(all_extracted_data), "fields": fields})
    task_running = False


def process_pdf_library_thread(task_desc, fields):
    """桥接 Flask 同步环境与 asyncio"""
    global task_running, cancel_requested
    task_running = True
    cancel_requested = False
    asyncio.run(async_process_pdf_library(task_desc, fields))


# ==========================================
# 🌐 Flask 路由设计
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/task_stream')
def task_stream():
    def event_stream():
        while True:
            try:
                msg = task_queue.get(timeout=2)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get("type") == "complete": break
            except queue.Empty:
                if not task_running: break
                yield ": heartbeat\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/api/cancel_task', methods=['POST'])
def cancel_task():
    global cancel_requested
    cancel_requested = True
    return jsonify({"status": "stopping"})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files: return jsonify({'error': '没有收到文件'}), 400
    os.makedirs(PDF_FOLDER, exist_ok=True)
    saved_files = []
    for file in request.files.getlist('files'):
        if file.filename.lower().endswith('.pdf'):
            file.save(os.path.join(PDF_FOLDER, file.filename))
            saved_files.append(file.filename)
    return jsonify({'status': 'success', 'saved': saved_files})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    action = data.get('action', 'chat')

    # 🌟 流程：用户确认字段后，启动 Pydantic-AI 提取线程
    if action == 'start_extraction':
        task_desc, fields = data.get('task_desc'), data.get('fields')
        while not task_queue.empty(): task_queue.get()
        threading.Thread(target=process_pdf_library_thread, args=(task_desc, fields)).start()
        return jsonify({'type': 'task_trigger', 'reply': "指令确认！正在调度解析引擎，实时进度见下方..."})

    # 🌟 拦截提取指令：让大模型推理字段
    if user_message.startswith("帮我搜寻："):
        global task_running
        if task_running: return jsonify({'type': 'system', 'reply': "⚠️ 当前已有一个提取任务正在运行。"})

        task_desc = user_message.replace("帮我搜寻：", "").strip()
        if not task_desc:
            task_desc = "专门用于 FAPbI3 钙钛矿体系的钝化剂"
            fields = ["钝化剂名称", "原文原句", "作用机理"]
            while not task_queue.empty(): task_queue.get()
            threading.Thread(target=process_pdf_library_thread, args=(task_desc, fields)).start()
            return jsonify({'type': 'task_trigger', 'reply': f"检测到默认指令！已启动 FAPbI3 钝化剂解析..."})
        else:
            class FieldAnalysis(BaseModel):
                fields: list[str] = Field(description="推断出的需要提取的数据列名列表")

            field_agent = Agent(
                OpenAIChatModel("Qwen/Qwen2.5-72B-Instruct", provider=custom_provider),
                system_prompt="你是一个数据分析专家。请根据用户的任务描述，推断需要提取哪些数据列名。",
                output_type=FieldAnalysis
            )

            try:
                result = asyncio.run(field_agent.run(f"任务：{task_desc}"))
                fields = result.data.fields
            except Exception as e:
                fields = ["提取目标", "详细参数"]
                print(f"字段推断报错: {e}")

            return jsonify({'type': 'field_confirm', 'task_desc': task_desc, 'fields': fields,
                            'reply': f"为了完成提取，我为你规划了以下表头：\n`{', '.join(fields)}`\n请确认："})

    # 🌟 拦截硬件控制 (基于 Pydantic-AI 工具调用)
    if user_message.startswith("硬件控制："):
        cmd_text = user_message.replace("硬件控制：", "").strip()

        # 🌟 修正：将 hardware 模块的具体执行函数通过 list 传入 tools
        hw_agent = Agent(
            ai_model,
            system_prompt="你是一个实验室硬件控制智能体。请根据用户的指令，精准调用合适的底层工具(旋涂、温控、机械臂)完成任务。请返回简洁的中文执行汇报。",
            tools=[execute_spin_coating, execute_set_temperature, execute_move_robot_arm]
        )
        try:
            result = asyncio.run(hw_agent.run(cmd_text))
            return jsonify({'type': 'system', 'reply': f"🔧 **硬件调度结果**\n\n{result.data}"})
        except Exception as e:
            return jsonify({'type': 'system', 'reply': f"❌ 硬件调度异常: {str(e)}"})

    # 🌟 拦截软件算法 (基于 Pydantic-AI 强制结构化输出)
    if user_message.startswith("优化算法："):
        cmd_text = user_message.replace("优化算法：", "").strip()

        class AlgoDecision(BaseModel):
            action: Literal["call_existing", "generate_new"] = Field(description="选择调用已有算法还是生成新代码")
            algo_name: Optional[str] = Field(description="如果是 call_existing，指定算法名称(如 bayes_opt)")
            code: Optional[str] = Field(description="如果是 generate_new，填写完整的 Python 代码")
            reason: str = Field(description="AI决策原因说明")

        sw_agent = Agent(
            ai_model,
            system_prompt="你是一个顶尖算法工程师。你可以调用已有算法(software文件夹)，或根据需求从零编写新Python分析脚本。",
            output_type=AlgoDecision
        )

        try:
            decision = asyncio.run(sw_agent.run(cmd_text)).data

            if decision.action == "call_existing":
                return jsonify(
                    {'type': 'system', 'reply': f"⚙️ **调用预置算法** `{decision.algo_name}`\n原因：{decision.reason}"})

            elif decision.action == "generate_new":
                os.makedirs("software", exist_ok=True)
                script_path = os.path.join("software", "dynamic_generated.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(decision.code)

                run_res = subprocess.run(["python", script_path], capture_output=True, text=True)
                output_msg = run_res.stdout if run_res.returncode == 0 else run_res.stderr
                return jsonify({'type': 'system',
                                'reply': f"✨ **动态生成代码执行完毕**\n\n**决策原因**：{decision.reason}\n**执行日志**：\n```text\n{output_msg.strip()}\n```"})

        except Exception as e:
            return jsonify({'type': 'system', 'reply': f"❌ 算法路由异常: {str(e)}"})

    # 🌟 普通聊天流式输出
    def generate_chat():
        payload = {"model": "Qwen/Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": user_message}],
                   "stream": True}
        try:
            response = requests.post(CHAT_API_URL, headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                                     json=payload,
                                     stream=True, timeout=30)
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: ") and "[DONE]" not in decoded:
                        try:
                            content = json.loads(decoded[6:])['choices'][0]['delta'].get('content', '')
                            if content: yield content
                        except:
                            pass
        except Exception as e:
            yield f"\n[网络请求失败: {str(e)}]"

    return Response(generate_chat(), content_type='text/plain; charset=utf-8')


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == '__main__':
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000, threaded=True)