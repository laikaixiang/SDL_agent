from flask import Flask, request, jsonify, render_template, Response
import requests
import threading
import os
import base64
import json
import time
import re
import csv
import fitz
from PIL import Image
import io
import queue
import webbrowser
from threading import Timer

import hardware_controller

app = Flask(__name__)

# ==========================================
# 配置参数
# ==========================================
SILICONFLOW_API_KEY = "sk-"
PDF_FOLDER = r"test"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

DPI = 200
REQUEST_DELAY = 3.0

task_queue = queue.Queue()
task_running = False
cancel_requested = False  # 新增：用于中断后台线程的全局标志位


def pdf_page_to_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    doc.close()
    return img_str


# LLM 动态提取字段生成器
def get_dynamic_fields_from_llm(task_desc):
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        f"你是一个文献数据抽取专家。用户希望进行以下信息提取任务：【{task_desc}】。\n"
        "请你推断为了完成这个任务，最终的数据表格需要包含哪些列名（字段）？\n"
        "必须严格输出 JSON 格式的字符串数组。例如：[\"反溶剂名称\", \"旋涂时间\", \"旋涂转速\", \"文献来源\"]\n"
        "重要：不要输出任何其他解释性文字。"
    )
    payload = {"model": "Qwen/Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.1}
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        content = resp.json()['choices'][0]['message']['content'].strip()
        clean_text = re.sub(r'```json\n|\n```|```', '', content).strip()
        fields = json.loads(clean_text)
        if isinstance(fields, list) and len(fields) > 0:
            return fields
    except Exception as e:
        print(f"动态字段生成失败: {e}")
    return ["提取目标", "相关详细参数", "文献来源"]  # Fallback


# LLM 动态生成英文文件名前缀
def get_filename_prefix(task_desc):
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    prompt = f"将以下提取任务的核心关键词翻译为简短的英文（单词之间用下划线连接），仅输出英文，不要有其他字符。任务：{task_desc}"
    payload = {"model": "Qwen/Qwen2.5-72B-Instruct", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.1}
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        return resp.json()['choices'][0]['message']['content'].strip().replace(" ", "_").lower()
    except:
        return "extraction_result"


# 核心提取主程序：支持动态字段与中断
def process_pdf_library(task_description, fields):
    global task_running, cancel_requested
    task_running = True
    cancel_requested = False

    task_queue.put({"type": "info", "message": f"🚀 提取任务启动！目标：【{task_description}】"})

    # 1. 创建 temporal 文件夹
    save_dir = "extract"
    os.makedirs(save_dir, exist_ok=True)
    prefix = get_filename_prefix(task_description)

    if not os.path.exists(PDF_FOLDER):
        task_queue.put({"type": "error", "message": f"找不到文件夹: {PDF_FOLDER}"})
        task_running = False
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)

    all_extracted_data = []  # 存储所有动态抓取的数据

    for file_idx, filename in enumerate(pdf_files):
        if cancel_requested:
            task_queue.put({"type": "info", "message": "⚠️ 接收到停止指令！正在终止并保存当前数据..."})
            break

        pdf_path = os.path.join(PDF_FOLDER, filename)
        doc_id = os.path.splitext(filename)[0]
        try:
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            task_queue.put({"type": "progress", "message": f"正在处理第 {file_idx + 1}/{total_files} 篇: {filename}"})

            for page_num in range(num_pages):
                if cancel_requested: break

                img_base64 = pdf_page_to_image(pdf_path, page_num)

                # 构建动态 JSON Keys
                fields_format = ", ".join([f'"{f}": "..."' for f in fields])

                # 向前端推送当前 AI 正在阅读的页面图像
                task_queue.put({
                    "type": "page_reading",
                    "data": {
                        "filename": filename,
                        "page": page_num + 1,
                        "image": img_base64
                    }
                })

                sys_prompt = (
                    f"你是一个专业的学术文献分析专家。你的任务是从提供的文献页面图像中提取：\n【目标】：{task_description}\n\n"
                    "请严格遵循以下规则提取并返回 JSON 数组：\n"
                    f"1. 必须包含以下字段：{json.dumps(fields, ensure_ascii=False)}\n"
                    "2. 提取要精准，如果是复合材料不可拆分，如果是复合材料中中间有+，and或者其他标示复合的字符，并且之前已经提取过，都视为一个材料，不要重复输出。复合材料需要提取其比例，如果未提取出，请在名称后标明（未说明比例）。\n"
                    "3. 如果需要提取溶剂量/浓度/转速/温度等，必须标明单位，若无法提取单位，请标明。\n"
                    "4. 不需要提取参考文献里的数据。\n"
                    "5. 必须且只能输出合法的 JSON 数组，不要包含任何 Markdown 标记。未发现目标请输出 []。\n\n"
                    f"JSON 输出范例：\n[{{{fields_format}}}]"
                )

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",
                     "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]}
                ]
                headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.1, "max_tokens": 1024}

                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    result_text = response.json()['choices'][0]['message']['content'].strip()
                    try:
                        clean_text = re.sub(r'```json\n|\n```|```', '', result_text).strip()
                        extracted_data = json.loads(clean_text)
                        if isinstance(extracted_data, list) and len(extracted_data) > 0:
                            for item in extracted_data:
                                item['_source_doc'] = doc_id  # 追加隐藏的来源字段
                                all_extracted_data.append(item)
                                task_queue.put({
                                    "type": "finding",
                                    "data": {"page": page_num + 1, "filename": filename, "details": item}
                                })
                    except json.JSONDecodeError:
                        pass
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            task_queue.put({"type": "error", "message": f"处理 {filename} 失败: {str(e)}"})

    # 2. 将结果动态写入 CSV
    csv_filename = os.path.join(save_dir, f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    if all_extracted_data:
        # 补全可能缺失的 keys
        all_keys = set(fields)
        for d in all_extracted_data: all_keys.update(d.keys())
        all_keys = list(all_keys)

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_keys)
            writer.writeheader()
            for row in all_extracted_data:
                writer.writerow(row)
    else:
        # 即使没有数据也建个空文件
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(",".join(fields))

    # 写入一个固定名字的excel表，方便后续调用
    csv_filename_temporal = "temporal/extraction.csv"
    if all_extracted_data:
        # 补全可能缺失的 keys
        all_keys = set(fields)
        for d in all_extracted_data: all_keys.update(d.keys())
        all_keys = list(all_keys)

        with open(csv_filename_temporal, 'w', newline='', encoding='utf-8') as csvfile_temporal:
            writer = csv.DictWriter(csvfile_temporal, fieldnames=all_keys)
            writer.writeheader()
            for row in all_extracted_data:
                writer.writerow(row)
    else:
        # 即使没有数据也建个空文件
        with open(csv_filename_temporal, 'w', newline='', encoding='utf-8') as csvfile_temporal:
            csvfile_temporal.write(",".join(fields))

    task_queue.put({"type": "complete", "csv": csv_filename_temporal, "count": len(all_extracted_data), "fields": fields})
    task_running = False


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': '没有收到文件'}), 400

    files = request.files.getlist('files')

    # PDF_FOLDER是前面全局定义的那个
    os.makedirs(PDF_FOLDER, exist_ok=True)  # 确保文件夹存在
    saved_files = []

    for file in files:
        if file.filename.lower().endswith('.pdf'):
            path = os.path.join(PDF_FOLDER, file.filename)
            file.save(path)
            saved_files.append(file.filename)

    return jsonify({'status': 'success', 'saved': saved_files})

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


# 🌟 新增：手动中断任务接口
@app.route('/api/cancel_task', methods=['POST'])
def cancel_task():
    global cancel_requested
    cancel_requested = True
    return jsonify({"status": "stopping"})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    action = data.get('action', 'chat')  # 用于区分普通对话还是特殊指令

    # 🌟 特殊流程：用户已确认字段，正式开始提取
    if action == 'start_extraction':
        task_desc = data.get('task_desc')
        fields = data.get('fields')
        while not task_queue.empty(): task_queue.get()
        threading.Thread(target=process_pdf_library, args=(task_desc, fields)).start()
        return jsonify({'type': 'task_trigger', 'reply': "指令确认！正在调度解析引擎，实时进度见下方..."})

    # 🌟 拦截提取指令：Agentic 判断与 Schema 生成
    if user_message.startswith("帮我搜寻："):
        global task_running
        if task_running:
            return jsonify({'type': 'system', 'reply': "⚠️ 当前已有一个提取任务正在运行。"})

        task_desc = user_message.replace("帮我搜寻：", "").strip()

        # 场景 1：如果没有任何输入，直接采用默认值并默认字段
        if not task_desc:
            # 默认字段：
            task_desc = "专门用于 FAPbI3 钙钛矿体系的钝化剂(Passivator)"
            default_fields = ["钝化剂名称", "原文原句", "作用机理", "文献来源"]
            while not task_queue.empty(): task_queue.get()
            threading.Thread(target=process_pdf_library, args=(task_desc, default_fields)).start()
            return jsonify({'type': 'task_trigger', 'reply': f"已启动 FAPbI3 钝化剂解析..."})

        # 场景 2：自定义输入，去 LLM 询问字段，并返回前端要求用户确认
        else:
            fields = get_dynamic_fields_from_llm(task_desc)
            confirm_msg = f"我分析了你的需求，为了完美完成【{task_desc}】的提取，我为你规划了以下输出表格列名：\n\n`{', '.join(fields)}`\n\n请问是否确认使用这些字段进行解析？"
            return jsonify({
                'type': 'field_confirm',
                'task_desc': task_desc,
                'fields': fields,
                'reply': confirm_msg
            })

    # 🌟 硬件控制
    if user_message.startswith("硬件控制："):
        cmd_text = user_message.replace("硬件控制：", "").strip()

        # 专门针对现有硬件接口设计的高精度 Prompt
        hw_prompt = """你是一个专业的实验室自动化硬件控制智能体。你的任务是将用户的自然语言指令，精准转换为可由下位机执行的 JSON 格式数据。

        目前系统仅支持以下三种硬件操作，请根据用户的指令匹配最合适的 action，并提取相应 params：
        
        1. 设置温度 (set_temperature)
           - 格式: {"action": "set_temperature", "params": {"target": 浮点数}}
           - 示例: 用户说"把温度设为30度"，输出 {"action": "set_temperature", "params": {"target": 30.0}}
        
        2. 移动机械臂 (move_robot_arm)
           - 格式: {"action": "move_robot_arm", "params": {"x": 浮点数, "y": 浮点数, "z": 浮点数}}
           - 示例: 用户说"机械臂移动到10, 20, 5"，输出 {"action": "move_robot_arm", "params": {"x": 10.0, "y": 20.0, "z": 5.0}}
        
        3. 执行原位旋涂实验 (do_experiment)
           - 格式: {"action": "do_experiment", "params": {"reagent": "字符串", "spin_speed": 整数, "spin_acc": 整数, "spin_dur": 整数, "volume": 整数}}
           - 参数说明: reagent(试剂名称), spin_speed(转速 rpm，最大6000), spin_acc(加速度 rpm/s，默认1000), spin_dur(旋涂时间 ms，注意：如果用户说秒，请乘以1000换算为毫秒), volume(溶液体积)
           - 示例: 用户说"用氯苯以3000转、1000加速度旋涂30秒，体积50"，输出 {"action": "do_experiment", "params": {"reagent": "氯苯", "spin_speed": 3000, "spin_acc": 1000, "spin_dur": 30000, "volume": 50}}
        
        重要规则：
        - 必须且只能输出一个合法的 JSON 对象。
        - 绝对不要输出任何多余的解释、思考过程或问候语。
        - 绝对不要包含 Markdown 标记（例如不要写 ```json ）。
        - 如果用户指令中缺失了旋涂实验(do_experiment)的某些参数，请使用常规默认值填补（如 spin_acc 默认为 1000）。"""

        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [
                {"role": "system", "content": hw_prompt},
                {"role": "user", "content": cmd_text}
            ],
            "temperature": 0.1  # 调低温度以保证JSON输出的稳定性和确定性
        }
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            llm_json = resp.json()['choices'][0]['message']['content'].strip()

            # 双重保险：清理大模型可能残留的 markdown code block 标记
            clean_json = re.sub(r'```json\n|\n```|```', '', llm_json).strip()

            # 将清理后的 JSON 喂给硬件控制器
            hw_result = hardware_controller.execute_llm_hardware_command(clean_json)

            # 优化前端 UI 反馈
            status_icon = "✅" if hw_result.get("status") == "success" else "❌"
            reply_msg = (
                f"🔧 **硬件调度执行完毕**\n\n"
                f"**JSON 指令**：\n`{clean_json}`\n\n"
                f"**执行状态**：{status_icon} {hw_result.get('status')}\n"
                f"**底层反馈**：{hw_result.get('output', hw_result.get('message', '无反馈'))}"
            )

            return jsonify({'type': 'system', 'reply': reply_msg})
        except Exception as e:
            return jsonify({'type': 'system', 'reply': f"❌ 硬件调度失败: {str(e)}"})

    # 🌟 普通聊天流式输出
    def generate_chat():
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": user_message}], "stream": True}
        try:
            response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=30)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]": break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content: yield content
                        except:
                            pass
        except Exception as e:
            yield f"\n[请求失败: {str(e)}]"

    return Response(generate_chat(), content_type='text/plain; charset=utf-8')


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == '__main__':
    print("🚀 服务即将启动...")
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000, threaded=True)