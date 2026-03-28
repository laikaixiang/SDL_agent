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
from pydantic import BaseModel, Field, create_model, ValidationError
from typing import List, Optional, Literal, Dict, Any
from hardware import tools

app = Flask(__name__)

# ==========================================
# 配置参数
# ==========================================
SILICONFLOW_API_KEY = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"
PDF_FOLDER = r"test"
MODEL_NAME_VL = "Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_NAME_TALK = "Qwen/Qwen2.5-7B-Instruct"
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


class DynamicFieldsResponse(BaseModel):
    fields: List[str] = Field(description="推断出的需要提取的数据列名列表")

# LLM 动态提取字段生成器
def get_dynamic_fields_from_llm(task_desc):
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    # 自动获取 Pydantic 模型的 JSON Schema
    schema_str = json.dumps(DynamicFieldsResponse.model_json_schema(), ensure_ascii=False)

    prompt = (
        f"你是一个文献数据抽取专家。用户希望进行以下信息提取任务：【{task_desc}】。\n"
        "请推断为了完成这个任务，最终的数据表格需要包含哪些列名（字段）？\n"
        "🚨 你必须直接输出一个 JSON 对象，不要输出任何 Markdown 标记（如 ```json）、不要输出代码块，也不要输出任何解释性文字。\n"
        "🚨 你的输出必须严格符合以下格式：\n"
        '{"fields": ["列名1", "列名2", "列名3"]}\n'
    )

    payload = {
        "model": MODEL_NAME_TALK,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}  # 强制后端启用 JSON 模式
    }

    max_retries = 3
    last_error = ""
    for attempt in range(max_retries):
        # 3次尝试，保证不超时
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'].strip()
            clean_text = re.sub(r'```json\n|\n```|```', '', content).strip()

            result = DynamicFieldsResponse.model_validate_json(clean_text)
            if result.fields:
                return True, result.fields  # 成功：返回 (True, 结果)
        except ValidationError as ve:
            last_error = f"模型未按格式输出: {ve}"
        except requests.exceptions.Timeout:
            last_error = "API 请求超时"
        except Exception as e:
            last_error = f"网络请求异常: {str(e)}"

        if attempt < max_retries - 1:
            time.sleep(2.0)

    # 失败：返回 (False, 错误信息)
    return False, f"尝试 {max_retries} 次后仍失败。最后一次错误：{last_error}"


# LLM 动态生成英文文件名前缀
def get_filename_prefix(task_desc):
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    prompt = f"将以下提取任务的核心关键词翻译为简短的英文（单词之间用下划线连接），仅输出英文，不要有其他字符。任务：{task_desc}"
    payload = {"model": MODEL_NAME_TALK, "messages": [{"role": "user", "content": prompt}],
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

    save_dir = "extract"
    os.makedirs(save_dir, exist_ok=True)
    prefix = get_filename_prefix(task_description)

    if not os.path.exists(PDF_FOLDER):
        task_queue.put({"type": "error", "message": f"找不到文件夹: {PDF_FOLDER}"})
        task_running = False
        return

    # 🌟 动态生成当前任务专属的 Pydantic 模型
    # 所有字段设为 Optional，防止大模型没找到某些字段时直接崩溃
    field_definitions = {f: (Optional[str], Field(default="", description=f"提取：{f}")) for f in fields}
    DynamicRecord = create_model('DynamicRecord', **field_definitions)

    # 包装成最终期望的输出格式
    class PageExtractionResponse(BaseModel):
        data: List[DynamicRecord] = Field(default=[],
                                          description="提取到的文献数据实体列表，如果没有发现目标，返回空列表 []")

    schema_str = json.dumps(PageExtractionResponse.model_json_schema(), ensure_ascii=False)

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    all_extracted_data = []

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
                # 中断判断
                if cancel_requested: break

                img_base64 = pdf_page_to_image(pdf_path, page_num)
                task_queue.put(
                    {"type": "page_reading", "data": {"filename": filename, "page": page_num + 1, "image": img_base64}})

                # 🌟 构造一个极简的 JSON 示例给视觉模型抄作业，代替复杂的 Schema
                example_item = {f: "提取的内容" for f in fields}
                example_json = json.dumps({"data": [example_item]}, ensure_ascii=False)

                sys_prompt = (
                    f"你是一个专业的学术文献分析专家。你的任务是从提供的文献页面图像中提取：\n【目标】：{task_description}\n\n"
                    "提取规则：\n"
                    "1. 复合材料（含+、and等）不可拆分，需提取比例，若无比例标注（未说明比例）。若已提取过则不重复。\n"
                    "2. 溶剂量/浓度/转速/温度必须包含单位。\n"
                    "3. 忽略参考文献条目中的数据。\n\n"
                    "🚨 你必须直接输出一个 JSON 对象，绝不要包含 Markdown 标记（如 ```json）或任何其他解释性文字！\n"
                    f"🚨 必须严格遵循以下 JSON 格式：\n{example_json}"
                )

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",
                     "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]}
                ]

                headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
                # 移除 response_format，很多视觉模型不支持这个参数会导致直接报错
                payload = {"model": MODEL_NAME_VL, "messages": messages, "temperature": 0.1, "max_tokens": 1024,
                           "stream": True}
                # 为视觉提取加入重试机制，并将错误推送到前端侧边栏任务流
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=90, stream=True)
                        response.raise_for_status()  # 拦截 502/504 等错误

                        # 发送一个信号给前端：大模型开始输出了
                        task_queue.put({"type": "reading_start"})

                        result_text = ""
                        # 🌟 1. 逐行读取流式数据，推送到前端产生“打字机”效果
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    if data_str.strip() == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(data_str)
                                        content = chunk_json['choices'][0]['delta'].get('content', '')
                                        if content:
                                            result_text += content
                                            # 推送单字片段给前端悬浮窗
                                            task_queue.put({"type": "reading_chunk", "chunk": content})
                                    except Exception:
                                        pass

                        # 🌟 2. 打印原始输出，方便在后台排查大模型到底说了啥废话
                        print(f"\n--- 第 {page_num + 1} 页 模型原始输出 ---\n{result_text}\n-----------------------")

                        # 🌟 3. 强力正则：无视前后的废话和 Markdown，直接把最外层的 {...} 或 [...] 抠出来
                        json_match = re.search(r'(\{.*\}|\[.*\])', result_text, re.DOTALL)
                        if json_match:
                            clean_text = json_match.group(1).strip()
                        else:
                            clean_text = result_text.strip()

                        # 🌟 4. 容错处理：如果大模型很随性地直接返回了数组 [...]，手动帮它套上 {"data": ...}
                        if clean_text.startswith('['):
                            clean_text = f'{{"data": {clean_text}}}'

                        # 🌟 5. 依然使用 Pydantic 进行严格反序列化校验
                        parsed_res = PageExtractionResponse.model_validate_json(clean_text)

                        for item in parsed_res.data:
                            item_dict = item.model_dump()
                            item_dict['_source_doc'] = doc_id
                            all_extracted_data.append(item_dict)
                            # 推送到前端展示提取到的数据
                            task_queue.put({"type": "finding",
                                            "data": {"page": page_num + 1, "filename": filename, "details": item_dict}})

                        # 如果没有报错，说明提取并校验成功，跳出重试循环
                        break

                    except ValidationError as ve:
                        # 🌟 报错直接推给前端，不再隐藏在控制台
                        task_queue.put({"type": "error",
                                        "message": f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) 模型输出格式异常。"})
                    except requests.exceptions.Timeout:
                        task_queue.put({"type": "error",
                                        "message": f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) API 请求超时！"})
                    except Exception as e:
                        task_queue.put({"type": "error",
                                        "message": f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) 解析失败: {str(e)}"})

                    # 失败后稍作等待再次重试
                    time.sleep(2.0)

        # pdf处理失败
        except FileNotFoundError:
            task_queue.put({"type": "error", "message": f"❌ 文件不存在：{filename}"})
        except fitz.FileDataError:
            task_queue.put({"type": "error", "message": f"❌ PDF 文件损坏/无法读取：{filename}"})
        except Exception as e:
            task_queue.put({"type": "error", "message": f"❌ 处理文件 {filename} 失败：{str(e)}"})

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

    task_queue.put(
        {"type": "complete", "csv": csv_filename_temporal, "count": len(all_extracted_data), "fields": fields})
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
            success, fields = get_dynamic_fields_from_llm(task_desc)

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

    # 🌟 硬件控制
    if user_message.startswith("硬件控制："):
        cmd_text = user_message.replace("硬件控制：", "").strip()

        # 定义严格的硬件参数类型验证器
        class HardwareCommand(BaseModel):
            # Literal 强制大模型只能在这三个词里选，极大降低幻觉率
            action: Literal["set_temperature", "move_robot_arm", "do_experiment"] = Field(
                description="必须选择执行的动作")
            params: Dict[str, Any] = Field(description="动作所需参数")

        schema_str = json.dumps(HardwareCommand.model_json_schema(), ensure_ascii=False)

        hw_prompt = (
            "你是一个专业的实验室自动化硬件控制智能体。请将用户的自然语言指令转换为下位机可执行的数据。\n\n"
            "系统支持的 action 及所需 params 规则：\n"
            "1. set_temperature: 需要参数 target (浮点数)\n"
            "2. move_robot_arm: 需要参数 x, y, z (浮点数)\n"
            "3. do_experiment: 需要试剂 reagent(字符串), spin_speed(整数 rpm), spin_acc(整数 rpm/s,默认1000), spin_dur(整数 毫秒), volume(整数)\n\n"
            f"🚨 必须严格按照以下 JSON Schema 输出对象：\n{schema_str}"
        )

        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME_TALK,
            "messages": [
                {"role": "system", "content": hw_prompt},
                {"role": "user", "content": cmd_text}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            llm_json = resp.json()['choices'][0]['message']['content'].strip()
            clean_json = re.sub(r'```json\n|\n```|```', '', llm_json).strip()

            # 🌟 Pydantic 拦截硬件指令，校验不通过将直接进入 except
            valid_cmd = HardwareCommand.model_validate_json(clean_json)

            # 使用通过校验的数据生成纯净的 JSON 传给底层
            safe_payload = json.dumps(valid_cmd.model_dump(), ensure_ascii=False)
            hw_result = tools.execute_llm_hardware_command(safe_payload)

            status_icon = "✅" if hw_result.get("status") == "success" else "❌"
            reply_msg = (
                f"🔧 **硬件调度执行完毕**\n\n"
                f"**安全解析指令**：\n`{safe_payload}`\n\n"
                f"**执行状态**：{status_icon} {hw_result.get('status')}\n"
                f"**底层反馈**：{hw_result.get('output', hw_result.get('message', '无反馈'))}"
            )
            return jsonify({'type': 'system', 'reply': reply_msg})

        except ValidationError as ve:
            return jsonify(
                {'type': 'system', 'reply': f"❌ 指令拦截：AI生成的硬件参数不合法或缺失必要字段。\n详细错误：{ve}"})
        except Exception as e:
            return jsonify({'type': 'system', 'reply': f"❌ 硬件调度失败: {str(e)}"})

    # 🌟 普通聊天流式输出
    def generate_chat():
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME_TALK, "messages": [{"role": "user", "content": user_message}], "stream": True}
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
