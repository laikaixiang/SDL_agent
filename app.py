from flask import Flask, request, jsonify, render_template
import requests
import threading
import os
import base64
import json
import time
import re
import csv
import random
import pickle
from collections import defaultdict, Counter
import fitz  # PyMuPDF
from PIL import Image
import io
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

app = Flask(__name__)

# ==========================================
# ⚙️ 全局配置参数 (请在这里修改为你的真实信息)
# ==========================================
SILICONFLOW_API_KEY = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"
PDF_FOLDER = r"test"  # 建议加上 r 防止路径转义报错
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 提取任务专属参数
DPI = 200
REQUEST_DELAY = 3.0
MAX_RETRIES = 5
MIN_DELAY = 2.0
MAX_DELAY = 60.0
PROGRESS_FILE = "progress_state.pkl"


# ==========================================
# 🛠️ 核心功能：PDF 处理与 API 请求函数
# ==========================================
def save_progress_state(state):
    """保存进度状态到文件"""
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(state, f)


def load_progress_state():
    """从文件加载进度状态"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None


def pdf_page_to_image(pdf_path, page_num):
    """将PDF页面转换为Base64编码的图像"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    doc.close()
    return img_str


def extract_chat_completion(messages, model=MODEL_NAME):
    """专门用于提取任务的 API 调用 - 带重试机制和速率限制处理"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 500
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip(), None
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                wait_time = float(retry_after) if retry_after else min(MAX_DELAY, MIN_DELAY * (2 ** attempt))
                jitter = random.uniform(0.5, 1.5)
                wait_time *= jitter
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"API错误: {response.status_code}"
                return "", error_msg
        except requests.exceptions.RequestException as e:
            wait_time = min(MAX_DELAY, MIN_DELAY * (2 ** attempt))
            time.sleep(wait_time)
            continue
        except (KeyError, ValueError) as e:
            return "", f"响应解析错误: {str(e)}"
    return "", "达到最大重试次数后仍然失败"


def normalize_chemical_name(name):
    """简单标准化化学名称"""
    name = re.sub(r'[^\w\s-]', '', name.strip())
    return name


def process_pdf_library():
    """处理整个PDF文献库的主干函数（将在后台线程运行）"""
    print("\n--- 🚀 后台提取任务已启动 ---")
    progress_state = load_progress_state()
    if progress_state:
        passivator_docs = progress_state['passivator_docs']
        unique_passivators = progress_state['unique_passivators']
        processed_files = progress_state['processed_files']
        api_errors = progress_state['api_errors']
        start_file_idx = progress_state['file_idx'] + 1
    else:
        passivator_docs = defaultdict(set)
        unique_passivators = set()
        api_errors = []
        processed_files = set()
        start_file_idx = 0

    if not os.path.exists(PDF_FOLDER):
        print(f"❌ 错误：找不到 PDF 文件夹路径 {PDF_FOLDER}，请检查配置！")
        return

    pdf_files = []
    for f in os.listdir(PDF_FOLDER):
        if f.lower().endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, f)
            try:
                with fitz.open(pdf_path) as doc:
                    pdf_files.append((f, len(doc)))
            except Exception as e:
                print(f"无法读取文件: {f} - {str(e)}")

    pdf_files = [(f, p) for f, p in pdf_files if f not in processed_files]

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeRemainingColumn()
    ]

    with Progress(*progress_columns) as progress:
        main_task = progress.add_task("[cyan]处理文献...", total=len(pdf_files), visible=True)

        for file_idx, (filename, num_pages) in enumerate(pdf_files):
            if file_idx < start_file_idx:
                progress.update(main_task, advance=1)
                continue

            pdf_path = os.path.join(PDF_FOLDER, filename)
            doc_id = os.path.splitext(filename)[0]
            progress.update(main_task, advance=1,
                            description=f"[cyan]处理文献 {file_idx + 1}/{len(pdf_files)}: {filename[:20]}")

            with progress:
                page_task = progress.add_task(f"[green]处理 {filename}...", total=num_pages, visible=True)
                try:
                    for page_num in range(num_pages):
                        progress.update(page_task, advance=1,
                                        description=f"[green]{filename} 页面 {page_num + 1}/{num_pages}")
                        img_base64 = pdf_page_to_image(pdf_path, page_num)

                        messages = [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "你是有机光伏材料领域的专业研究员。请从这张文献页面图像中识别并列出所有"
                                            "专门用于FAPbI3钙钛矿体系的钝化剂(passivator)化学名称。\n"
                                            "重要要求：\n"
                                            "1. 只返回针对FAPbI3体系的钝化剂化学名称\n"
                                            "2. 多个名称用英文分号分隔(;)\n"
                                            "3. 不要添加任何解释、分析或前缀文本\n"
                                            "4. 严格按照以下示例格式之一输出：\n"
                                            "   - 多个名称: `CsI; RbI; EAI`\n"
                                            "   - 单个名称: `CsI`\n"
                                            "   - 没有钝化剂: `无`\n"
                                            "5. 如果无法确定，返回`无`\n"
                                            "请严格遵守输出格式要求，这是最重要的！"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                                }
                            ]
                        }]

                        response, error = extract_chat_completion(messages)

                        if error:
                            api_errors.append(f"{filename} 第 {page_num + 1} 页: {error}")
                            if "速率限制" in error or "429" in error:
                                save_progress_state({
                                    'passivator_docs': passivator_docs, 'unique_passivators': unique_passivators,
                                    'api_errors': api_errors, 'processed_files': processed_files | {filename},
                                    'file_idx': file_idx
                                })
                                return

                        elif response and response.lower() != "无":
                            page_passivators = [normalize_chemical_name(p.strip()) for p in response.split(';') if
                                                p.strip() and p.strip().lower() != "无"]
                            for p in page_passivators:
                                unique_passivators.add((p, doc_id))
                                passivator_docs[p].add(doc_id)

                        time.sleep(REQUEST_DELAY + random.uniform(0, 1.0))

                except Exception as e:
                    api_errors.append(f"{filename} 处理错误: {str(e)}")
                    continue
                finally:
                    progress.remove_task(page_task)
                    processed_files.add(filename)
                    save_progress_state({
                        'passivator_docs': passivator_docs, 'unique_passivators': unique_passivators,
                        'api_errors': api_errors, 'processed_files': processed_files, 'file_idx': file_idx
                    })

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    final_counts = Counter()
    for passivator, doc_id in unique_passivators:
        final_counts[passivator] += 1

    csv_filename = f"extract/FAPbI3_passivators_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Passivator", "文献出现次数", "相关文献"])
        for passivator, count in final_counts.most_common():
            related_docs = passivator_docs.get(passivator, set())
            doc_list = "; ".join(related_docs)
            if len(doc_list) > 200: doc_list = doc_list[:197] + "..."
            writer.writerow([passivator, count, doc_list])

    print(f"\n✅ 提取任务全部完成！结果已保存到: {csv_filename}\n")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': '无效的请求'}), 400

    user_message = data['message'].strip()

    # 🌟 拦截器：检测关键词
    if user_message.startswith("帮我搜寻："):
        # 开启后台线程执行庞大的 PDF 解析任务
        thread = threading.Thread(target=process_pdf_library)
        thread.start()

        return jsonify({
                           'reply': "好的老板！我已经收到了提取指令，正在后台为你启动【FAPbI3体系钝化剂】的 AI 提取任务。\n\n由于处理 PDF 图像非常耗时，请切回后端的终端窗口（命令行）查看实时的跑动进度条。处理完成后，会自动在你的项目目录下生成一份详尽的 CSV 统计文件！"})

    # 🌟 正常的对话逻辑
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个有用的 AI 助手。"},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return jsonify({'reply': response.json()['choices'][0]['message']['content'].strip()})
        elif response.status_code == 429:
            return jsonify({'reply': 'API 请求太频繁，请稍后再试。'})
        else:
            return jsonify({'reply': f'API 请求失败: {response.status_code}'})
    except Exception as e:
        return jsonify({'reply': f'请求大模型异常: {str(e)}'})


if __name__ == '__main__':
    print("🚀 后端服务已启动！请访问: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

