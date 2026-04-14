"""
简单的视觉API测试 - 直接测试不依赖其他模块
"""

import requests
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 直接配置（从config.py复制）
API_KEY = "ak_2hX7A491t2kw4Oy9Ae8Bi77e1kQ3J"
API_URL = "https://api.longcat.chat/openai/v1/chat/completions"
MODEL_NAME_VL = "LongCat-Flash-Omni-2603"

print(f"配置信息:")
print(f"  API_URL: {API_URL}")
print(f"  MODEL: {MODEL_NAME_VL}")
print(f"  API_KEY: {API_KEY[:20]}...")
print()


def create_test_image(text):
    """创建测试图片"""
    img = Image.new('RGB', (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((400 - text_width) // 2, (300 - text_height) // 2)

    draw.text(position, text, fill=(0, 0, 0), font=font)

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


def test_vision_api():
    """测试视觉API"""

    print("=" * 70)
    print("  简单视觉API测试")
    print("=" * 70)

    # 创建测试图片
    print("\n[1] 创建测试图片...")
    img_base64 = create_test_image("Hello World")
    print(f"    图片大小: {len(img_base64)} 字符")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 测试1: 标准OpenAI格式（content数组）
    print("\n[2] 测试标准OpenAI格式（content数组）...")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图片的内容"
                }
            ]
        }
    ]

    payload = {
        "model": MODEL_NAME_VL,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.1
    }

    print(f"    模型: {MODEL_NAME_VL}")
    print(f"    发送请求...")

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False
        )

        print(f"    状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"    [成功] 响应: {content[:200]}...")
            return True
        else:
            print(f"    [失败] 响应: {response.text[:300]}")

            # 测试2: 简化格式（只有文本）
            print("\n[3] 测试简化格式（纯文本）...")

            messages2 = [
                {
                    "role": "user",
                    "content": "你好，请介绍一下你自己"
                }
            ]

            payload2 = {
                "model": MODEL_NAME_VL,
                "messages": messages2,
                "max_tokens": 100
            }

            response2 = requests.post(
                API_URL,
                headers=headers,
                json=payload2,
                timeout=30,
                verify=False
            )

            print(f"    状态码: {response2.status_code}")
            print(f"    请求体: {json.dumps(payload2, ensure_ascii=False)[:200]}")

            if response2.status_code == 200:
                data2 = response2.json()
                content2 = data2.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"    [成功] 纯文本可用: {content2[:200]}...")
                print("\n    [结论] 模型可用但可能不支持图片输入")
                return False
            else:
                print(f"    [失败] 响应: {response2.text[:300]}")
                print("\n    [结论] API配置可能有问题")
                return False

    except Exception as e:
        print(f"    [错误] {str(e)}")
        return False


if __name__ == "__main__":
    result = test_vision_api()

    print("\n" + "=" * 70)
    if result:
        print("  [结论] 视觉API工作正常，支持图片输入")
    else:
        print("  [结论] 视觉API可能不支持图片输入或配置有问题")
    print("=" * 70)
