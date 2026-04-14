"""
多图片输入支持测试 - 验证API是否支持批量图片输入

功能：
1. 测试单张图片输入（验证当前实现）
2. 测试多张图片输入（验证批量方案可行性）
3. 测试PDF原生支持（验证备选方案）

使用方法：
    python test/api_test/test_multi_image_support.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import requests
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 安全导入Config，避免触发core/__init__.py的所有导入
try:
    from core.config import Config
except ImportError as e:
    print(f"导入配置模块出错: {e}")
    print("尝试使用备用导入方法...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "core/config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    Config = config_module.Config


class MultiImageTester:
    """多图片输入测试器"""

    def __init__(self):
        self.config = Config()
        self.test_results = []

    def print_header(self, title):
        """打印测试标题"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_result(self, test_name, success, message=""):
        """打印测试结果"""
        status = "[通过]" if success else "[失败]"
        print(f"{status} {test_name}")
        if message:
            print(f"      {message}")
        self.test_results.append((test_name, success, message))

    def create_test_image(self, text, color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        创建测试图片

        Args:
            text: 图片上的文字
            color: 文字颜色
            bg_color: 背景颜色

        Returns:
            Base64编码的图片字符串
        """
        # 创建一个简单的测试图片
        img = Image.new('RGB', (400, 300), color=bg_color)
        draw = ImageDraw.Draw(img)

        # 尝试使用默认字体
        try:
            # 在Windows上尝试使用Arial字体
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            # 如果失败，使用默认字体
            font = ImageFont.load_default()

        # 计算文字位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((400 - text_width) // 2, (300 - text_height) // 2)

        # 绘制文字
        draw.text(position, text, fill=color, font=font)

        # 转换为Base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    def create_test_pdf(self):
        """
        创建测试PDF文件

        Returns:
            Base64编码的PDF字符串
        """
        try:
            import fitz  # PyMuPDF

            # 创建一个简单的PDF
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)  # A4尺寸

            # 添加文字
            text = "This is a test PDF document.\nPage 1"
            page.insert_text((50, 50), text, fontsize=20)

            # 保存到内存
            pdf_bytes = doc.tobytes()
            doc.close()

            # 转换为Base64
            pdf_str = base64.b64encode(pdf_bytes).decode("utf-8")
            return pdf_str

        except Exception as e:
            print(f"创建测试PDF失败: {e}")
            return None

    def test_single_image(self):
        """测试1: 单张图片输入"""
        self.print_header("测试1: 单张图片输入（当前实现验证）")

        try:
            # 创建测试图片
            img_base64 = self.create_test_image("Test Image 1", color=(255, 255, 0))

            headers = {
                "Authorization": f"Bearer {self.config.API_KEY}",
                "Content-Type": "application/json"
            }

            # 尝试方式1: 使用content数组（当前系统使用的格式）
            messages_v1 = [
                {
                    "role": "system",
                    "content": "你是一个图片分析助手。请描述你看到的图片内容。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]

            payload_v1 = {
                "model": self.config.MODEL_NAME_VL,
                "messages": messages_v1,
                "max_tokens": 200,
                "temperature": 0.1
            }

            print(f"正在测试模型: {self.config.MODEL_NAME_VL}")
            print("尝试方式1: content数组格式...")

            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload_v1,
                timeout=30,
                verify=False
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result(
                        "单张图片输入 (content数组)",
                        True,
                        f"API成功处理单张图片\n      响应: {content[:150]}..."
                    )
                    return True
            else:
                print(f"      方式1失败: {response.status_code}")
                print(f"      响应: {response.text[:200]}")

            # 尝试方式2: 使用简单的文本content + 图片URL
            print("\n尝试方式2: 简单文本格式...")

            messages_v2 = [
                {
                    "role": "user",
                    "content": f"请描述这张图片的内容。\n\n![image](data:image/jpeg;base64,{img_base64[:100]}...)"
                }
            ]

            payload_v2 = {
                "model": self.config.MODEL_NAME_VL,
                "messages": messages_v2,
                "max_tokens": 200,
                "temperature": 0.1
            }

            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload_v2,
                timeout=30,
                verify=False
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result(
                        "单张图片输入 (简单文本)",
                        True,
                        f"API成功处理单张图片\n      响应: {content[:150]}..."
                    )
                    return True

            self.print_result(
                "单张图片输入",
                False,
                f"两种格式都失败\n      状态码: {response.status_code}\n      响应: {response.text[:200]}"
            )
            return False

        except Exception as e:
            self.print_result("单张图片输入", False, f"错误: {str(e)}")
            return False

    def test_multiple_images(self):
        """测试2: 多张图片输入"""
        self.print_header("测试2: 多张图片输入（批量方案验证）")

        try:
            # 创建3张测试图片
            img1 = self.create_test_image("Image 1", color=(255, 0, 0), bg_color=(255, 255, 255))
            img2 = self.create_test_image("Image 2", color=(0, 255, 0), bg_color=(255, 255, 255))
            img3 = self.create_test_image("Image 3", color=(0, 0, 255), bg_color=(255, 255, 255))

            headers = {
                "Authorization": f"Bearer {self.config.API_KEY}",
                "Content-Type": "application/json"
            }

            # 构建消息 - 包含多张图片
            messages = [
                {
                    "role": "system",
                    "content": "你是一个图片分析助手。请描述你看到的所有图片内容，并说明一共有几张图片。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img1}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img2}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img3}"}
                        }
                    ]
                }
            ]

            payload = {
                "model": self.config.MODEL_NAME_VL,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.1
            }

            print(f"正在测试模型: {self.config.MODEL_NAME_VL}")
            print("发送3张图片...")

            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload,
                timeout=60,
                verify=False  # 禁用SSL验证
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result(
                        "多张图片输入",
                        True,
                        f"API支持多张图片输入！\n      响应: {content[:200]}..."
                    )
                    print("\n      [重要] 批量多图片方案可行！可以实施批量处理功能。")
                    return True
                else:
                    self.print_result("多张图片输入", False, "响应内容为空")
                    return False
            else:
                error_msg = response.text[:300]
                self.print_result(
                    "多张图片输入",
                    False,
                    f"状态码: {response.status_code}\n      响应: {error_msg}"
                )
                print("\n      [提示] API可能不支持多张图片输入，建议保持逐页处理模式。")
                return False

        except Exception as e:
            self.print_result("多张图片输入", False, f"错误: {str(e)}")
            print("\n      [提示] 测试失败，建议保持逐页处理模式。")
            return False

    def test_pdf_native_support(self):
        """测试3: PDF原生支持"""
        self.print_header("测试3: PDF原生支持（备选方案验证）")

        try:
            # 创建测试PDF
            pdf_base64 = self.create_test_pdf()

            if not pdf_base64:
                self.print_result("PDF原生支持", False, "无法创建测试PDF")
                return False

            headers = {
                "Authorization": f"Bearer {self.config.API_KEY}",
                "Content-Type": "application/json"
            }

            # 尝试方式1: 使用document类型（Claude API格式）
            messages_v1 = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64
                            }
                        }
                    ]
                }
            ]

            payload_v1 = {
                "model": self.config.MODEL_NAME_VL,
                "messages": messages_v1,
                "max_tokens": 200
            }

            print(f"正在测试模型: {self.config.MODEL_NAME_VL}")
            print("尝试方式1: document类型...")

            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload_v1,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result(
                        "PDF原生支持 (document类型)",
                        True,
                        f"API支持PDF原生输入！\n      响应: {content[:150]}..."
                    )
                    print("\n      [重要] 可以考虑使用PDF原生方案，无需转换为图片！")
                    return True

            # 尝试方式2: 使用image_url类型但传PDF
            print("\n尝试方式2: image_url类型传PDF...")

            messages_v2 = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_base64}"
                            }
                        }
                    ]
                }
            ]

            payload_v2 = {
                "model": self.config.MODEL_NAME_VL,
                "messages": messages_v2,
                "max_tokens": 200
            }

            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload_v2,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result(
                        "PDF原生支持 (image_url类型)",
                        True,
                        f"API支持PDF输入！\n      响应: {content[:150]}..."
                    )
                    return True

            # 两种方式都失败
            self.print_result(
                "PDF原生支持",
                False,
                "API不支持PDF原生输入，需要转换为图片"
            )
            print("\n      [提示] 继续使用PDF转图片的方案。")
            return False

        except Exception as e:
            self.print_result("PDF原生支持", False, f"错误: {str(e)}")
            return False

    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试总结与建议")

        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed

        print(f"\n总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {failed}")

        # 分析结果并给出建议
        print("\n" + "=" * 70)
        print("  实施建议")
        print("=" * 70)

        single_image_passed = self.test_results[0][1] if len(self.test_results) > 0 else False
        multi_image_passed = self.test_results[1][1] if len(self.test_results) > 1 else False
        pdf_native_passed = self.test_results[2][1] if len(self.test_results) > 2 else False

        if pdf_native_passed:
            print("\n[推荐] 推荐方案: PDF原生输入")
            print("   - API支持直接读取PDF文件")
            print("   - 无需转换为图片，节省处理时间")
            print("   - 实施优先级: 最高")
        elif multi_image_passed:
            print("\n[推荐] 推荐方案: 批量多图片输入")
            print("   - API支持一次发送多张图片")
            print("   - 可以实施批量处理功能")
            print("   - 建议每批处理5-10页")
            print("   - 实施优先级: 高")
        elif single_image_passed:
            print("\n[提示] 推荐方案: 保持当前逐页处理")
            print("   - API仅支持单张图片输入")
            print("   - 继续使用现有的逐页处理逻辑")
            print("   - 可以优化: 并发处理多个页面")
        else:
            print("\n[警告] 警告: 所有测试均失败")
            print("   - 请检查API配置和模型设置")
            print("   - 确认视觉模型是否正确")

        print("\n" + "=" * 70 + "\n")

        return passed > 0

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("  SDL Agent - 多图片输入支持测试")
        print("  验证API是否支持批量图片/PDF输入")
        print("=" * 70)

        # 运行测试
        self.test_single_image()
        print("\n等待2秒后继续...")
        import time
        time.sleep(2)

        self.test_multiple_images()
        print("\n等待2秒后继续...")
        time.sleep(2)

        self.test_pdf_native_support()

        # 打印总结
        return self.print_summary()


def main():
    """主函数"""
    tester = MultiImageTester()
    success = tester.run_all_tests()

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
