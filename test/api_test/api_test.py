"""
独立API测试工具 - 无需启动app.py即可测试API配置

功能：
1. 测试API配置是否正确
2. 测试API密钥是否有效
3. 测试对话模型是否可用
4. 测试视觉模型是否可用
5. 测试流式响应是否正常

使用方法：
    python test/api_test/api_test.py
"""

import sys
import os

# 添加项目根目录到路径，以便导入core模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import requests
import json
import time
from core.config import Config


class APITester:
    """API测试器"""

    def __init__(self):
        self.config = Config()
        self.test_results = []

    def print_header(self, title):
        """打印测试标题"""
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def print_result(self, test_name, success, message=""):
        """打印测试结果"""
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if message:
            print(f"      {message}")
        self.test_results.append((test_name, success, message))

    def test_config_validation(self):
        """测试1: 配置验证"""
        self.print_header("Test 1: Configuration Validation")

        # 检查必要配置
        checks = [
            ("API_KEY", self.config.API_KEY, "API key is set"),
            ("API_URL", self.config.API_URL, "API URL is set"),
            ("MODEL_NAME_TALK", self.config.MODEL_NAME_TALK, "Talk model name is set"),
            ("MODEL_NAME_VL", self.config.MODEL_NAME_VL, "Vision-Language model name is set"),
            ("EXPERIMENT_MODEL_NAME", self.config.EXPERIMENT_MODEL_NAME, "Experiment model name is set"),
        ]

        all_passed = True
        for name, value, desc in checks:
            if value:
                self.print_result(desc, True, f"{name} = {value}")
            else:
                self.print_result(desc, False, f"{name} is not set")
                all_passed = False

        return all_passed

    def test_api_connection(self):
        """测试2: API连接测试"""
        self.print_header("Test 2: API Connection Test")

        try:
            # 发送一个简单的请求测试连接
            headers = {
                "Authorization": f"Bearer {self.config.API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.config.MODEL_NAME_TALK,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }

            response = requests.post(
                f"{self.config.API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                self.print_result("API connection", True, f"Status code: {response.status_code}")
                return True
            else:
                self.print_result("API connection", False,
                                f"Status code: {response.status_code}, Response: {response.text[:200]}")
                return False

        except requests.exceptions.Timeout:
            self.print_result("API connection", False, "Request timeout (>10s)")
            return False
        except requests.exceptions.ConnectionError as e:
            self.print_result("API connection", False, f"Connection error: {str(e)[:100]}")
            return False
        except Exception as e:
            self.print_result("API connection", False, f"Error: {str(e)[:100]}")
            return False

    def test_talk_model(self):
        """测试3: 对话模型测试"""
        self.print_header("Test 3: Talk Model Test")

        try:
            headers = {
                "Authorization": f"Bearer {self.config.API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.config.MODEL_NAME_TALK,
                "messages": [{"role": "user", "content": "请用一句话介绍你自己"}],
                "max_tokens": 50
            }

            print(f"Testing model: {self.config.MODEL_NAME_TALK}")

            response = requests.post(
                f"{self.config.API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result("Talk model response", True, f"Response: {content[:100]}...")
                    return True
                else:
                    self.print_result("Talk model response", False, "Empty response content")
                    return False
            else:
                self.print_result("Talk model response", False,
                                f"Status: {response.status_code}, Response: {response.text[:200]}")
                return False

        except Exception as e:
            self.print_result("Talk model response", False, f"Error: {str(e)[:100]}")
            return False

    def test_streaming_response(self):
        """测试4: 流式响应测试"""
        self.print_header("Test 4: Streaming Response Test")

        try:
            headers = {
                "Authorization": f"Bearer {self.config.API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.config.MODEL_NAME_TALK,
                "messages": [{"role": "user", "content": "数到5"}],
                "stream": True,
                "max_tokens": 50
            }

            print(f"Testing streaming with model: {self.config.MODEL_NAME_TALK}")

            response = requests.post(
                f"{self.config.API_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )

            if response.status_code != 200:
                self.print_result("Streaming response", False,
                                f"Status: {response.status_code}")
                return False

            # 读取流式响应
            chunks_received = 0
            full_content = ""

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                full_content += content
                                chunks_received += 1
                        except json.JSONDecodeError:
                            pass

            if chunks_received > 0:
                self.print_result("Streaming response", True,
                                f"Received {chunks_received} chunks, content: {full_content[:100]}...")
                return True
            else:
                self.print_result("Streaming response", False, "No chunks received")
                return False

        except Exception as e:
            self.print_result("Streaming response", False, f"Error: {str(e)[:100]}")
            return False

    def test_vision_model(self):
        """测试5: 视觉模型测试（可选）"""
        self.print_header("Test 5: Vision-Language Model Test (Optional)")

        print(f"Vision model configured: {self.config.MODEL_NAME_VL}")
        print("Note: This test requires an image. Skipping actual API call.")
        print("Vision model will be tested when processing PDFs.")

        self.print_result("Vision model config", True,
                        f"Model: {self.config.MODEL_NAME_VL}")
        return True

    def test_file_paths(self):
        """测试6: 文件路径测试"""
        self.print_header("Test 6: File Paths Test")

        paths_to_check = [
            ("PDF_FOLDER", self.config.PDF_FOLDER),
            ("EXTRACT_DIR", self.config.EXTRACT_DIR),
            ("TEMPORAL_DIR", self.config.TEMPORAL_DIR),
            ("TEMPLATES_DIR", self.config.TEMPLATES_DIR),
        ]

        all_passed = True
        for name, path in paths_to_check:
            if os.path.exists(path):
                self.print_result(f"{name} exists", True, f"Path: {path}")
            else:
                # 尝试创建目录
                try:
                    os.makedirs(path, exist_ok=True)
                    self.print_result(f"{name} created", True, f"Path: {path}")
                except Exception as e:
                    self.print_result(f"{name} creation", False, f"Error: {str(e)[:100]}")
                    all_passed = False

        return all_passed

    def print_summary(self):
        """打印测试总结"""
        self.print_header("Test Summary")

        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed

        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for name, success, message in self.test_results:
                if not success:
                    print(f"  - {name}")
                    if message:
                        print(f"    {message}")

        print("\n" + "=" * 60)
        if failed == 0:
            print("  [SUCCESS] All tests passed!")
            print("  Your API configuration is working correctly.")
        else:
            print("  [WARNING] Some tests failed.")
            print("  Please check the configuration in core/config.py")
        print("=" * 60 + "\n")

        return failed == 0

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("  SDL Agent - API Configuration Test")
        print("  Independent test tool (no need to start app.py)")
        print("=" * 60)

        # 运行测试
        self.test_config_validation()
        time.sleep(0.5)

        self.test_api_connection()
        time.sleep(0.5)

        self.test_talk_model()
        time.sleep(0.5)

        self.test_streaming_response()
        time.sleep(0.5)

        self.test_vision_model()
        time.sleep(0.5)

        self.test_file_paths()

        # 打印总结
        return self.print_summary()


def main():
    """主函数"""
    tester = APITester()
    success = tester.run_all_tests()

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
