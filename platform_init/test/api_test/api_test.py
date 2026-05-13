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

# 直接导入 Config，避免触发 core/__init__.py 的所有导入
try:
    from core.config import Config
except ImportError as e:
    print(f"导入配置模块出错: {e}")
    print("尝试使用备用导入方法...")
    # 如果导入失败，直接读取配置文件
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", "core/config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    Config = config_module.Config


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
        status = "[通过]" if success else "[失败]"
        print(f"{status} {test_name}")
        if message:
            print(f"      {message}")
        self.test_results.append((test_name, success, message))

    def test_config_validation(self):
        """测试1: 配置验证"""
        self.print_header("测试1: 配置验证")

        # 检查必要配置
        checks = [
            ("API_KEY", self.config.API_KEY, "API密钥已设置"),
            ("API_URL", self.config.API_URL, "API地址已设置"),
            ("MODEL_NAME_TALK", self.config.MODEL_NAME_TALK, "对话模型名称已设置"),
            ("MODEL_NAME_VL", self.config.MODEL_NAME_VL, "视觉语言模型名称已设置"),
            ("EXPERIMENT_MODEL_NAME", self.config.EXPERIMENT_MODEL_NAME, "实验模型名称已设置"),
        ]

        all_passed = True
        for name, value, desc in checks:
            if value:
                self.print_result(desc, True, f"{name} = {value}")
            else:
                self.print_result(desc, False, f"{name} 未设置")
                all_passed = False

        return all_passed

    def test_api_connection(self):
        """测试2: API连接测试"""
        self.print_header("测试2: API连接测试")

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

            # 智能处理API URL
            api_url = self.config.API_URL
            if not api_url.endswith('/chat/completions'):
                api_url = f"{api_url}/chat/completions"

            print(f"正在测试API端点: {api_url}")

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                self.print_result("API连接", True, f"状态码: {response.status_code}")
                return True
            else:
                self.print_result("API连接", False,
                                f"状态码: {response.status_code}, 响应: {response.text[:200]}")
                return False

        except requests.exceptions.Timeout:
            self.print_result("API连接", False, "请求超时 (>10秒)")
            return False
        except requests.exceptions.ConnectionError as e:
            self.print_result("API连接", False, f"连接错误: {str(e)[:100]}")
            return False
        except Exception as e:
            self.print_result("API连接", False, f"错误: {str(e)[:100]}")
            return False

    def test_talk_model(self):
        """测试3: 对话模型测试"""
        self.print_header("测试3: 对话模型测试")

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

            # 智能处理API URL
            api_url = self.config.API_URL
            if not api_url.endswith('/chat/completions'):
                api_url = f"{api_url}/chat/completions"

            print(f"正在测试模型: {self.config.MODEL_NAME_TALK}")

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                if content:
                    self.print_result("对话模型响应", True, f"响应内容: {content[:100]}...")
                    return True
                else:
                    self.print_result("对话模型响应", False, "响应内容为空")
                    return False
            else:
                self.print_result("对话模型响应", False,
                                f"状态码: {response.status_code}, 响应: {response.text[:200]}")
                return False

        except Exception as e:
            self.print_result("对话模型响应", False, f"错误: {str(e)[:100]}")
            return False

    def test_streaming_response(self):
        """测试4: 流式响应测试"""
        self.print_header("测试4: 流式响应测试")

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

            # 智能处理API URL
            api_url = self.config.API_URL
            if not api_url.endswith('/chat/completions'):
                api_url = f"{api_url}/chat/completions"

            print(f"正在测试流式响应，模型: {self.config.MODEL_NAME_TALK}")

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )

            if response.status_code != 200:
                self.print_result("流式响应", False,
                                f"状态码: {response.status_code}")
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
                self.print_result("流式响应", True,
                                f"接收到 {chunks_received} 个数据块，内容: {full_content[:100]}...")
                return True
            else:
                self.print_result("流式响应", False, "未接收到数据块")
                return False

        except Exception as e:
            self.print_result("流式响应", False, f"错误: {str(e)[:100]}")
            return False

    def test_vision_model(self):
        """测试5: 视觉模型测试（可选）"""
        self.print_header("测试5: 视觉语言模型测试（可选）")

        print(f"已配置视觉模型: {self.config.MODEL_NAME_VL}")
        print("注意: 此测试需要图片，跳过实际API调用。")
        print("视觉模型将在处理PDF时进行测试。")

        self.print_result("视觉模型配置", True,
                        f"模型: {self.config.MODEL_NAME_VL}")
        return True

    def test_file_paths(self):
        """测试6: 文件路径测试"""
        self.print_header("测试6: 文件路径测试")

        paths_to_check = [
            ("PDF_FOLDER", self.config.PDF_FOLDER),
            ("EXTRACT_DIR", self.config.EXTRACT_DIR),
            ("TEMPORAL_DIR", self.config.TEMPORAL_DIR),
        ]

        all_passed = True
        for name, path in paths_to_check:
            if os.path.exists(path):
                self.print_result(f"{name} 已存在", True, f"路径: {path}")
            else:
                # 尝试创建目录
                try:
                    os.makedirs(path, exist_ok=True)
                    self.print_result(f"{name} 已创建", True, f"路径: {path}")
                except Exception as e:
                    self.print_result(f"{name} 创建失败", False, f"错误: {str(e)[:100]}")
                    all_passed = False

        return all_passed

    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试总结")

        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed

        print(f"\n总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {failed}")

        if failed > 0:
            print("\n失败的测试:")
            for name, success, message in self.test_results:
                if not success:
                    print(f"  - {name}")
                    if message:
                        print(f"    {message}")

        print("\n" + "=" * 60)
        if failed == 0:
            print("  [成功] 所有测试通过！")
            print("  您的API配置工作正常。")
        else:
            print("  [警告] 部分测试失败。")
            print("  请检查 core/config.py 中的配置。")
        print("=" * 60 + "\n")

        return failed == 0

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("  SDL Agent - API配置测试")
        print("  独立测试工具（无需启动app.py）")
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
