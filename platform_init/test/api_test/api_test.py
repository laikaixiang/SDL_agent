"""
独立API测试工具 - 无需启动app.py即可测试API配置

功能：
1. 测试配置完整性（所有模型 + API密钥）
2. 测试 TALK API 密钥是否有效
3. 测试 VL API 密钥是否有效
4. 测试 EXPERIMENT API 密钥是否有效
5. 测试 EMBEDDING API 密钥是否有效
6. 测试流式响应是否正常（TALK 模型）
7. 文件路径检查

使用方法：
    python platform_init/test/api_test/api_test.py
"""

import sys
import os

# 添加项目根目录到路径，以便导入core模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import requests
import json
import time
import base64

# 直接导入 Config，避免触发 core/__init__.py 的所有导入
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


# 1x1 白色像素 PNG（用于 VL 模型快速测试）
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)


class APITester:
    """API测试器 — 测试每个模型的独立 API_KEY/URL 是否有效"""

    def __init__(self):
        self.config = Config()
        self.test_results = []

    # ---- helpers ----

    def _header(self, title):
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def _ok(self, name, msg=""):
        print(f"[通过] {name}")
        if msg:
            print(f"       {msg}")
        self.test_results.append((name, True, msg))

    def _fail(self, name, msg=""):
        print(f"[失败] {name}")
        if msg:
            print(f"       {msg}")
        self.test_results.append((name, False, msg))

    def _skip(self, name, msg=""):
        print(f"[跳过] {name} — {msg}")
        self.test_results.append((name, None, msg))

    def _api_url_clean(self, raw_url):
        """确保 URL 以 /chat/completions 结尾（chat 类端点）"""
        if not raw_url.endswith('/chat/completions'):
            return f"{raw_url.rstrip('/')}/chat/completions"
        return raw_url

    # ---- test 1: 配置完整性 ----

    def test_config(self):
        self._header("测试1: 配置完整性")

        # 模型名称
        for key in ("MODEL_NAME_TALK", "MODEL_NAME_VL", "EXPERIMENT_MODEL_NAME", "EMBEDDING_MODEL"):
            v = getattr(self.config, key, "")
            if v:
                self._ok(key, v)
            else:
                self._fail(key, "未设置")

        # API 凭证（全局 + 独立），任一可用即可
        models = [
            ("TALK",    self.config.TALK_API_KEY,    self.config.TALK_API_URL),
            ("VL",      self.config.VL_API_KEY,      self.config.VL_API_URL),
            ("EXPERIMENT", self.config.EXPERIMENT_API_KEY, self.config.EXPERIMENT_API_URL),
            ("EMBEDDING",  self.config.EMBEDDING_API_KEY,  self.config.EMBEDDING_API_URL),
        ]
        for name, key, url in models:
            if key and url:
                self._ok(f"{name}_API_KEY/URL", f"key={key[:15]}... url={url}")
            elif key and not url:
                self._fail(f"{name}_API_KEY/URL", "API_KEY 已设置但 API_URL 为空")
            elif not key and url:
                self._fail(f"{name}_API_KEY/URL", "API_URL 已设置但 API_KEY 为空")
            else:
                # 回退到全局
                if self.config.API_KEY and self.config.API_URL:
                    self._ok(f"{name}_API_KEY/URL",
                             f"未单独设置，回退到全局 API_KEY (key={self.config.API_KEY[:15]}...)")
                else:
                    self._fail(f"{name}_API_KEY/URL", "未设置且全局 API_KEY/URL 也不可用")

    # ---- test 2: TALK API ----

    def test_talk_api(self):
        self._header("测试2: TALK API 密钥")

        key = self.config.TALK_API_KEY
        url = self._api_url_clean(self.config.TALK_API_URL)
        model = self.config.MODEL_NAME_TALK

        if not key:
            self._skip("TALK API", "TALK_API_KEY 为空且无全局回退")
            return

        self._do_chat_test("TALK API", key, url, model,
                           "请用一句话介绍你自己", "对话模型响应")

    # ---- test 3: VL API ----

    def test_vl_api(self):
        self._header("测试3: VL API 密钥（视觉模型）")

        key = self.config.VL_API_KEY
        url = self._api_url_clean(self.config.VL_API_URL)
        model = self.config.MODEL_NAME_VL

        if not key:
            self._skip("VL API", "VL_API_KEY 为空且无全局回退")
            return

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        # VL 模型用图片 + 文字请求测试
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_BASE64}"}},
                    {"type": "text", "text": "描述这张图片"}
                ]
            }],
            "max_tokens": 30,
            "stream": False,
        }

        print(f"  模型: {model}")
        print(f"  端点: {url}")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content:
                    self._ok("VL API", f"响应: {content[:100]}")
                else:
                    self._fail("VL API", "响应 content 为空")
            else:
                self._fail("VL API", f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            self._fail("VL API", "请求超时 (>30s)")
        except Exception as e:
            self._fail("VL API", str(e)[:150])

    # ---- test 4: EXPERIMENT API ----

    def test_experiment_api(self):
        self._header("测试4: EXPERIMENT API 密钥")

        key = self.config.EXPERIMENT_API_KEY
        url = self._api_url_clean(self.config.EXPERIMENT_API_URL)
        model = self.config.EXPERIMENT_MODEL_NAME

        if not key:
            self._skip("EXPERIMENT API", "EXPERIMENT_API_KEY 为空且无全局回退")
            return

        self._do_chat_test("EXPERIMENT API", key, url, model,
                           "Say hello in one sentence", "实验模型响应")

    # ---- test 5: EMBEDDING API ----

    def test_embedding_api(self):
        self._header("测试5: EMBEDDING API 密钥")

        key = self.config.EMBEDDING_API_KEY
        url = self.config.EMBEDDING_API_URL
        model = self.config.EMBEDDING_MODEL

        if not key:
            self._skip("EMBEDDING API", "EMBEDDING_API_KEY 为空")
            return

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "input": "Hello, this is a test sentence for embedding.",
        }

        print(f"  模型: {model}")
        print(f"  端点: {url}")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                vec = data.get('data', [{}])[0].get('embedding', [])
                dim = len(vec) if vec else 0
                self._ok("EMBEDDING API", f"返回 embedding 维度: {dim}")
            else:
                self._fail("EMBEDDING API", f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            self._fail("EMBEDDING API", "请求超时 (>30s)")
        except Exception as e:
            self._fail("EMBEDDING API", str(e)[:150])

    # ---- test 6: 流式响应 (TALK) ----

    def test_streaming(self):
        self._header("测试6: 流式响应 (TALK 模型)")

        key = self.config.TALK_API_KEY
        url = self._api_url_clean(self.config.TALK_API_URL)
        model = self.config.MODEL_NAME_TALK

        if not key:
            self._skip("流式响应", "TALK_API_KEY 为空且无全局回退")
            return

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "数到5"}],
            "stream": True,
            "max_tokens": 50,
        }

        print(f"  模型: {model}  (stream=True)")

        try:
            resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            if resp.status_code != 200:
                self._fail("流式响应", f"HTTP {resp.status_code}")
                return

            chunks = 0
            full = ""
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            c = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if c:
                                full += c
                                chunks += 1
                        except json.JSONDecodeError:
                            pass

            if chunks > 0:
                self._ok("流式响应", f"{chunks} 个 chunk, 内容: {full[:100]}")
            else:
                self._fail("流式响应", "未收到任何 chunk")
        except requests.exceptions.Timeout:
            self._fail("流式响应", "请求超时 (>30s)")
        except Exception as e:
            self._fail("流式响应", str(e)[:150])

    # ---- test 7: 文件路径 ----

    def test_file_paths(self):
        self._header("测试7: 文件路径")

        for name in ("PDF_FOLDER", "EXTRACT_DIR", "TEMPORAL_DIR"):
            path = getattr(self.config, name, "")
            if os.path.exists(path):
                self._ok(name, path)
            else:
                try:
                    os.makedirs(path, exist_ok=True)
                    self._ok(name, f"已创建: {path}")
                except Exception as e:
                    self._fail(name, str(e)[:100])

    # ---- internal ----

    def _do_chat_test(self, label, key, url, model, user_msg, ok_label):
        """通用 chat-completion 测试"""
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": 50,
            "stream": False,
        }

        print(f"  模型: {model}")
        print(f"  端点: {url}")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content:
                    self._ok(label, f"{ok_label}: {content[:100]}")
                else:
                    self._fail(label, "响应 content 为空")
            else:
                self._fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            self._fail(label, "请求超时 (>30s)")
        except Exception as e:
            self._fail(label, str(e)[:150])

    # ---- summary ----

    def print_summary(self):
        self._header("测试总结")

        total = len(self.test_results)
        passed = sum(1 for _, s, _ in self.test_results if s is True)
        failed = sum(1 for _, s, _ in self.test_results if s is False)
        skipped = sum(1 for _, s, _ in self.test_results if s is None)

        print(f"\n总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"跳过: {skipped}")

        if failed > 0:
            print("\n失败的测试:")
            for name, success, msg in self.test_results:
                if success is False:
                    print(f"  - {name}")
                    if msg:
                        print(f"    {msg}")

        print("\n" + "=" * 60)
        if failed == 0 and passed > 0:
            print("  [成功] 所有已执行的测试通过！")
        elif failed > 0:
            print("  [警告] 部分测试失败，请检查 config.json。")
        elif passed == 0:
            print("  [注意] 所有测试被跳过，请检查配置。")
        print("=" * 60 + "\n")

        return failed == 0

    # ---- run ----

    def run_all_tests(self):
        print("\n" + "=" * 60)
        print("  SDL Agent - 独立 API 配置测试")
        print("  测试每个模型的 API_KEY/URL 是否有效")
        print("=" * 60)

        self.test_config()
        time.sleep(0.3)
        self.test_talk_api()
        time.sleep(0.3)
        self.test_vl_api()
        time.sleep(0.3)
        self.test_experiment_api()
        time.sleep(0.3)
        self.test_embedding_api()
        time.sleep(0.3)
        self.test_streaming()
        time.sleep(0.3)
        self.test_file_paths()

        return self.print_summary()


def main():
    tester = APITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
