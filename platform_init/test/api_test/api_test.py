"""
独立API测试工具 - 使用工程代码中的 LLMClient / AdaptiveStreamHandler 进行测试

功能：
1. 测试配置完整性（所有模型 + API密钥 + EXTRA_BODY）
2. 测试 TALK API 密钥是否有效（通过 LLMClient.call_api）
3. 测试 VL API 密钥是否有效（通过 LLMClient.call_api，视觉消息）
4. 测试 EXPERIMENT API 密钥是否有效（通过 LLMClient.call_api）
5. 测试 EMBEDDING API 密钥是否有效（裸 requests，无 LLMClient 封装）
6. 测试流式响应是否正常（通过 AdaptiveStreamHandler）
7. 文件路径检查

使用方法：
    python platform_init/test/api_test/api_test.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import requests
import json
import time

# 直接导入 Config，避免触发 core/__init__.py 的所有导入
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from core.config import Config
    from core.llm_client import LLMClient
except ImportError as e:
    print(f"导入 core 模块出错: {e}，使用备用导入方法...")
    import importlib.util

    def _load_module(name, rel_path):
        spec = importlib.util.spec_from_file_location(name, os.path.join(_PROJECT_ROOT, rel_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    config_module = _load_module("config", "core/config.py")
    Config = config_module.Config
    llm_module = _load_module("llm_client", "core/llm_client.py")
    LLMClient = llm_module.LLMClient

# 1x1 白色像素 PNG
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)


class APITester:
    """API测试器 — 使用工程 LLMClient 调用，测试每个模型的独立 API_KEY/URL 是否有效"""

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

    def _make_llm_client(self, model_type: str):
        """按模型类型创建 LLMClient，自动注入正确的 api_key / api_url / extra_body"""
        c = self.config
        key = getattr(c, f"{model_type.upper()}_API_KEY")
        url = getattr(c, f"{model_type.upper()}_API_URL")
        extra = c.get_extra_body(model_type)
        return LLMClient(api_key=key, api_url=url, extra_body=extra)

    # ---- test 1: 配置完整性 ----

    def test_config(self):
        self._header("测试1: 配置完整性")

        for key in ("MODEL_NAME_TALK", "MODEL_NAME_VL", "EXPERIMENT_MODEL_NAME", "EMBEDDING_MODEL"):
            v = getattr(self.config, key, "")
            if v:
                self._ok(key, v)
            else:
                self._fail(key, "未设置")

        for name in ("TALK", "VL", "EXPERIMENT", "EMBEDDING"):
            key = getattr(self.config, f"{name}_API_KEY")
            url = getattr(self.config, f"{name}_API_URL")
            if key and url:
                self._ok(f"{name}_API_KEY/URL", f"key={key[:15]}... url={url}")
            elif not key and not url:
                if self.config.API_KEY and self.config.API_URL:
                    self._ok(f"{name}_API_KEY/URL", "未单独设置，回退到全局")
                else:
                    self._fail(f"{name}_API_KEY/URL", "未设置且全局也不可用")
            else:
                self._fail(f"{name}_API_KEY/URL", "KEY 和 URL 必须同时设置")

        # EXTRA_BODY
        for name in ("TALK", "VL", "EXPERIMENT"):
            raw = getattr(self.config, f"{name}_EXTRA_BODY")
            if raw:
                try:
                    parsed = json.loads(raw)
                    self._ok(f"{name}_EXTRA_BODY", f"已配置: {json.dumps(parsed, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    self._fail(f"{name}_EXTRA_BODY", f"JSON 解析失败: {raw[:80]}")
            else:
                self._ok(f"{name}_EXTRA_BODY", "未设置（不需要供应商特有参数）")

    # ---- test 2: TALK API (通过 LLMClient) ----

    def test_talk_api(self):
        self._header("测试2: TALK API (LLMClient.call_api)")

        if not self.config.TALK_API_KEY:
            self._skip("TALK API", "TALK_API_KEY 为空")
            return

        client = self._make_llm_client("TALK")
        model = self.config.MODEL_NAME_TALK

        print(f"  模型: {model}")
        print(f"  端点: {client.get_api_url()}")

        result = client.call_api(
            model=model,
            messages=[{"role": "user", "content": "请用一句话介绍你自己"}],
            max_tokens=100,
            stream=False,
        )

        if not result:
            self._fail("TALK API", "call_api 返回 None")
            return

        content = result['choices'][0]['message'].get('content', '')
        if not content:
            content = result['choices'][0]['message'].get('reasoning_content', '')
        if content:
            self._ok("TALK API", f"响应: {content[:100]}")
        else:
            self._fail("TALK API", "content 和 reasoning_content 均为空")

    # ---- test 3: VL API (通过 LLMClient) ----

    def test_vl_api(self):
        self._header("测试3: VL API (LLMClient.call_api, 视觉消息)")

        if not self.config.VL_API_KEY:
            self._skip("VL API", "VL_API_KEY 为空")
            return

        client = self._make_llm_client("VL")
        model = self.config.MODEL_NAME_VL

        print(f"  模型: {model}")
        print(f"  端点: {client.get_api_url()}")

        result = client.call_api(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_BASE64}"}},
                    {"type": "text", "text": "描述这张图片"},
                ]
            }],
            max_tokens=30,
            stream=False,
        )

        if not result:
            self._fail("VL API", "call_api 返回 None")
            return

        content = result['choices'][0]['message'].get('content', '')
        if not content:
            content = result['choices'][0]['message'].get('reasoning_content', '')
        if content:
            self._ok("VL API", f"响应: {content[:100]}")
        else:
            self._fail("VL API", "content 和 reasoning_content 均为空")

    # ---- test 4: EXPERIMENT API (通过 LLMClient) ----

    def test_experiment_api(self):
        self._header("测试4: EXPERIMENT API (LLMClient.call_api)")

        if not self.config.EXPERIMENT_API_KEY:
            self._skip("EXPERIMENT API", "EXPERIMENT_API_KEY 为空")
            return

        client = self._make_llm_client("EXPERIMENT")
        model = self.config.EXPERIMENT_MODEL_NAME

        print(f"  模型: {model}")
        print(f"  端点: {client.get_api_url()}")

        result = client.call_api(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence"}],
            max_tokens=100,
            stream=False,
        )

        if not result:
            self._fail("EXPERIMENT API", "call_api 返回 None")
            return

        content = result['choices'][0]['message'].get('content', '')
        if not content:
            content = result['choices'][0]['message'].get('reasoning_content', '')
        if content:
            self._ok("EXPERIMENT API", f"响应: {content[:100]}")
        else:
            self._fail("EXPERIMENT API", "content 和 reasoning_content 均为空")

    # ---- test 5: EMBEDDING API (裸 requests) ----

    def test_embedding_api(self):
        self._header("测试5: EMBEDDING API (裸 requests)")

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
        payload = {"model": model, "input": "Hello, embedding test."}

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

    # ---- test 6: 流式响应 (通过 AdaptiveStreamHandler) ----

    def test_streaming(self):
        self._header("测试6: 流式响应 (AdaptiveStreamHandler)")

        if not self.config.TALK_API_KEY:
            self._skip("流式响应", "TALK_API_KEY 为空")
            return

        client = self._make_llm_client("TALK")
        from core.adaptive_stream import AdaptiveStreamHandler
        handler = AdaptiveStreamHandler(self.config, client)
        model = self.config.MODEL_NAME_TALK

        print(f"  模型: {model}  (stream=True)")

        gen = handler.generate_streaming_response("数到5", model=model)
        chunks = 0
        full = ""
        try:
            for chunk in gen:
                if chunk.startswith("\n[API错误") or chunk.startswith("\n[请求失败"):
                    self._fail("流式响应", chunk)
                    return
                full += chunk
                chunks += 1
        except Exception as e:
            self._fail("流式响应", str(e)[:100])
            return

        if chunks > 0:
            self._ok("流式响应", f"{chunks} 个 chunk, 内容: {full[:100]}")
        else:
            self._fail("流式响应", "未收到任何 chunk")

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
        print("  SDL Agent - API 配置测试 (使用工程 LLMClient)")
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
