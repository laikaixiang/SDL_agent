"""
平台启动时检测模型流式能力的模块。

调用方式：app.py 启动时 import 并调用 ensure_capability_check(config)
结果缓存到 model_capabilities.json，同一模型无需重复检测。

作为平台启动流程之一，后续可扩展为统一的平台能力检测入口。
"""

import json
import time
import os
from typing import Dict, Any

CAPABILITY_FILE = os.path.join(os.path.dirname(__file__), "model_capabilities.json")


def test_model_streaming(model_name: str, api_key: str, api_url: str,
                         timeout: int = 15) -> Dict[str, Any]:
    """
    检测单个模型是否支持流式响应。

    发一个极短 stream=True 请求（max_tokens=50），
    统计收到 >=2 个独立 data chunk → 判定支持流式。

    Returns:
        {
            "supports_streaming": bool,
            "chunks_received": int,
            "latency_ms": float,
            "tested_at": "ISO",
            "error": str | null
        }
    """
    import requests

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 50,
        "temperature": 0,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.time()
    chunks = 0
    error = None

    try:
        resp = requests.post(api_url, headers=headers, json=payload,
                             timeout=timeout, stream=True)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: ") and decoded[6:].strip() != "[DONE]":
                    chunks += 1
    except Exception as e:
        error = str(e)

    latency = (time.time() - start) * 1000
    return {
        "supports_streaming": chunks >= 2,
        "chunks_received": chunks,
        "latency_ms": round(latency, 1),
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "error": error,
    }


def load_capabilities() -> Dict[str, Any]:
    """读取已有的能力文件，不存在则返回空字典。"""
    if os.path.exists(CAPABILITY_FILE):
        with open(CAPABILITY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_capabilities(data: Dict[str, Any]):
    """写入能力文件。"""
    os.makedirs(os.path.dirname(CAPABILITY_FILE), exist_ok=True)
    with open(CAPABILITY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_capability_check(config) -> Dict[str, Any]:
    """
    确保当前模型已被检测。若已检测过同一模型则跳过；否则运行检测。

    此函数在 app.py 启动时调用一次。

    Args:
        config: core.config.Config 实例

    Returns:
        capabilities 字典，前端可通过 /api/capabilities 获取
    """
    caps = load_capabilities()
    model = config.EXPERIMENT_MODEL_NAME

    # 同一模型已检测过 → 跳过
    if model in caps.get("models", {}):
        entry = caps["models"][model]
        print(f"[platform_init] 模型 '{model}' 已有缓存: "
              f"streaming={entry.get('supports_streaming')} "
              f"(chunks={entry.get('chunks_received')}, "
              f"tested_at={entry.get('tested_at')})")
        return caps

    print(f"[platform_init] 检测模型流式能力: {model} ...")
    result = test_model_streaming(model, config.API_KEY, config.API_URL)

    caps.setdefault("models", {})[model] = result
    caps["default"] = {
        "streaming_enabled": result["supports_streaming"],
        "last_model": model,
        "last_check": result["tested_at"],
    }
    save_capabilities(caps)

    status = "支持 ✓" if result["supports_streaming"] else "不支持 ✗"
    print(f"[platform_init] 流式检测完成: {status} "
          f"(chunks={result['chunks_received']}, latency={result['latency_ms']}ms)")
    if result.get("error"):
        print(f"[platform_init] 检测详情: {result['error']}")

    return caps


if __name__ == "__main__":
    # 独立运行：手动检测当前配置的模型
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.config import Config
    caps = ensure_capability_check(Config())
    print("\n结果:")
    print(json.dumps(caps, ensure_ascii=False, indent=2))
