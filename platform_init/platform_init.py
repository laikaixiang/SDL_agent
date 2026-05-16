"""
平台启动统一验证/诊断模块。

三种运行模式:
- app.py 启动时自动运行 (scope="local", 仅本地检查, <1.5s)
- CLI 独立运行 (scope="full", 全量检查含网络)
- REST API (GET /api/platform_check, POST /api/platform_check/full)

使用::

    from platform_init.platform_init import run_checks

    result = run_checks(scope="local")   # 返回 dict
    print(result["summary"])             # {"total": 5, "pass": 5, "fail": 0, "warn": 0, "skip": 3}

CLI::

    python platform_init/platform_init.py           # 全量
    python platform_init/platform_init.py --quick   # 仅本地
"""

import sys
import os
import json
import time
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from datetime import datetime


# ── 数据结构 ────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    category: str
    name: str
    status: str           # pass | fail | warn | skip
    message: str = ""
    detail: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
        }


class PlatformCheckRunner:
    """收集检查项、运行、汇总结果。"""

    def __init__(self, config=None):
        self._config = config
        self._checks: List[tuple] = []  # (name, category, fn, scope)

    @property
    def config(self):
        if self._config is None:
            from core.config import Config
            self._config = Config()
        return self._config

    def register(self, category: str, name: str, fn: Callable[[], CheckResult],
                 scope: str = "full"):
        self._checks.append((category, name, fn, scope))

    def run(self, scope: str = "local", timeout_per_check: float = 30.0) -> dict:
        results = []
        checked_at = datetime.now().isoformat()

        for category, name, fn, check_scope in self._checks:
            if scope == "local" and check_scope == "full":
                results.append(CheckResult(
                    category=category, name=name, status="skip",
                    message="仅 full 模式执行", detail=""
                ))
                continue

            t0 = time.time()
            try:
                result = fn()
            except Exception as e:
                result = CheckResult(
                    category=category, name=name, status="fail",
                    message=str(e)[:200],
                    detail=""
                )
            result.duration_ms = round((time.time() - t0) * 1000, 1)
            results.append(result)

        summary = {
            "total": len(results),
            "pass": sum(1 for r in results if r.status == "pass"),
            "fail": sum(1 for r in results if r.status == "fail"),
            "warn": sum(1 for r in results if r.status == "warn"),
            "skip": sum(1 for r in results if r.status == "skip"),
        }

        return {
            "checked_at": checked_at,
            "scope": scope,
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }


# ── 检查函数 ────────────────────────────────────────────────────────────────────

def _check_config(cfg) -> CheckResult:
    """检查配置完整性（本地）。"""
    issues = []

    # 模型名称
    model_keys = {
        "MODEL_NAME_TALK": "对话模型",
        "MODEL_NAME_VL": "视觉模型",
        "EXPERIMENT_MODEL_NAME": "实验设计模型",
        "EMBEDDING_MODEL": "嵌入模型",
    }
    for key, label in model_keys.items():
        v = getattr(cfg, key, "")
        if not v:
            issues.append(f"{label} ({key}) 未配置")

    # API key/url — 每个模块独立检查，未独立配置但全局可用的 OK
    api_modules = ["TALK", "VL", "EXPERIMENT", "EMBEDDING"]
    for name in api_modules:
        key = getattr(cfg, f"{name}_API_KEY", "")
        url = getattr(cfg, f"{name}_API_URL", "")
        if key and url:
            continue  # OK
        # fallback to globals
        if not key and cfg.API_KEY:
            pass  # uses global
        elif not key:
            issues.append(f"{name}_API_KEY 未配置且全局 API_KEY 也不可用")
        if not url and cfg.API_URL:
            pass
        elif not url:
            issues.append(f"{name}_API_URL 未配置且全局 API_URL 也不可用")

    # EXTRA_BODY JSON 合法性
    for name in ("TALK", "VL", "EXPERIMENT"):
        raw = getattr(cfg, f"{name}_EXTRA_BODY", "")
        if raw:
            try:
                json.loads(raw)
            except json.JSONDecodeError:
                issues.append(f"{name}_EXTRA_BODY JSON 解析失败")

    if issues:
        return CheckResult(category="config", name="配置完整性", status="warn",
                           message=f"{len(issues)} 个问题", detail="; ".join(issues))
    return CheckResult(category="config", name="配置完整性", status="pass",
                       message="所有模型和 API 配置完整")


def _check_apis(cfg) -> CheckResult:
    """检查 4 组 API 连通性（并发，full only）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests as req_lib

    api_configs = [
        ("TALK", cfg.MODEL_NAME_TALK, cfg.TALK_API_KEY, cfg.TALK_API_URL,
         cfg.get_extra_body("TALK"), False),
        ("VL", cfg.MODEL_NAME_VL, cfg.VL_API_KEY, cfg.VL_API_URL,
         cfg.get_extra_body("VL"), True),
        ("EXPERIMENT", cfg.EXPERIMENT_MODEL_NAME, cfg.EXPERIMENT_API_KEY,
         cfg.EXPERIMENT_API_URL, cfg.get_extra_body("EXPERIMENT"), False),
    ]

    # 1x1 白色 PNG（VL 模型需要图片消息）
    TINY_PNG = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )

    def test_one(name, model, key, url, extra_body, is_vision):
        if not key:
            return name, "skip", "API_KEY 为空"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if is_vision:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}},
                    {"type": "text", "text": "Say hi in one word."},
                ]
            }]
        else:
            messages = [{"role": "user", "content": "Say hi in one word."}]

        body = {"model": model, "messages": messages, "max_tokens": 10}
        if extra_body:
            body.update(extra_body)

        try:
            resp = req_lib.post(url, headers=headers, json=body, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                content = (data.get("choices", [{}])[0]
                           .get("message", {})
                           .get("content", ""))
                if not content:
                    content = (data.get("choices", [{}])[0]
                               .get("message", {})
                               .get("reasoning_content", ""))
                if content:
                    return name, "pass", f"响应正常 ({len(content)} chars)"
                return name, "warn", "content 和 reasoning_content 均为空"
            return name, "fail", f"HTTP {resp.status_code}: {resp.text[:150]}"
        except req_lib.exceptions.Timeout:
            return name, "fail", "请求超时 (>15s)"
        except Exception as e:
            return name, "fail", str(e)[:150]

    # 并发 3 个 LLM API
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(test_one, *ac): ac[0] for ac in api_configs}
        for fut in as_completed(futures):
            name, status, msg = fut.result()
            results[name] = (status, msg)

    # embedding 串行
    emb_name = "EMBEDDING"
    if not cfg.EMBEDDING_API_KEY:
        results[emb_name] = ("skip", "EMBEDDING_API_KEY 为空")
    else:
        try:
            headers = {"Authorization": f"Bearer {cfg.EMBEDDING_API_KEY}",
                       "Content-Type": "application/json"}
            payload = {"model": cfg.EMBEDDING_MODEL, "input": "hello"}
            resp = req_lib.post(cfg.EMBEDDING_API_URL, headers=headers,
                                json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                dim = len(data.get("data", [{}])[0].get("embedding", []))
                results[emb_name] = ("pass", f"维度={dim}")
            else:
                results[emb_name] = ("fail", f"HTTP {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            results[emb_name] = ("fail", str(e)[:150])

    parts = []
    all_ok = True
    has_fail = False
    for name in ["TALK", "VL", "EXPERIMENT", "EMBEDDING"]:
        status, msg = results.get(name, ("fail", "未执行"))
        icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "○"}.get(status, "?")
        parts.append(f"{icon} {name}: {msg}")
        if status == "fail":
            has_fail = True
        if status != "pass":
            all_ok = False

    status = "pass" if all_ok else ("fail" if has_fail else "warn")
    return CheckResult(category="api", name="API 连通性", status=status,
                       message="; ".join(parts), detail="")


def _check_streaming(cfg) -> CheckResult:
    """检查流式能力（full only，复用已有模块）。"""
    import importlib.util

    try:
        # platform_init/ 无 __init__.py，用 importlib 按文件路径加载
        _mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "check_stream_capability.py")
        _spec = importlib.util.spec_from_file_location("_check_stream_capability", _mod_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        test_model_streaming = _mod.test_model_streaming
    except Exception:
        return CheckResult(category="streaming", name="流式传输", status="warn",
                           message="无法导入 check_stream_capability 模块")

    model = cfg.EXPERIMENT_MODEL_NAME
    result = test_model_streaming(model, cfg.EXPERIMENT_API_KEY,
                                  cfg.EXPERIMENT_API_URL)

    if result.get("error"):
        return CheckResult(category="streaming", name="流式传输",
                           status="fail", message=result["error"])

    if result.get("supports_streaming"):
        return CheckResult(category="streaming", name="流式传输",
                           status="pass",
                           message=f'{model} 支持流式 '
                                   f'(chunks={result["chunks_received"]}, '
                                   f'latency={result["latency_ms"]}ms)')
    return CheckResult(category="streaming", name="流式传输",
                       status="warn",
                       message=f'{model} 不支持流式 '
                               f'(chunks={result["chunks_received"]})')


def _check_hardware(cfg) -> CheckResult:
    """检查 MQTT 连接 + 工具注册表（full only）。"""
    parts = []

    # 1. MQTT broker 连通性
    try:
        from hardware.mqtt.client import get_mqtt_client
        import threading

        connected = [False]
        mqtt_error = [None]

        def do_connect():
            try:
                client = get_mqtt_client()
                connected[0] = client.is_connected
            except Exception as e:
                mqtt_error[0] = str(e)

        t = threading.Thread(target=do_connect, daemon=True)
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            parts.append("✗ MQTT: 连接超时 (>10s)")
        elif mqtt_error[0]:
            parts.append(f"✗ MQTT: {mqtt_error[0][:100]}")
        elif connected[0]:
            parts.append("✓ MQTT: broker 可达")
        else:
            parts.append("✗ MQTT: broker 不可达")
    except ImportError as e:
        parts.append(f"○ MQTT: 无法导入硬件模块 ({e})")
    except Exception as e:
        parts.append(f"✗ MQTT: {str(e)[:100]}")

    # 2. 工具注册表完整性
    try:
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "hardware", "tools", "REGISTRY.json"
        )
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            tool_count = len(reg)
            invalid = []
            for tool_name, tool_def in reg.items():
                if not isinstance(tool_def, dict):
                    invalid.append(f"{tool_name}: 非字典类型")
                    continue
                if "name" not in tool_def:
                    invalid.append(f"{tool_name}: 缺少 name")
                if "description" not in tool_def:
                    invalid.append(f"{tool_name}: 缺少 description")
                if "params" not in tool_def or not isinstance(tool_def["params"], dict):
                    invalid.append(f"{tool_name}: params 格式错误")

            if invalid:
                parts.append(f"⚠ Registry: {tool_count} 个工具, "
                             f"{len(invalid)} 个有问题 ({'; '.join(invalid[:3])})")
            else:
                parts.append(f"✓ Registry: {tool_count} 个工具, 格式正常")
        else:
            parts.append("✗ Registry: REGISTRY.json 不存在")
    except Exception as e:
        parts.append(f"✗ Registry: {str(e)[:100]}")

    has_fail = any("✗" in p for p in parts)
    has_warn = any("⚠" in p for p in parts)
    status = "fail" if has_fail else ("warn" if has_warn else "pass")
    return CheckResult(category="hardware", name="硬件 / MQTT / Registry",
                       status=status, message="; ".join(parts))


def _check_software(cfg) -> CheckResult:
    """检查软件算法注册表（本地）。"""
    try:
        from core.software_manager import SoftwareManager
        mgr = SoftwareManager()
        algos = mgr.list_algorithms()

        if not algos:
            return CheckResult(category="software", name="软件算法", status="warn",
                               message="未发现任何已注册算法")

        issues = []
        names_seen = set()
        for a in algos:
            name = (a.get("name") or a.get("chinese_name") or str(a))
            if not a.get("chinese_name"):
                issues.append(f"{name}: 缺少 chinese_name")
            if a.get("name") in names_seen:
                issues.append(f"{name}: 重复注册")
            if a.get("name"):
                names_seen.add(a.get("name"))

        if issues:
            return CheckResult(category="software", name="软件算法",
                               status="warn",
                               message=f"{len(algos)} 个算法, {len(issues)} 个问题",
                               detail="; ".join(issues[:5]))
        return CheckResult(category="software", name="软件算法", status="pass",
                           message=f"{len(algos)} 个算法, 全部正常")
    except ImportError as e:
        return CheckResult(category="software", name="软件算法", status="skip",
                           message=f"无法导入软件模块: {e}")
    except Exception as e:
        return CheckResult(category="software", name="软件算法", status="fail",
                           message=str(e)[:200])


def _check_paths(cfg) -> CheckResult:
    """检查数据目录是否存在/可创建（本地）。"""
    issues = []
    ok = []

    for key, label in [("PDF_FOLDER", "PDF 文件夹"),
                       ("EXTRACT_DIR", "提取输出目录"),
                       ("TEMPORAL_DIR", "临时数据目录")]:
        path = getattr(cfg, key, "")
        if not path:
            issues.append(f"{label}: 未配置路径")
            continue
        if os.path.exists(path):
            ok.append(f"{label}: {path}")
        else:
            try:
                os.makedirs(path, exist_ok=True)
                ok.append(f"{label}: 已创建 {path}")
            except Exception as e:
                issues.append(f"{label}: 不存在且无法创建 ({e})")

    if issues:
        return CheckResult(category="paths", name="文件路径", status="warn",
                           message=f"{len(ok)} OK, {len(issues)} 问题",
                           detail="; ".join(issues))
    return CheckResult(category="paths", name="文件路径", status="pass",
                       message="; ".join(ok))


def _check_frontend(cfg) -> CheckResult:
    """检查前端构建产物（本地）。"""
    dist_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "dist"
    )
    index_html = os.path.join(dist_dir, "index.html")

    if not os.path.exists(index_html):
        return CheckResult(category="frontend", name="前端构建", status="warn",
                           message="frontend/dist/index.html 不存在，前端将不可用")

    with open(index_html, "r", encoding="utf-8") as f:
        html = f.read()

    # 提取 <script src> 和 <link href>，排除 favicon.ico（非关键）
    assets_ok = True
    missing = []
    for pattern, attr in [(r'<script[^>]+src="([^"]+)"', "src"),
                           (r'<link[^>]+href="([^"]+)"', "href")]:
        for m in re.finditer(pattern, html):
            url = m.group(1)
            if url.startswith("http") or url.startswith("//"):
                continue
            if url.endswith("favicon.ico") or url.endswith("favicon.png"):
                continue  # non-critical, often missing
            # 去掉 leading /v2-static/ 或 /
            file_path = url.lstrip("/")
            if file_path.startswith("v2-static/"):
                file_path = file_path[len("v2-static/"):]
            full = os.path.join(dist_dir, file_path)
            if not os.path.exists(full):
                missing.append(file_path)
                assets_ok = False

    if assets_ok:
        return CheckResult(category="frontend", name="前端构建", status="pass",
                           message="index.html 及所有引用的 assets 均存在")
    return CheckResult(category="frontend", name="前端构建", status="warn",
                       message=f"缺少 {len(missing)} 个文件",
                       detail=", ".join(missing[:5]))


def _check_dependencies(cfg) -> CheckResult:
    """检查关键依赖版本是否兼容（本地）。"""
    issues = []
    ok = []

    checks = [
        ("rich", (14, 0), "fastmcp 兼容性，需 >= 14.0"),
        ("paho-mqtt", (1, 6), "MQTT 客户端"),
        ("flask", (3, 0), "Web 框架"),
        ("requests", (2, 0), "HTTP 库"),
    ]

    for pkg, min_ver, note in checks:
        try:
            from importlib.metadata import version as pkg_version
            v = pkg_version(pkg)
            parts = tuple(int(x) for x in v.split(".")[:2])
            if parts >= min_ver:
                ok.append(f"{pkg}=={v}")
            else:
                issues.append(f"{pkg}=={v} (< {min_ver[0]}.{min_ver[1]}) — {note}")
        except Exception:
            issues.append(f"{pkg}: 未安装 — {note}")

    if issues:
        return CheckResult(category="dependencies", name="依赖版本",
                           status="warn",
                           message=f"{len(ok)} OK, {len(issues)} 问题",
                           detail="; ".join(issues))
    return CheckResult(category="dependencies", name="依赖版本",
                       status="pass", message="; ".join(ok))


# ── 公开 API ────────────────────────────────────────────────────────────────────

def run_checks(scope: str = "local", config=None) -> dict:
    """
    运行平台启动验证。

    Args:
        scope: "local" (仅本地, <1.5s) | "full" (含网络, 15-30s)
        config: Config 实例，不传则自动创建

    Returns:
        {"checked_at": str, "scope": str, "summary": {...}, "results": [...]}
    """
    runner = PlatformCheckRunner(config)

    # 注册检查项 (category, name, fn, scope)
    runner.register("config", "配置完整性",
                    lambda: _check_config(runner.config), "local")
    runner.register("api", "API 连通性",
                    lambda: _check_apis(runner.config), "full")
    runner.register("streaming", "流式传输",
                    lambda: _check_streaming(runner.config), "full")
    runner.register("hardware", "硬件 / MQTT / Registry",
                    lambda: _check_hardware(runner.config), "full")
    runner.register("software", "软件算法",
                    lambda: _check_software(runner.config), "local")
    runner.register("paths", "文件路径",
                    lambda: _check_paths(runner.config), "local")
    runner.register("frontend", "前端构建",
                    lambda: _check_frontend(runner.config), "local")
    runner.register("dependencies", "依赖版本",
                    lambda: _check_dependencies(runner.config), "local")

    return runner.run(scope)


def _print_results(result: dict) -> None:
    """打印结果到控制台。"""
    STATUS_ICON = {"pass": "[PASS]", "fail": "[FAIL]", "warn": "[WARN]", "skip": "[SKIP]"}

    print(f"\n{'='*60}")
    print(f"  Platform Check  [{result['scope']}]  {result['checked_at'][:19]}")
    print(f"{'='*60}")

    for r in result["results"]:
        icon = STATUS_ICON.get(r["status"], "[????]")
        line = f"  {icon} [{r['category']}] {r['name']}"
        if r["duration_ms"] > 0:
            line += f"  ({r['duration_ms']:.0f}ms)"
        print(line)
        if r["message"]:
            print(f"      {r['message']}")
        if r["detail"]:
            for d in r["detail"].split("; "):
                print(f"        - {d.strip()}")

    s = result["summary"]
    print(f"\n  Total: {s['total']} | Pass: {s['pass']} | Fail: {s['fail']} | "
          f"Warn: {s['warn']} | Skip: {s['skip']}")
    print(f"{'='*60}\n")


# ── CLI 入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Windows: fix stdout encoding for emoji / Chinese characters
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="平台启动验证")
    parser.add_argument("--quick", action="store_true",
                        help="仅本地检查（无网络）")
    parser.add_argument("--json", action="store_true",
                        help="输出 JSON（非 human-readable）")
    args = parser.parse_args()

    # 确保项目根在 sys.path 中
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    scope = "local" if args.quick else "full"
    result = run_checks(scope=scope)

    if args.json:
        # Windows 终端 emoji 问题处理
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_results(result)

    # 有 fail 则非零退出
    if result["summary"]["fail"] > 0:
        sys.exit(1)
