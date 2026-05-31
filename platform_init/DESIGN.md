# platform_init 统一启动验证模块 — 设计文档

日期: 2026-05-16

## 目标

将分散的启动验证逻辑（配置、API、流式、MQTT、算法、路径等）统一为一个模块，
支持三种运行模式：Flask 启动时自动（local）、命令行独立全量（full）、REST API。

## 检查项清单

| # | 类别 | 检查内容 | 预计耗时 | scope |
|---|------|----------|----------|-------|
| 1 | config | 4 个模型名是否配置、4 组 API_KEY/URL 是否可用、EXTRA_BODY JSON 是否合法 | < 0.1s | local |
| 2 | api | TALK / VL / EXPERIMENT / EMBEDDING 各发一次短请求验证连通性 | 8-30s | full |
| 3 | streaming | 实验模型是否支持 SSE streaming（复用 check_stream_capability） | 3-10s | full |
| 4 | hardware | MQTT broker 是否可达；REGISTRY.json 是否可解析、参数结构是否合法 | 2-10s | full |
| 5 | software | software/algorithms/ 下算法类可加载性、中文名、重复检测 | < 0.5s | local |
| 6 | paths | PDF_FOLDER / EXTRACT_DIR / TEMPORAL_DIR 是否存在或可创建 | < 0.1s | local |
| 7 | frontend | frontend/dist/index.html 存在；引用的 JS/CSS 文件存在 | < 0.1s | local |
| 8 | dependencies | 关键包版本是否兼容（rich>=14, paho-mqtt, flask 等） | < 0.5s | local |

## 架构

```
platform_init/platform_init.py          # 统一入口（本文件）
    ├── CheckResult                     # dataclass: name, category, status, message, detail, duration_ms
    ├── PlatformCheckRunner             # 核心类：收集检查项、运行、超时处理、汇总
    ├── 8 个 _check_*() 函数            # 每项检查独立函数，签名 () -> CheckResult
    ├── run_checks(scope="local")       # 便捷函数，供 app.py 调用
    └── if __name__ == "__main__":      # CLI 入口，argparse
```

## 调用方式

| 场景 | 入口 | scope |
|------|------|-------|
| app.py 启动 | `from platform_init.platform_init import run_checks; run_checks()` | local |
| CLI 全量 | `python platform_init/platform_init.py` | full |
| CLI 快速 | `python platform_init/platform_init.py --quick` | local |
| API 本地 | `GET /api/platform_check` | local |
| API 全量 | `POST /api/platform_check/full` | full |

## 输出格式

```json
{
  "checked_at": "2026-05-16T10:00:00",
  "scope": "local",
  "summary": {"total": 8, "pass": 6, "fail": 0, "warn": 1, "skip": 1},
  "results": [
    {"category": "config", "name": "模型名称配置", "status": "pass", "message": "...", "duration_ms": 2}
  ]
}
```

status: "pass" | "fail" | "warn" | "skip"
控制台输出: ✓ pass / ✗ fail / ⚠ warn / ○ skip

## 改动文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `platform_init/platform_init.py` | 新建 | 全部逻辑 |
| `app.py` | 修改 | 3 处：(1) 启动时调用 run_checks(local) (2) GET /api/platform_check (3) POST /api/platform_check/full |

不改动现有 platform_init 下的其他模块（check_stream_capability.py 等），通过 import 复用。

## app.py 集成

替换现有的单独 `ensure_capability_check(config)` 调用（第 125-131 行），改为：

```python
from platform_init.platform_init import run_checks
try:
    _platform_status = run_checks(scope="local")
    print(f"[platform_init] 本地检查完成: {_platform_status['summary']}")
except Exception as e:
    print(f"[platform_init] 检查失败（不影响正常功能）: {e}")
```

新增两个 API 路由：

```python
@app.route('/api/platform_check', methods=['GET'])
def platform_check():
    """本地检查（快速，无网络）"""
    return jsonify(run_checks(scope="local"))

@app.route('/api/platform_check/full', methods=['POST'])
def platform_check_full():
    """全量检查（含网络）"""
    return jsonify(run_checks(scope="full"))
```

## 各检查项实现要点

### 1. config — 配置完整性
- 读 Config()，遍历 MODEL_NAME_TALK/VL/EXPERIMENT_MODEL_NAME/EMBEDDING_MODEL
- 遍历 TALK/VL/EXPERIMENT/EMBEDDING 的 API_KEY/URL
- 尝试 JSON.parse 各 EXTRA_BODY
- 纯本地，无网络

### 2. api — API 连通性 (full only)
- 复用 LLMClient，4 个模型各发一次短请求（max_tokens=10）
- ThreadPoolExecutor 并发，单请求超时 15s
- 检查 content 或 reasoning_content 非空

### 3. streaming — 流式能力 (full only)
- 复用 check_stream_capability.test_model_streaming()
- 与现有的 model_capabilities.json 缓存兼容

### 4. hardware — MQTT + registry (full only)
- MQTT: 尝试 paho 连接，设 5s 超时
- Registry: 读 REGISTRY.json → json.loads → 检查每个 entry 的 name/description/params 完整性

### 5. software — 算法注册表
- 导入 SoftwareManager → 调用 list_algorithms()
- 检查是否有中文名、是否有重复 name
- 纯本地

### 6. paths — 数据目录
- os.path.exists(PDF_FOLDER, EXTRACT_DIR, TEMPORAL_DIR)
- 不存在尝试 os.makedirs
- 纯本地

### 7. frontend — 前端构建产物
- 检查 frontend/dist/index.html 存在
- 解析 <script src> / <link href>，验证对应文件存在
- 纯本地

### 8. dependencies — 关键依赖版本
- importlib.metadata.version() 获取 rich, paho-mqtt, flask, requests 版本
- rich >= 14.0（防 fastmcp 崩溃）、paho-mqtt >= 1.6
- 纯本地

## 超时与并发

- 单检查项默认超时: local=5s, full=30s
- 网络检查（api + streaming + MQTT）可并发，但第一版先顺序执行以简化错误处理
- 任一检查失败不阻塞后续
- 启动时检查必须 < 2s → 仅 local，无网络

## 与现有模块关系

- `check_stream_capability.py`: 复用 `test_model_streaming()`，不修改
- `update_registry.py`: 不涉及，独立工具
- `test/api_test/api_test.py`: 参考其 API 测试逻辑，但平台启动只做连通性检查（短请求），不做内容验证
