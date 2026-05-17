"""
自动数据分析流水线 (software/auto_analyze.py)
=============================================

纯 Python 调用 LLM 实现"智能选算法 + 读数据 + 运行 + 保存结果"的全流程。
不依赖任何 Agent 框架，与 prompt_template.py 采用相同的直接 LLM 调用模式。

算法注册表与执行由调用方（SoftwareManager）提供，避免重复初始化 SoftwareController。

工作流：
    Step 1  读取 CSV 列名
            ↓
    Step 2  调用 LLM（列名 + 算法描述 + 读取函数描述）
            → LLM 返回 JSON: {algorithm, read_function, read_params, reasoning}
            ↓
    Step 3  Python 通过 READER_REGISTRY 动态调用读取函数
            ↓
    Step 4  Python 调用 run_fn(algorithm, data) 执行算法
            ↓
    Step 5  保存结果到 results/ 目录（覆盖写 + 时间戳存档）
            ↓
    Step 6  通过回调推送 SSE 消息

对外接口：
    from software.auto_analyze import run_pipeline

    run_pipeline(
        csv_path  = "temporal/extraction.csv",
        send_msg  = callback,           # (msg_type, data) → None
        algorithms = [...],             # SoftwareManager.list_algorithms() 的返回值
        run_fn    = software_manager.run_algorithm,  # (name, data, params) → dict
    )
"""

import os
import sys
import json
import re
import requests
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 复用 API 配置
try:
    # 直接导入 config 模块，避免 core.__init__ 中的 pydantic_ai 依赖
    import sys
    import os
    _config_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'config.py')
    if os.path.exists(_config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", _config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        _API_KEY = config_module.Config.TALK_API_KEY
        _API_URL = config_module.Config.TALK_API_URL  # 已包含完整路径，不需要拼接
        _MODEL   = config_module.Config.MODEL_NAME_TALK
    else:
        raise ImportError("Config file not found")
except Exception as e:
    # 备用配置（仅在无法加载 config.py 时使用）
    _API_KEY = ""
    _API_URL = "https://api.longcat.chat/openai/v1/chat/completions"
    _MODEL   = "Qwen/Qwen2.5-7B-Instruct"

# 结果保存目录（项目根目录下的 results/）
_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


# ==============================================================================
# LLM 提示词（已迁移至 prompts/data_analysis/）
# ==============================================================================


# ==============================================================================
# 内部工具
# ==============================================================================

def _find_csv_file(csv_path: str) -> str:
    """
    智能查找 CSV 文件，支持多种路径格式和自动搜索

    Args:
        csv_path: 用户提供的文件路径（可能是相对路径、文件名或不存在）

    Returns:
        找到的文件绝对路径

    Raises:
        FileNotFoundError: 找不到文件时抛出，包含详细的搜索信息
    """
    # 1. 直接路径存在
    if os.path.exists(csv_path):
        return os.path.abspath(csv_path)

    # 2. 尝试在常见目录中查找
    search_dirs = [
        "temporal",
        "results",
        "extract",
        ".",
        os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
    ]

    filename = os.path.basename(csv_path)

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        candidate = os.path.join(search_dir, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    # 3. 在 temporal 目录中查找所有 CSV 文件（模糊匹配）
    temporal_dir = "temporal"
    if os.path.exists(temporal_dir):
        csv_files = [f for f in os.listdir(temporal_dir) if f.endswith('.csv')]
        if csv_files:
            # 优先匹配 extraction.csv
            if 'extraction.csv' in csv_files:
                return os.path.abspath(os.path.join(temporal_dir, 'extraction.csv'))
            # 返回第一个找到的 CSV
            return os.path.abspath(os.path.join(temporal_dir, csv_files[0]))

    # 4. 未找到，抛出详细错误
    searched = ", ".join([d for d in search_dirs if os.path.exists(d)])
    raise FileNotFoundError(
        f"找不到文件 '{csv_path}'。已搜索目录: {searched}。"
        f"请确保文件存在或将 CSV 文件放入 temporal/ 目录。"
    )


def _try_alternative_readers(csv_path: str, failed_reader: str, read_params: dict) -> tuple:
    """
    当指定的读取函数失败时，尝试其他读取函数

    Args:
        csv_path: CSV 文件路径
        failed_reader: 失败的读取函数名
        read_params: 原始读取参数

    Returns:
        (data, reader_name) 或 (None, None) 如果所有方法都失败
    """
    from software.readfile import READER_REGISTRY

    # 定义备选读取顺序
    fallback_order = [
        ("read_numeric_columns", {}),
        ("read_as_columns_dict", {}),
    ]

    for reader_name, params in fallback_order:
        if reader_name == failed_reader:
            continue
        try:
            reader_fn = READER_REGISTRY[reader_name]
            data = reader_fn(csv_path, **params)
            if data:  # 确保读取到了数据
                return data, reader_name
        except Exception:
            continue

    return None, None


def _call_llm(system_prompt: str, user_message: str) -> str:
    """调用 LLM，返回模型回复文本"""
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type" : "application/json",
    }
    payload = {
        "model"      : _MODEL,
        "messages"   : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.1,
    }
    if config_module.Config.MAX_TOKENS is not None:
        payload["max_tokens"] = config_module.Config.MAX_TOKENS
    # merge TALK extra_body
    try:
        _talk_raw = config_module.Config.TALK_EXTRA_BODY
        if _talk_raw:
            import json as _ej
            payload.update(_ej.loads(_talk_raw))
    except Exception:
        pass
    resp = requests.post(_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    if not content:
        content = message.get("reasoning_content", "")
    return content.strip()


def _strip_json(text: str) -> str:
    """去除 LLM 回复中可能出现的 markdown 代码块"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _build_algorithms_desc(algorithms: list) -> str:
    """将算法列表格式化为 LLM 提示词"""
    lines = []
    for algo in algorithms:
        lines.append(f"- {algo['name']}: {algo.get('description', '')}")
    return "\n".join(lines) if lines else "（无可用算法）"


def _extract_result_summary(result: dict, max_items: int = 8) -> dict:
    """
    从算法结果中递归提取顶层数值字段作为摘要（供前端展示）

    Args:
        result   : 算法 run() 返回的 result 字典
        max_items: 最多提取几个字段

    Returns:
        {"字段名": 数值, ...}
    """
    summary = {}

    def _collect(d: dict, prefix: str = ""):
        for k, v in d.items():
            if len(summary) >= max_items:
                return
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[full_key] = round(v, 6)
            elif isinstance(v, dict):
                _collect(v, full_key)

    _collect(result)
    return summary


def _save_result(csv_path: str, algorithm: str, reasoning: str, result: dict) -> str:
    """
    将完整结果写入 results/ 目录：
      - 覆盖写：results/analysis_{algorithm}.json（始终是最新结果）
      - 时间戳存档：results/analysis_{algorithm}_{timestamp}.json（永久保留）

    Args:
        csv_path : 来源 CSV 路径
        algorithm: 使用的算法名称
        reasoning: LLM 的选择理由
        result   : 算法原始返回结果

    Returns:
        时间戳存档文件的绝对路径
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    payload = {
        "timestamp" : datetime.now().isoformat(timespec="seconds"),
        "csv_source": csv_path,
        "algorithm" : algorithm,
        "reasoning" : reasoning,
        "result"    : result,
    }

    # 覆盖写（最新结果）
    latest_path = os.path.join(_RESULTS_DIR, f"analysis_{algorithm}.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 时间戳存档（历史记录）
    archive_path = os.path.join(_RESULTS_DIR, f"analysis_{algorithm}_{timestamp}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return archive_path


# ==============================================================================
# 主流水线
# ==============================================================================

def run_pipeline(
    csv_path  : str,
    send_msg,
    algorithms: list,
    run_fn,
) -> None:
    """
    纯 Python LLM 自动分析流水线（无 Agent 框架）

    Args:
        csv_path  : CSV 文件路径（如 "temporal/extraction.csv"）
        send_msg  : SSE 推送回调，签名为 send_msg(msg_type: str, data: Any)
        algorithms: 算法元数据列表，由 SoftwareManager.list_algorithms() 提供
        run_fn    : 算法执行函数，签名为 run_fn(name, data, params) → dict
                    由 SoftwareManager.run_algorithm 提供
    """
    from software.readfile import (
        read_column_names, READER_REGISTRY, FUNCTIONS_DESCRIPTION
    )

    # ---- Step 1: 智能查找并读取列名 ----
    send_msg("info", "正在查找并读取 CSV 文件...")

    try:
        csv_path = _find_csv_file(csv_path)
        send_msg("info", f"已找到文件: {csv_path}")
    except FileNotFoundError as e:
        send_msg("complete", {"error": str(e)})
        return

    try:
        columns = read_column_names(csv_path)
    except Exception as e:
        send_msg("complete", {"error": f"读取列名失败: {e}"})
        return

    if not columns:
        send_msg("complete", {"error": "CSV 文件为空或无列名"})
        return

    send_msg("info", f"已识别 {len(columns)} 列：{', '.join(columns)}")

    # ---- Step 2: 调用 LLM 分析 ----
    send_msg("progress", "大模型正在分析数据结构，请稍候...")

    algos_desc = _build_algorithms_desc(algorithms)

    from prompts import create_prompt_manager
    pm = create_prompt_manager(lang='zh')

    user_msg = pm.get(
        "data_analysis_user",
        csv_path=csv_path,
        columns=json.dumps(columns, ensure_ascii=False),
        algorithms_desc=algos_desc,
        functions_desc=FUNCTIONS_DESCRIPTION,
    )

    try:
        raw_response = _call_llm(pm.get("data_analysis_system"), user_msg)
        spec = json.loads(_strip_json(raw_response))
    except json.JSONDecodeError as e:
        send_msg("complete", {"error": f"LLM 返回格式无效，JSON 解析失败: {e}"})
        return
    except Exception as e:
        send_msg("complete", {"error": f"LLM 调用失败: {e}"})
        return

    algorithm     = spec.get("algorithm", "")
    read_function = spec.get("read_function", "")
    read_params   = spec.get("read_params", {})
    reasoning     = spec.get("reasoning", "")

    # 校验 LLM 返回是否合法
    algo_names = [a["name"] for a in algorithms]
    if algorithm not in algo_names:
        send_msg("complete", {
            "error": f"LLM 指定了未知算法 '{algorithm}'，可用：{algo_names}"
        })
        return
    if read_function not in READER_REGISTRY:
        send_msg("complete", {
            "error": f"LLM 指定了未知读取函数 '{read_function}'，"
                     f"可用：{list(READER_REGISTRY)}"
        })
        return

    send_msg("info", f"已选定算法：{algorithm} — {reasoning}")

    # ---- Step 3: 读取数据（带容错重试）----
    send_msg("info", f"正在读取数据（使用 {read_function}）...")

    try:
        reader_fn = READER_REGISTRY[read_function]
        data      = reader_fn(csv_path, **read_params)
    except Exception as e:
        send_msg("info", f"⚠️ {read_function} 读取失败: {e}，尝试备用方法...")
        # 尝试其他读取函数
        data, alternative_reader = _try_alternative_readers(csv_path, read_function, read_params)
        if data is None:
            send_msg("complete", {"error": f"所有读取方法均失败。原始错误: {e}"})
            return
        send_msg("info", f"✓ 使用 {alternative_reader} 成功读取数据")
        read_function = alternative_reader  # 更新为实际使用的读取函数

    # ---- Step 4: 运行算法 ----
    send_msg("progress", f"正在执行 {algorithm} 算法...")

    try:
        algo_result = run_fn(algorithm, data, {})
    except Exception as e:
        send_msg("complete", {"error": f"算法执行异常: {e}"})
        return

    if not algo_result.get("success"):
        msg = algo_result.get("message", "未知错误")
        send_msg("complete", {"error": f"算法执行失败: {msg}"})
        return

    result_data = algo_result.get("result", {})

    # ---- Step 5: 保存结果 ----
    try:
        archive_path = _save_result(csv_path, algorithm, reasoning, result_data)
        send_msg("info", f"结果已保存至 {archive_path}")
    except Exception as e:
        archive_path = ""
        send_msg("info", f"⚠️ 结果保存失败: {e}")

    # ---- Step 6: 推送结果到前端 ----
    summary = _extract_result_summary(result_data)
    send_msg("analysis_result", {
        "algorithm"     : algorithm,
        "reasoning"     : reasoning,
        "result_summary": summary,
        "filepath"      : archive_path,
    })

    send_msg("complete", {
        "file"     : archive_path,
        "algorithm": algorithm,
    })


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import math
    import csv as _csv

    # 生成测试 CSV（光谱数据，覆盖写不删除）
    _TEST_CSV = os.path.join(os.path.dirname(__file__), "_test_spectrum.csv")
    wl        = list(range(400, 701))
    intensity = [0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2) for w in wl]

    with open(_TEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["wavelength", "intensity"])
        for w, i in zip(wl, intensity):
            writer.writerow([w, round(i, 6)])

    print(f"测试 CSV 已写入: {_TEST_CSV}")
    print("=" * 60)

    # 使用 SoftwareManager 提供算法列表和执行函数（避免直接实例化 controller）
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.software_manager import SoftwareManager
    mgr = SoftwareManager()

    def _print_msg(msg_type, data):
        import json as _json
        if isinstance(data, dict):
            print(f"[{msg_type}] {_json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"[{msg_type}] {data}")

    run_pipeline(
        csv_path   = _TEST_CSV,
        send_msg   = _print_msg,
        algorithms = mgr.list_algorithms(),
        run_fn     = mgr.run_algorithm,
    )

    print("\n测试完成。结果文件已保存在 results/ 目录（不会自动删除）。")
