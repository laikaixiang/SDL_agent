"""
software 测试 conftest — 共享 fixture

运行方式: pytest platform_init/test/software/  或
          python platform_init/test/software/test_software_direct.py
"""
import sys
import os
import io
import json
import csv
import shutil
import tempfile
from pathlib import Path

# 项目根目录加进 sys.path（4 级上跳：software → test → platform_init → project_root）
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Windows stdout 编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# 样本数据 fixture
# =============================================================================

import pytest


@pytest.fixture
def tmp_csv_path(tmp_path) -> str:
    """生成 3 列（PCE/thickness/label）10 行的样本 CSV，第三列故意是非数值列"""
    csv_file = tmp_path / "extraction.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PCE", "thickness", "label"])
        for i in range(10):
            writer.writerow([15.0 + i * 0.3, 100 + i * 10, f"sample_{i}"])
    return str(csv_file)


@pytest.fixture
def tmp_spectral_csv(tmp_path) -> str:
    """生成 301 波长点的高斯峰光谱 CSV（峰值在 532nm）"""
    import math
    csv_file = tmp_path / "spectrum.csv"
    wl = list(range(400, 701))
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength", "intensity"])
        for w in wl:
            intensity = 0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2)
            writer.writerow([w, round(intensity, 6)])
    return str(csv_file)


@pytest.fixture
def sample_spectrum_data():
    """内存里合成的高斯峰光谱 dict,复用 D5/E8"""
    import math
    wl = list(range(400, 701))
    intensity = [0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2) for w in wl]
    return {"wavelength": wl, "intensity": intensity}


@pytest.fixture
def tmp_results_dir(tmp_path) -> str:
    """SoftwareManager 写 results 的目录"""
    results = tmp_path / "results"
    results.mkdir()
    return str(results)


@pytest.fixture
def software_manager(tmp_csv_path, tmp_results_dir):
    """实例化 SoftwareManager（指向 tmp CSV/results）

    注:SoftwareManager 真实路径在 core/，不在 software/。
    compiler.py 第 193 行的 import 路径是错的,实际生成的代码会 ImportError。
    """
    from core.software_manager import SoftwareManager
    return SoftwareManager(temporal_dir=str(Path(tmp_csv_path).parent),
                           results_dir=tmp_results_dir)


# =============================================================================
# LLM mock fixture（auto_analyze + prompt_template 共用 _call_llm 签名）
# =============================================================================

@pytest.fixture
def mock_llm_clean_json(monkeypatch):
    """LLM 直接返干净 JSON。auto_analyze 用 1 次。"""
    response = json.dumps({
        "algorithm": "data_statistics",
        "read_function": "read_numeric_columns",
        "read_params": {},
        "reasoning": "用户需要统计描述",
    })

    def fake_call(system, user):
        return response

    monkeypatch.setattr("software.auto_analyze._call_llm", fake_call)
    return response


@pytest.fixture
def mock_llm_markdown_wrapped(monkeypatch):
    """LLM 返 ```json ... ``` 包裹的 JSON,验证 _strip_json 鲁棒"""
    inner = json.dumps({
        "algorithm": "data_statistics",
        "read_function": "read_numeric_columns",
        "read_params": {},
        "reasoning": "wrapped",
    })
    response = f"```json\n{inner}\n```"

    monkeypatch.setattr("software.auto_analyze._call_llm", lambda s, u: response)
    return response


@pytest.fixture
def mock_llm_invalid_json(monkeypatch):
    """LLM 返无效 JSON"""
    monkeypatch.setattr("software.auto_analyze._call_llm",
                        lambda s, u: "这不是 JSON { 无效")
    return None


@pytest.fixture
def mock_llm_unknown_algorithm(monkeypatch):
    """LLM 选了不在 registry 里的算法,验证拒绝"""
    response = json.dumps({
        "algorithm": "this_algo_does_not_exist",
        "read_function": "read_numeric_columns",
        "read_params": {},
        "reasoning": "幻觉",
    })
    monkeypatch.setattr("software.auto_analyze._call_llm", lambda s, u: response)
    return response


@pytest.fixture
def mock_llm_for_generate(monkeypatch, tmp_path):
    """prompt_template._call_llm 专用:返 spec + 代码。
    注:本 fixture 同时 mock 2 个 LLM 调用点(spec 提取 + 代码生成)。
    副作用:会真把代码写到 extra_algorithms_fromProjects/<name>.py,
            teardown 自动清理。
    """
    spec = {
        "name": "_test_moving_average_xxx",  # 唯一前缀,避免与历史冲突
        "description": "测试用移动平均",
        "input_format": "dict with 'values' key",
        "output_fields": ["smoothed", "residuals"],
        "params": [
            {"name": "window_size", "type": "int",
             "description": "窗口大小", "default": 3},
        ],
    }
    code = f'''
"""自动生成的测试算法(由 conftest mock_llm_for_generate 注入)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from software.algorithms.base import BaseAlgorithm

class _TestMovingAverageXxx(BaseAlgorithm):
    name = "{spec["name"]}"
    chinese_name = "测试移动平均"
    description = "测试用"
    params_schema = {{"window_size": {{"type": "int", "default": 3, "required": False}}}}

    def run(self, data, params=None):
        params = params or {{}}
        w = int(params.get("window_size", 3))
        if isinstance(data, dict) and "values" in data:
            vals = data["values"]
        else:
            vals = data if isinstance(data, list) else []
        smoothed = [sum(vals[max(0,i-w+1):i+1])/min(i+1,w) for i in range(len(vals))]
        return self._build_success({{"smoothed": smoothed}}, "ok")

if __name__ == "__main__":
    pass
'''

    state = {"calls": 0, "spec": spec, "code": code, "filepath": None}

    def fake_call(system, user):
        state["calls"] += 1
        if state["calls"] == 1:
            return json.dumps(spec)
        return code

    monkeypatch.setattr(
        "software.algorithms.extra_algorithms_fromProjects.prompt_template._call_llm",
        fake_call,
    )

    yield state

    # teardown: 删除生成的文件,清缓存
    if state["filepath"] and os.path.exists(state["filepath"]):
        try:
            os.remove(state["filepath"])
        except OSError:
            pass
    # 强制 reload 让后续 test 看到清理
    try:
        from software.software_controller import SoftwareController
        SoftwareController()
    except Exception:
        pass


# =============================================================================
# 算法注册表 fixture（disposable + broken）
# =============================================================================

@pytest.fixture
def disposable_algo_file(tmp_path, monkeypatch):
    """在 extra_algorithms_fromProjects/ 下注入临时算法文件,teardown 删
    返回 (file_path, class_name)"""
    name = "_test_disposable_algo_xxx"
    code = f'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from software.algorithms.base import BaseAlgorithm

class _TestDisposableAlgoXxx(BaseAlgorithm):
    name = "{name}"
    chinese_name = "测试可弃算法"
    description = "disposable fixture 用"
    params_schema = {{}}
    def run(self, data, params=None):
        return self._build_success({{"echo": data}}, "ok")
'''
    target_dir = Path(_PROJECT_ROOT) / "software" / "algorithms" / "extra_algorithms_fromProjects"
    target_path = target_dir / f"{name}.py"
    target_path.write_text(code, encoding="utf-8")

    yield (str(target_path), name)

    if target_path.exists():
        try:
            target_path.unlink()
        except OSError:
            pass
    # 强制 reload
    try:
        from software.software_controller import SoftwareController
        SoftwareController()
    except Exception:
        pass


@pytest.fixture
def broken_algo_file(tmp_path, monkeypatch):
    """注入一个会 ImportError 的 .py,验证 get_load_errors 捕获"""
    name = "_test_broken_algo_xxx"
    code = '''
import this_module_definitely_does_not_exist_xyz  # noqa
'''
    target_dir = Path(_PROJECT_ROOT) / "software" / "algorithms" / "extra_algorithms_fromProjects"
    target_path = target_dir / f"{name}.py"
    target_path.write_text(code, encoding="utf-8")

    yield str(target_path)

    if target_path.exists():
        try:
            target_path.unlink()
        except OSError:
            pass
    try:
        from software.software_controller import SoftwareController
        SoftwareController()
    except Exception:
        pass
