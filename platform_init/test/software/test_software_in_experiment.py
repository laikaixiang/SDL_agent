"""
software 在 experiment 里跑 — compiler + executor 的 software 步骤测试

运行: pytest platform_init/test/software/ -v
"""
import sys
import os
import json
import csv
from unittest.mock import MagicMock, patch

# 项目根目录加进 sys.path
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pytest


# =============================================================================
# E1-E5, E9: ExperimentCompiler 对 software step 的处理（纯字符串测试）
# =============================================================================

def test_compile_software_step_imports():
    """E1: compiler 给 type:software 步生成 SoftwareManager import

    TODO(lkx): compiler.py 第 193 行的 import 路径写成 'software.software_manager',
    但真实模块在 core/。当前生成的代码执行时会 ImportError。
    此测试断言原状 (回归测试),bug 修复后需更新断言。
    """
    from experiment.compiler import ExperimentCompiler
    compiler = ExperimentCompiler()
    code = compiler.compile_to_python({
        "experiment_name": "t",
        "steps": [
            {"type": "software", "name": "data_statistics", "params": {}}
        ]
    })
    print(f"\n=== E1: import line in code: 'SoftwareManager' present: {'SoftwareManager' in code} ===")
    assert "SoftwareManager" in code
    # 当前实现有 bug：import 路径错
    assert "from software.software_manager import SoftwareManager" in code
    print("PASS")


def test_compile_software_step_call():
    """E2: 生成 sm.run_algorithm(name, _data, params) 调用字符串"""
    from experiment.compiler import ExperimentCompiler
    compiler = ExperimentCompiler()
    code = compiler.compile_to_python({
        "experiment_name": "t",
        "steps": [
            {"type": "software", "name": "data_statistics",
             "params": {"include_correlation": True}}
        ]
    })
    print(f"\n=== E2: code contains sm.run_algorithm: {'sm.run_algorithm' in code} ===")
    assert "sm.run_algorithm(" in code
    assert '"data_statistics"' in code
    # params 序列化为 JSON 字符串
    assert '"include_correlation": true' in code or '"include_correlation": True' in code
    print("PASS")


def test_compile_software_step_with_user_params():
    """E3: 编译层不处理 user_params(那是 executor 运行时的事), 验证 params 序列化

    注:user_params 的合并逻辑在 experiment/executor.py:_execute_software_algorithm,
    不在 compiler。E3 在 executor 那侧覆盖(E8 补充)。
    """
    from experiment.compiler import ExperimentCompiler
    compiler = ExperimentCompiler()
    code = compiler.compile_to_python({
        "experiment_name": "t",
        "steps": [
            {"type": "software", "name": "data_statistics",
             "params": {"method": "minmax"}}
        ]
    })
    assert '"method": "minmax"' in code
    print(f"\n=== E3: params 序列化通过 ===")
    print("PASS")


def test_compile_software_step_uses_input_file():
    """E4: 指定 input_file 时,生成 sm._read_csv_as_columns 调用"""
    from experiment.compiler import ExperimentCompiler
    compiler = ExperimentCompiler()
    code = compiler.compile_to_python({
        "experiment_name": "t",
        "steps": [
            {"type": "software", "name": "data_statistics",
             "params": {}, "input_file": "temporal/data.csv",
             "output_file": "results/out.json"}
        ]
    })
    print(f"\n=== E4: code has _read_csv_as_columns: {'_read_csv_as_columns' in code} ===")
    assert "_read_csv_as_columns" in code
    assert '"temporal/data.csv"' in code
    assert '"results/out.json"' in code
    assert "json.dump" in code  # output_file 写盘
    print("PASS")


def test_compile_mixed_plan_syntax_valid():
    """E5: tool + helper + software 三类混,编译出的代码语法合法(可 compile)"""
    from experiment.compiler import ExperimentCompiler
    compiler = ExperimentCompiler()
    code = compiler.compile_to_python({
        "experiment_name": "mixed",
        "steps": [
            {"type": "tool", "name": "move_robot_arm",
             "params": {"x": 100, "y": 200, "z": 300}},
            {"type": "helper", "name": "WAIT", "params": {"duration": 1000}},
            {"type": "software", "name": "data_statistics", "params": {}},
        ]
    })
    # 语法检查:compile() 不执行,只 parse
    try:
        compile(code, "<test_compile_mixed_plan>", "exec")
        print(f"\n=== E5: mixed plan compiles to syntactically valid Python ===")
        print("PASS")
    except SyntaxError as e:
        pytest.fail(f"compiled code has syntax error: {e}\n--- code ---\n{code}")


def test_compile_action_field_backcompat():
    """E9: 旧格式 (action 字段) 用于 tool 步骤,_build_imports 仍能识别

    注:compiler 的 action 兼容只覆盖 import 行,不覆盖 tool 调用 (call 仍用 name)。
    这是真实现状,见 _build_imports 第 45 行: name = step.get("name") or step.get("action", "")
    """
    from experiment.compiler import ExperimentCompiler
    compiler = ExperimentCompiler()
    steps = [
        {"type": "tool", "action": "move_robot_arm",
         "params": {"x": 1, "y": 2, "z": 3}},
    ]
    registry = compiler._load_registry()
    import_line = compiler._build_imports(steps, registry)
    print(f"\n=== E9: import line with action fallback: {import_line!r} ===")
    assert "move_robot_arm" in import_line, f"import should resolve via action field, got: {import_line}"
    print("PASS")


# =============================================================================
# E6-E8: ExperimentExecutor 跑 software step
# =============================================================================

def _make_software_manager_mock():
    """构造 SoftwareManager mock,提供 _read_csv_as_columns + run_algorithm"""
    mgr = MagicMock()
    mgr._read_csv_as_columns = MagicMock(return_value={"col": [1.0, 2.0, 3.0]})
    mgr.run_algorithm = MagicMock(return_value={
        "success": True,
        "algorithm": "data_statistics",
        "result": {"statistics": {"col": {"mean": 2.0}}},
        "message": "ok",
    })
    return mgr


def test_executor_software_only_plan():
    """E6: 纯 software 步 plan,execute_plan 跑通"""
    from experiment.executor import ExperimentExecutor

    mgr = _make_software_manager_mock()
    # 注入 mock,避开 HardwareAgent 真实构造
    with patch("experiment.executor.HardwareAgent" if False else "core.hardware_controller.HardwareAgent", MagicMock()):
        executor = ExperimentExecutor(software_manager=mgr)

    plan = {
        "experiment_name": "sw-only",
        "steps": [
            {"step_number": 1, "type": "software",
             "name": "data_statistics", "params": {}, "description": "统计分析"}
        ]
    }
    result = executor.execute_plan(plan)
    print(f"\n=== E6: success={result['success']}, results count={len(result['results'])} ===")
    assert result["success"] is True
    assert mgr.run_algorithm.called
    call_args = mgr.run_algorithm.call_args
    # run_algorithm(algo_name, data, params)
    assert call_args[0][0] == "data_statistics", f"expected data_statistics, got {call_args[0][0]}"
    print(f"  call_args: {call_args}")
    print("PASS")


def test_executor_unknown_software_algo():
    """E7: 不存在的算法,executor 不崩溃,标记失败"""
    from experiment.executor import ExperimentExecutor

    mgr = MagicMock()
    mgr.run_algorithm = MagicMock(return_value={
        "success": False, "algorithm": "fake_algo", "result": None,
        "message": "未找到算法 'fake_algo'"
    })

    with patch("core.hardware_controller.HardwareAgent", MagicMock()):
        executor = ExperimentExecutor(software_manager=mgr)

    plan = {
        "experiment_name": "unknown",
        "steps": [
            {"type": "software", "name": "fake_algo", "params": {}}
        ]
    }
    result = executor.execute_plan(plan)
    print(f"\n=== E7: success={result['success']} (expect False) ===")
    assert result["success"] is False
    # step-level: software 步的 result.message 应含"未找到"
    sw_step = result["results"][0]
    assert "未找到" in str(sw_step.get("result", "")) or sw_step["success"] is False
    print("PASS")


def test_executor_software_step_with_input_file(tmp_path):
    """E8: input_file 存在时,executor 读 CSV → 调 software_manager.run_algorithm"""
    from experiment.executor import ExperimentExecutor

    # 准备真实 CSV
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["col"])
        writer.writerow([1.0])
        writer.writerow([2.0])
        writer.writerow([3.0])

    mgr = _make_software_manager_mock()

    with patch("core.hardware_controller.HardwareAgent", MagicMock()):
        executor = ExperimentExecutor(software_manager=mgr)

    plan = {
        "experiment_name": "with-input",
        "steps": [
            {"type": "software", "name": "data_statistics",
             "params": {}, "input_file": str(csv_file)}
        ]
    }
    result = executor.execute_plan(plan)
    print(f"\n=== E8: success={result['success']} ===")
    assert result["success"] is True
    # 应真调 _read_csv_as_columns(file)
    mgr._read_csv_as_columns.assert_called_once_with(str(csv_file))
    print(f"  _read_csv_as_columns called with: {mgr._read_csv_as_columns.call_args}")
    print("PASS")


def test_executor_software_user_params_override():
    """E3-executor 侧:user_params 合并进 params,user_params 优先级更高

    注:params 中的字符串值会被 VariableResolver 当成变量名校验
    (匹配 [a-zA-Z_][a-zA-Z0-9_]* 规则的字符串)。本测试用非变量形状的
    bool/number 值绕开校验,聚焦测 merge 逻辑。
    """
    from experiment.executor import ExperimentExecutor

    mgr = _make_software_manager_mock()
    with patch("core.hardware_controller.HardwareAgent", MagicMock()):
        executor = ExperimentExecutor(software_manager=mgr)

    plan = {
        "experiment_name": "user-params-merge",
        "steps": [
            {
                "type": "software", "name": "data_statistics",
                "params": {"include_correlation": False, "threshold": 0.1},
                "user_params": {"include_correlation": True},  # 覆盖
            }
        ]
    }
    result = executor.execute_plan(plan)
    print(f"\n=== E3-executor: success={result['success']} ===")
    assert result["success"] is True
    call_kwargs = mgr.run_algorithm.call_args
    actual_params = call_kwargs[0][2]  # 第三位置参数
    print(f"  merged params: {actual_params}")
    assert actual_params["include_correlation"] is True, "user_params 应覆盖 params"
    assert actual_params["threshold"] == 0.1, "params 中未覆盖的字段应保留"
    print("PASS")


if __name__ == "__main__":
    import inspect
    import sys as _sys
    current_module = _sys.modules[__name__]
    fixture_params = {"tmp_path"}
    test_funcs = [(n, fn) for n, fn in inspect.getmembers(current_module, inspect.isfunction)
                  if n.startswith("test_")]
    failed = []
    for name, fn in test_funcs:
        sig = inspect.signature(fn)
        if any(p.name in fixture_params for p in sig.parameters.values()):
            print(f"SKIP {name} (needs pytest fixture)")
            continue
        try:
            fn()
        except BaseException as e:
            if "Skipped" in type(e).__name__:
                print(f"SKIP {name}: {e}")
                continue
            failed.append((name, e))
            print(f"FAIL {name}: {e}\n")
    print(f"\n{'='*60}\n{len(test_funcs) - len(failed)}/{len(test_funcs)} passed")
    if failed:
        for n, e in failed:
            print(f"  FAIL {n}: {e}")
        _sys.exit(1)
