"""
变量解析器 — 单元测试

运行方法: python platform_init/test/variable_system/test_variable_resolver.py
"""
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.variable_resolver import VariableResolver

vr = VariableResolver()

# ==================== 类型推断 ====================

def test_infer_type_int():
    print("\n=== test_infer_type_int ===")
    assert vr._infer_type(3000) == "int"
    assert vr._infer_type("3000") == "int"
    assert vr._infer_type(0) == "int"
    print("PASS")


def test_infer_type_float():
    print("\n=== test_infer_type_float ===")
    assert vr._infer_type(3.14) == "float"
    assert vr._infer_type("3.14") == "float"
    print("PASS")


def test_infer_type_str():
    print("\n=== test_infer_type_str ===")
    assert vr._infer_type("Perovskite") == "str"
    assert vr._infer_type("") == "str"
    print("PASS")


def test_infer_type_bool():
    print("\n=== test_infer_type_bool ===")
    assert vr._infer_type(True) == "bool"
    assert vr._infer_type("true") == "bool"
    assert vr._infer_type("false") == "bool"
    print("PASS")


# ==================== 变量名校验 ====================

def test_is_variable_name_valid():
    print("\n=== test_is_variable_name_valid ===")
    assert vr._is_variable_name("speed1") == True
    assert vr._is_variable_name("speed_1") == True
    assert vr._is_variable_name("_temp") == True
    assert vr._is_variable_name("a") == True
    assert vr._is_variable_name("ABC") == True
    print("PASS")


def test_is_variable_name_invalid():
    print("\n=== test_is_variable_name_invalid ===")
    assert vr._is_variable_name("1speed") == False
    assert vr._is_variable_name("speed-1") == False
    assert vr._is_variable_name("speed 1") == False
    assert vr._is_variable_name("") == False
    assert vr._is_variable_name("变量") == False
    print("PASS")


# ==================== 表达式求值 ====================

def test_expression_arithmetic():
    print("\n=== test_expression_arithmetic ===")
    assert vr.evaluate_expression("2 + 3", {}) == 5
    assert vr.evaluate_expression("10 - 4", {}) == 6
    assert vr.evaluate_expression("3 * 7", {}) == 21
    assert vr.evaluate_expression("10 // 3", {}) == 3
    assert vr.evaluate_expression("10 % 3", {}) == 1
    assert vr.evaluate_expression("2 ** 3", {}) == 8
    print("PASS")


def test_expression_with_variables():
    print("\n=== test_expression_with_variables ===")
    flat_vars = {"speed": 1000, "factor": 2}
    assert vr.evaluate_expression("speed * factor", flat_vars) == 2000
    assert vr.evaluate_expression("speed + 500", flat_vars) == 1500
    assert vr.evaluate_expression("(speed + 100) * factor", flat_vars) == 2200
    print("PASS")


def test_expression_comparison():
    print("\n=== test_expression_comparison ===")
    assert vr.evaluate_expression("5 > 3", {}) == True
    assert vr.evaluate_expression("5 < 3", {}) == False
    assert vr.evaluate_expression("5 == 5", {}) == True
    assert vr.evaluate_expression("5 != 3", {}) == True
    assert vr.evaluate_expression("5 >= 5", {}) == True
    assert vr.evaluate_expression("5 <= 3", {}) == False
    print("PASS")


def test_expression_logic():
    print("\n=== test_expression_logic ===")
    assert vr.evaluate_expression("True and True", {}) == True
    assert vr.evaluate_expression("True and False", {}) == False
    assert vr.evaluate_expression("True or False", {}) == True
    assert vr.evaluate_expression("not False", {}) == True
    print("PASS")


def test_expression_unknown_var():
    print("\n=== test_expression_unknown_var ===")
    try:
        vr.evaluate_expression("unknown_var + 1", {})
        assert False, "应该抛出异常"
    except ValueError as e:
        assert "未定义" in str(e) or "unknown_var" in str(e)
    print("PASS")


def test_expression_blocked_function():
    print("\n=== test_expression_blocked_function ===")
    try:
        vr.evaluate_expression("abs(-5)", {})
        assert False, "应该抛出异常（函数调用被禁止）"
    except ValueError as e:
        assert "不支持" in str(e)
    print("PASS")


# ==================== 变量校验 ====================

def test_validate_all_declared():
    print("\n=== test_validate_all_declared ===")
    variables = {
        "speed": {"type": "int", "default_value": 3000},
        "reagent": {"type": "str", "default_value": "Perovskite"},
    }
    steps = [
        {"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed"}},
        {"type": "tool", "name": "drop", "params": {"reagent": "reagent"}},
    ]
    ok, err = vr.validate_variables(variables, steps)
    assert ok, f"期望通过，但报错: {err}"
    print("PASS")


def test_validate_undeclared():
    print("\n=== test_validate_undeclared ===")
    variables = {"speed": {"type": "int", "default_value": 3000}}
    steps = [
        {"type": "tool", "name": "spin_coating", "params": {"spin_speed": "duration"}},
    ]
    ok, err = vr.validate_variables(variables, steps)
    assert not ok
    assert "duration" in err
    print("PASS")


def test_validate_no_variables():
    print("\n=== test_validate_no_variables ===")
    steps = [
        {"type": "tool", "name": "spin_coating", "params": {"spin_speed": 3000}},
    ]
    ok, err = vr.validate_variables({}, steps)
    assert ok
    print("PASS")


def test_validate_no_vars_but_referenced():
    print("\n=== test_validate_no_vars_but_referenced ===")
    steps = [
        {"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed"}},
    ]
    ok, err = vr.validate_variables({}, steps)
    assert not ok
    assert "speed" in err
    print("PASS")


def test_validate_type_mismatch():
    print("\n=== test_validate_type_mismatch ===")
    variables = {"speed": {"type": "int", "default_value": "不是数字"}}
    steps = []
    ok, err = vr.validate_variables(variables, steps)
    assert not ok
    assert "speed" in err
    print("PASS")


def test_validate_constraints_min():
    print("\n=== test_validate_constraints_min ===")
    variables = {"speed": {"type": "int", "default_value": 500, "constraints": {"min": 1000}}}
    steps = []
    ok, err = vr.validate_variables(variables, steps)
    assert not ok, f"期望校验失败，但通过了"
    assert "最小值" in err or "小于" in err or "min" in err.lower(), f"错误信息不包含范围相关: {err}"
    print("PASS")


def test_validate_constraints_max():
    print("\n=== test_validate_constraints_max ===")
    variables = {"speed": {"type": "int", "default_value": 7000, "constraints": {"max": 6000}}}
    steps = []
    ok, err = vr.validate_variables(variables, steps)
    assert not ok, f"期望校验失败，但通过了"
    assert "最大值" in err or "大于" in err or "max" in err.lower(), f"错误信息不包含范围相关: {err}"
    print("PASS")


def test_validate_normalize_missing_type():
    print("\n=== test_validate_normalize_missing_type ===")
    variables = {"speed1": {"default_value": 3000, "constraints": {"min": 1000, "max": 6000}}}
    steps = [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed1"}}]
    ok, err = vr.validate_variables(variables, steps)
    assert ok, f"期望通过，但报错: {err}"
    assert variables["speed1"]["type"] == "int", "应该自动推断type=int"
    print("PASS")


# ==================== 变量解析 ====================

def test_resolve_replaces_variable():
    print("\n=== test_resolve_replaces_variable ===")
    experiment = {
        "experiment_name": "test",
        "variables": {"speed": {"type": "int", "default_value": 3000}},
        "steps": [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed", "spin_acc": 500}}],
    }
    resolved = vr.resolve(experiment)
    assert resolved["steps"][0]["params"]["spin_speed"] == 3000
    assert resolved["steps"][0]["params"]["spin_acc"] == 500
    print("PASS")


def test_resolve_expression():
    print("\n=== test_resolve_expression ===")
    experiment = {
        "experiment_name": "test",
        "variables": {"base": {"type": "int", "default_value": 1000}},
        "steps": [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": "base * 2 + 500"}}],
    }
    resolved = vr.resolve(experiment)
    assert resolved["steps"][0]["params"]["spin_speed"] == 2500
    print("PASS")


def test_resolve_does_not_mutate_original():
    print("\n=== test_resolve_does_not_mutate_original ===")
    experiment = {
        "experiment_name": "test",
        "variables": {"speed": {"type": "int", "default_value": 3000}},
        "steps": [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed"}}],
    }
    _resolved = vr.resolve(experiment)
    assert experiment["steps"][0]["params"]["spin_speed"] == "speed"
    print("PASS")


def test_resolve_no_variables():
    print("\n=== test_resolve_no_variables ===")
    experiment = {
        "experiment_name": "test",
        "steps": [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": 3000}}],
    }
    resolved = vr.resolve(experiment)
    assert resolved["steps"][0]["params"]["spin_speed"] == 3000
    print("PASS")


# ==================== 批量解析 ====================

def test_resolve_batch():
    print("\n=== test_resolve_batch ===")
    experiment = {
        "experiment_name": "test",
        "variables": {"speed": {"type": "int", "default_value": 3000}},
        "steps": [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed"}}],
        "batch_data": [{"speed": 4000}, {"speed": 5000}],
    }
    results = vr.resolve_batch(experiment)
    assert len(results) == 2
    assert results[0]["steps"][0]["params"]["spin_speed"] == 4000
    assert results[1]["steps"][0]["params"]["spin_speed"] == 5000
    print("PASS")


def test_resolve_batch_normalize_type():
    print("\n=== test_resolve_batch_normalize_type ===")
    experiment = {
        "experiment_name": "test",
        "variables": {"speed": {"default_value": 3000}},
        "steps": [{"type": "tool", "name": "spin_coating", "params": {"spin_speed": "speed"}}],
        "batch_data": [{"speed": 4000}],
    }
    results = vr.resolve_batch(experiment)
    assert len(results) == 1
    assert results[0]["steps"][0]["params"]["spin_speed"] == 4000
    print("PASS")


# ==================== CSV 解析 ====================

def test_parse_csv():
    print("\n=== test_parse_csv ===")
    csv = "speed,duration,temp\n3000,30,150\n4000,25,200"
    variables, batch_data, err = vr.parse_csv(csv)
    assert err is None, f"期望无错误，但: {err}"
    assert "speed" in variables
    assert "duration" in variables
    assert "temp" in variables
    assert variables["speed"]["type"] == "int"
    assert variables["duration"]["type"] == "int"
    assert variables["temp"]["type"] == "int"
    assert len(batch_data) == 2
    assert batch_data[0]["speed"] == 3000
    assert batch_data[1]["temp"] == 200
    print("PASS")


def test_parse_csv_with_strings():
    print("\n=== test_parse_csv_with_strings ===")
    csv = "reagent,concentration\nPerovskite,1.5\nMAPbI3,2.0"
    variables, batch_data, err = vr.parse_csv(csv)
    assert err is None, f"期望无错误，但: {err}"
    assert variables["reagent"]["type"] == "str"
    assert variables["concentration"]["type"] == "float"
    assert batch_data[0]["reagent"] == "Perovskite"
    assert batch_data[1]["concentration"] == 2.0
    print("PASS")


def test_parse_csv_empty():
    print("\n=== test_parse_csv_empty ===")
    _, _, err = vr.parse_csv("")
    assert err is not None
    print("PASS")


def test_parse_csv_no_data_rows():
    print("\n=== test_parse_csv_no_data_rows ===")
    _, _, err = vr.parse_csv("a,b,c")
    assert err is not None
    print("PASS")


# ==================== 参数值解析（静态方法，用 flat dict） ====================

def test_resolve_param_value_variable():
    print("\n=== test_resolve_param_value_variable ===")
    flat_vars = {"speed": 3000}
    result = VariableResolver._resolve_param_value("speed", flat_vars)
    assert result == 3000, f"期望3000，得到 {result}"
    print("PASS")


def test_resolve_param_value_literal_number():
    print("\n=== test_resolve_param_value_literal_number ===")
    assert VariableResolver._resolve_param_value("3000", {}) == 3000
    assert VariableResolver._resolve_param_value("3.14", {}) == 3.14
    print("PASS")


def test_resolve_param_value_literal_string():
    print("\n=== test_resolve_param_value_literal_string ===")
    assert VariableResolver._resolve_param_value("Perovskite", {}) == "Perovskite"
    print("PASS")


def test_resolve_param_value_expression():
    print("\n=== test_resolve_param_value_expression ===")
    flat_vars = {"base": 1000}
    result = VariableResolver._resolve_param_value("base + 500", flat_vars)
    assert result == 1500, f"期望1500，得到 {result}"
    print("PASS")


def test_resolve_param_value_undeclared():
    print("\n=== test_resolve_param_value_undeclared ===")
    result = VariableResolver._resolve_param_value("unknown", {})
    assert result == "unknown"
    print("PASS")


def test_resolve_param_value_non_string():
    print("\n=== test_resolve_param_value_non_string ===")
    assert VariableResolver._resolve_param_value(3000, {}) == 3000
    assert VariableResolver._resolve_param_value(3.14, {}) == 3.14
    assert VariableResolver._resolve_param_value(None, {}) is None
    print("PASS")


# ==================== 值类型检查 ====================

def test_check_value_int():
    print("\n=== test_check_value_int ===")
    ok, _ = vr._check_value_against_type("x", 3000, "int", None)
    assert ok
    ok, _ = vr._check_value_against_type("x", 3000.5, "int", None)
    assert not ok
    print("PASS")


def test_check_value_str():
    print("\n=== test_check_value_str ===")
    ok, _ = vr._check_value_against_type("x", "hello", "str", None)
    assert ok
    ok, _ = vr._check_value_against_type("x", 123, "str", None)
    assert not ok
    print("PASS")


def test_check_value_str_options():
    print("\n=== test_check_value_str_options ===")
    c = {"options": ["A", "B"]}
    ok, err = vr._check_value_against_type("x", "A", "str", c)
    assert ok, f"A应在选项中，但: {err}"
    ok, err = vr._check_value_against_type("x", "C", "str", c)
    assert not ok
    print("PASS")


def test_check_value_min_max():
    print("\n=== test_check_value_min_max ===")
    ok, _ = vr._check_value_against_type("x", 3000, "int", {"min": 1000, "max": 6000})
    assert ok
    ok, _ = vr._check_value_against_type("x", 500, "int", {"min": 1000})
    assert not ok
    ok, _ = vr._check_value_against_type("x", 7000, "int", {"max": 6000})
    assert not ok
    print("PASS")


# ==================== 运行所有测试 ====================

if __name__ == "__main__":
    tests = [
        test_infer_type_int, test_infer_type_float, test_infer_type_str, test_infer_type_bool,
        test_is_variable_name_valid, test_is_variable_name_invalid,
        test_expression_arithmetic, test_expression_with_variables, test_expression_comparison,
        test_expression_logic, test_expression_unknown_var, test_expression_blocked_function,
        test_validate_all_declared, test_validate_undeclared, test_validate_no_variables,
        test_validate_no_vars_but_referenced, test_validate_type_mismatch,
        test_validate_constraints_min, test_validate_constraints_max,
        test_validate_normalize_missing_type,
        test_resolve_replaces_variable, test_resolve_expression, test_resolve_does_not_mutate_original,
        test_resolve_no_variables,
        test_resolve_batch, test_resolve_batch_normalize_type,
        test_parse_csv, test_parse_csv_with_strings, test_parse_csv_empty, test_parse_csv_no_data_rows,
        test_resolve_param_value_variable, test_resolve_param_value_literal_number,
        test_resolve_param_value_literal_string, test_resolve_param_value_expression,
        test_resolve_param_value_undeclared, test_resolve_param_value_non_string,
        test_check_value_int, test_check_value_str, test_check_value_str_options,
        test_check_value_min_max,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    if failed > 0:
        sys.exit(1)
