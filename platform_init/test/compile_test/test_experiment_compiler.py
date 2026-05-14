import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from experiment.compiler import ExperimentCompiler
import json

compiler = ExperimentCompiler()

def test_registry_loads():
    registry = compiler._load_registry()
    assert isinstance(registry, dict)
    assert "spin_coating" in registry
    assert "move_robot_arm" in registry
    print("PASS: test_registry_loads")

def test_import_generation():
    steps = [
        {"type": "tool", "name": "spin_coating"},
        {"type": "tool", "name": "move_robot_arm"},
    ]
    registry = compiler._load_registry()
    line = compiler._build_imports(steps, registry)
    assert "execute_spin_coating" in line
    assert "execute_move_robot_arm" in line
    assert line.startswith("from hardware import ")
    print("PASS: test_import_generation")

def test_positional_call_generation():
    registry = compiler._load_registry()
    call = compiler._build_tool_call("move_robot_arm", {"x": 100, "y": 200, "z": 300}, registry)
    assert "execute_move_robot_arm(100.0, 200.0, 300.0, 0.0)" == call, f"Got: {call}"
    print("PASS: test_positional_call_generation")

def test_missing_required_param_raises():
    registry = compiler._load_registry()
    try:
        compiler._build_tool_call("spin_coating", {"spin_speed": 3000}, registry)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_missing_required_param_raises")

def test_type_conversion():
    registry = compiler._load_registry()
    call = compiler._build_tool_call("spin_coating", {"spin_speed": 3000, "reagent": "Perovskite"}, registry)
    assert '"Perovskite"' in call  # str quoted
    assert '3000' in call  # int bare
    print("PASS: test_type_conversion")

def test_default_values_applied():
    registry = compiler._load_registry()
    call = compiler._build_tool_call("spin_coating", {"spin_speed": 3000, "reagent": "Perovskite"}, registry)
    assert '1000' in call   # spin_acc default
    assert '30000' in call  # spin_dur default
    assert '10' in call     # volume default
    print("PASS: test_default_values_applied")

def test_helper_steps_unchanged():
    code = compiler.compile_to_python({
        "experiment_name": "Test",
        "steps": [
            {"type": "helper", "name": "WAIT", "params": {"duration": 1000}, "description": "等待"}
        ]
    })
    assert "time.sleep(1.0)" in code
    print("PASS: test_helper_steps_unchanged")

def test_unknown_tool_does_not_crash():
    code = compiler.compile_to_python({
        "experiment_name": "Test",
        "steps": [
            {"type": "tool", "name": "nonexistent_tool", "params": {}}
        ]
    })
    assert "WARNING" in code
    print("PASS: test_unknown_tool_does_not_crash")

def test_backward_compat_action_field():
    steps = [{"type": "tool", "action": "spin_coating", "params": {"spin_speed": 3000, "reagent": "Test"}}]
    registry = compiler._load_registry()
    line = compiler._build_imports(steps, registry)
    assert "execute_spin_coating" in line
    print("PASS: test_backward_compat_action_field")

def test_full_compile_output():
    code = compiler.compile_to_python({
        "experiment_name": "完整测试",
        "steps": [
            {"type": "tool", "name": "move_robot_arm", "params": {"x": 100, "y": 200, "z": 300}, "description": "移动机械臂"},
            {"type": "helper", "name": "WAIT", "params": {"duration": 2000}},
            {"type": "tool", "name": "spin_coating", "params": {"spin_speed": 3000, "reagent": "Perovskite"}},
        ]
    })
    assert "from hardware import" in code
    assert "execute_move_robot_arm" in code
    assert "execute_spin_coating" in code
    assert "time.sleep" in code
    assert "def execute_experiment()" in code
    print("PASS: test_full_compile_output")

def test_empty_params_tool():
    registry = compiler._load_registry()
    call = compiler._build_tool_call("start_experiment", {}, registry)
    assert call == "execute_start_experiment()"
    print("PASS: test_empty_params_tool")

if __name__ == "__main__":
    test_registry_loads()
    test_import_generation()
    test_positional_call_generation()
    test_missing_required_param_raises()
    test_type_conversion()
    test_default_values_applied()
    test_helper_steps_unchanged()
    test_unknown_tool_does_not_crash()
    test_backward_compat_action_field()
    test_full_compile_output()
    test_empty_params_tool()
    print("\nAll tests passed!")
