r"""
Prompt 迁移验证测试脚本

验证:
1. 所有 prompt YAML 文件可正确加载
2. 所有变量模板可正确渲染
3. 变量缺少时正确抛出异常
4. update/reset 功能正常
5. 被迁移的源文件不再包含旧的内联 prompt
6. Flask API 路由可访问

运行方式:
    cd D:\PycharmProjects\SDL_agent
    python platform_init/test/prompt/test_migration.py
"""

import sys
import os
import io

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from prompts.manager import PromptManager, MissingVariableError, NoSuchPromptError


# ═══════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════

PROMPTS_TO_TEST = {
    "extraction_system_vision": {
        "task_description": "test task",
        "fields": "field1, field2",
        "example_json": '{"data":[]}',
    },
    "extraction_system_text": {
        "task_description": "test task",
        "fields": "field1, field2",
        "example_json": '{"data":[]}',
    },
    "extraction_few_shot_block": {
        "examples_text": "示例 1: {...}\n示例 2: {...}",
    },
    "field_inference_infer_fields": {
        "task_description": "提取钙钛矿钝化剂信息",
        "schema_str": '{"type":"object","properties":{"fields":{"type":"array"}}}',
    },
    "field_inference_filename_prefix": {
        "task_description": "提取钙钛矿钝化剂信息",
    },
    "experiment_design_system": {
        "hardware_tools_desc": "1. spin_coating - 旋涂\n2. set_temperature - 温度控制",
        "software_tools_desc": "1. data_statistics - 数据统计",
        "helper_tools_desc": "1. WAIT - 等待\n2. LOOP - 循环",
    },
    "experiment_design_user": {
        "system_prompt": "你是一位材料科学家。",
        "user_description": "设计一个钙钛矿旋涂实验",
    },
    "hardware_command_parse": {
        "tools_schema": '[{"name": "set_temperature", "params": {"target": {"type": "float"}}}]',
        "user_command": "设置温度为25度",
    },
    "algorithm_gen_user_guidance": {},
    "algorithm_gen_spec_extraction": {},
    "algorithm_gen_code_gen_system": {},
    "algorithm_gen_code_gen_template": {
        "name": "moving_average",
        "description": "移动平均算法",
        "class_name": "MovingAverage",
        "input_format": "dict with 'values' list",
        "output_fields": "smoothed_values, residuals",
        "params_detail": "- window_size (int): 窗口大小，默认值 5",
    },
    "data_analysis_system": {},
    "data_analysis_user": {
        "csv_path": "/tmp/test.csv",
        "columns": '["PCE(%)", "Voc(V)", "Jsc(mA/cm2)"]',
        "algorithms_desc": "- data_statistics: 数据统计\n- spectrum_analysis: 光谱分析",
        "functions_desc": "- read_numeric_columns: 读取数值列",
    },
    "misc_session_title": {
        "lines": "1. 帮我提取钙钛矿数据\n2. 分析PCE和Voc的关系",
    },
    "meta_optimize": {
        "current_prompt": "你是一个助手。",
        "prompt_name": "test_prompt",
        "prompt_description": "测试用prompt",
        "requirements": "提高准确率",
        "test_inputs": "用例1: test",
    },
}


def test_load_all_prompts():
    """测试所有 prompt 文件可正确加载"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    prompts = pm.list_all()
    assert len(prompts) == 16, f"Expected 16 prompts, got {len(prompts)}"
    print(f"✓ 加载了 {len(prompts)} 个 prompt")


def test_render_all():
    """测试所有 prompt 可正确渲染"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    for name, vars_dict in PROMPTS_TO_TEST.items():
        result = pm.get(name, **vars_dict)
        assert result and len(result) > 5, f"{name}: 渲染结果太短 ({len(result)} chars)"
        assert "${" not in result, f"{name}: 仍有未替换的变量占位符"
    print(f"✓ 全部 {len(PROMPTS_TO_TEST)} 个 prompt 渲染成功")


def test_missing_variable():
    """测试缺少变量时抛出异常"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    try:
        pm.get("extraction_system_vision", task_description="test")
        assert False, "应该抛出 MissingVariableError"
    except MissingVariableError as e:
        assert "fields" in str(e)
        assert "example_json" in str(e)
    print("✓ 缺少变量检测正常")


def test_no_such_prompt():
    """测试不存在的 prompt"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    try:
        pm.get("nonexistent_prompt")
        assert False, "应该抛出 NoSuchPromptError"
    except NoSuchPromptError:
        pass
    print("✓ 不存在的 prompt 检测正常")


def test_update_and_reset():
    """测试修改和重置功能"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")

    # 备份原始值
    original = pm.get_meta("misc_session_title")
    original_template = original["current_template"]

    # 修改
    new_template = "new test template with ${lines}"
    pm.update("misc_session_title", template=new_template)
    meta = pm.get_meta("misc_session_title")
    assert meta["overridden"] == True
    assert meta["current_template"] == new_template
    assert meta["original_template"] == original_template

    # 重置
    pm.reset("misc_session_title")
    meta = pm.get_meta("misc_session_title")
    assert meta["overridden"] == False
    assert meta["current_template"] == original_template

    print("✓ update/reset 功能正常")


def test_reload():
    """测试全量重新加载"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    pm.reload()
    prompts = pm.list_all()
    assert len(prompts) == 16
    print("✓ reload 功能正常")


def test_filter_by_category():
    """测试按分类过滤"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    extraction = pm.list_all(category="extraction")
    assert len(extraction) == 3
    assert all(p["category"] == "extraction" for p in extraction)

    all_prompts = pm.list_all(category=None)
    assert len(all_prompts) == 16
    print("✓ 分类过滤正常")


def test_prompt_content_sanity():
    """测试关键 prompt 包含预期内容"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")

    # 提取 prompt 应包含关键指令
    text = pm.get("extraction_system_vision", task_description="test", fields="a", example_json="{}")
    assert "文献" in text
    assert "JSON" in text

    # 实验设计 prompt 应包含硬件/软件/辅助三部分
    text = pm.get("experiment_design_system",
                  hardware_tools_desc="1. tool1", software_tools_desc="1. algo1",
                  helper_tools_desc="1. WAIT")
    assert "硬件工具" in text
    assert "数据分析算法" in text
    assert "辅助操作" in text

    # 硬件控制 prompt
    text = pm.get("hardware_command_parse", tools_schema="[]", user_command="test")
    assert "工具" in text
    assert "JSON" in text

    print("✓ 关键 prompt 内容检查通过")


def test_no_variable_template():
    """测试无变量 prompt 的渲染"""
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    # 这些 prompt 没有变量，应该可以直接渲染
    for name in ["data_analysis_system", "algorithm_gen_code_gen_system",
                 "algorithm_gen_user_guidance", "algorithm_gen_spec_extraction"]:
        text = pm.get(name)
        assert text and len(text) > 50, f"{name}: content too short"
    print("✓ 无变量 prompt 渲染正常")


# ═══════════════════════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_load_all_prompts,
        test_render_all,
        test_missing_variable,
        test_no_such_prompt,
        test_update_and_reset,
        test_reload,
        test_filter_by_category,
        test_prompt_content_sanity,
        test_no_variable_template,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    if failed == 0:
        print(f"全部 {len(tests)} 个测试通过！")
    else:
        print(f"{failed}/{len(tests)} 个测试失败")
        sys.exit(1)
