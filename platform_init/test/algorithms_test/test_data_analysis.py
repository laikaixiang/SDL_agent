"""
测试数据分析模块交互
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("测试 1: 文件查找功能")
print("=" * 60)

from software.auto_analyze import _find_csv_file

test_cases = [
    "extraction.csv",
    "temporal/extraction.csv",
    "nonexistent.csv"
]

for test_path in test_cases:
    try:
        result = _find_csv_file(test_path)
        print(f"[OK] '{test_path}' -> {result}")
    except FileNotFoundError as e:
        print(f"[FAIL] '{test_path}' -> Not Found")
    except Exception as e:
        print(f"[ERROR] '{test_path}' -> {e}")

print("\n" + "=" * 60)
print("测试 2: API URL 配置")
print("=" * 60)

# 直接读取配置文件内容
with open('core/config.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if 'API_URL: str = "https://api.longcat.chat/openai/v1/chat/completions"' in content:
        print("[OK] API_URL configured correctly (full path)")
    else:
        print("[FAIL] API_URL configuration may have issues")

# 检查 auto_analyze.py 是否正确使用 API_URL
with open('software/auto_analyze.py', 'r', encoding='utf-8') as f:
    content = f.read()
    if '_API_URL = Config.API_URL  # 已包含完整路径' in content:
        print("[OK] auto_analyze.py uses API_URL correctly (no duplicate concatenation)")
    elif '_API_URL = Config.API_URL + "/chat/completions"' in content:
        print("[FAIL] auto_analyze.py still concatenates /chat/completions")
    else:
        print("[UNKNOWN] auto_analyze.py API_URL usage unknown")

# 检查 prompt_template.py
prompt_template_path = 'software/algorithms/extra_algorithms_fromProjects/prompt_template.py'
if os.path.exists(prompt_template_path):
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if '_API_URL = Config.API_URL  # 已包含完整路径' in content:
            print("[OK] prompt_template.py uses API_URL correctly (no duplicate concatenation)")
        elif '_API_URL = Config.API_URL + "/chat/completions"' in content:
            print("[FAIL] prompt_template.py still concatenates /chat/completions")
        else:
            print("[UNKNOWN] prompt_template.py API_URL usage unknown")

print("\n" + "=" * 60)
print("测试 3: app.py 路由配置")
print("=" * 60)

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

    if 'if user_message.startswith("数据分析："):' in content:
        print("[OK] Data analysis route configured")
    else:
        print("[FAIL] Data analysis route missing")

    if 'if user_message.startswith("生成算法："):' in content:
        print("[OK] Algorithm generation route configured")
    else:
        print("[FAIL] Algorithm generation route missing")

    if 'def handle_auto_analyze' in content:
        print("[OK] handle_auto_analyze function defined")
    else:
        print("[FAIL] handle_auto_analyze function missing")

    if 'def handle_generate_algorithm' in content:
        print("[OK] handle_generate_algorithm function defined")
    else:
        print("[FAIL] handle_generate_algorithm function missing")

print("\n" + "=" * 60)
print("测试 4: 前端模式配置")
print("=" * 60)

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

    if "setMode('analyze', '数据分析：', '📊 分析数据')" in content:
        print("[OK] Data analysis mode frontend configured correctly")
    else:
        print("[FAIL] Data analysis mode frontend configuration issue")

    if "setMode('generate_algo', '生成算法：', '🔧 生成算法')" in content:
        print("[OK] Algorithm generation mode frontend configured correctly")
    else:
        print("[FAIL] Algorithm generation mode frontend configuration issue")

    if 'id="csv-file-btn"' in content:
        print("[OK] CSV file picker button added")
    else:
        print("[FAIL] CSV file picker button missing")

    if 'id="analyze-submenu"' in content:
        print("[OK] Data analysis submenu added")
    else:
        print("[FAIL] Data analysis submenu missing")

print("\n" + "=" * 60)
print("测试 5: SoftwareController 算法生成接口")
print("=" * 60)

with open('software/software_controller.py', 'r', encoding='utf-8') as f:
    content = f.read()

    if 'def generate_algorithm(self, user_description: str)' in content:
        print("[OK] generate_algorithm method added")
    else:
        print("[FAIL] generate_algorithm method missing")

    if 'self._registry.clear()' in content and 'self._discover_algorithms()' in content:
        print("[OK] Algorithm reload logic implemented")
    else:
        print("[FAIL] Algorithm reload logic missing")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("All configuration checks completed.")
print("If all items show [OK], configuration is correct.")
print("If there are [FAIL] items, please check the corresponding files.")
