"""
算法自动生成器 (extra_algorithms_fromProjects/prompt_template.py)
================================================================

使用大语言模型（LLM）根据用户的自然语言描述，自动生成符合接口规范的算法文件。

工作流（纯 Python 调用 LLM，无 Agent 框架）：

    Step 1  用户用自然语言描述需求
            ↓
    Step 2  调用 LLM 提取结构化算法规格（名称、输入格式、输出字段、参数）
            ↓
    Step 3  将规格填入代码生成 Prompt
            ↓
    Step 4  调用 LLM 生成完整的 Python 算法文件
            ↓
    Step 5  自动保存到当前目录
            ↓
    SoftwareController 下次实例化时自动注册该算法

对外接口：
    from software.algorithms.extra_algorithms_fromProjects.prompt_template import generate_algorithm

    result = generate_algorithm("我需要一个移动平均算法，输入是数值列表，窗口大小可配置")
    # result = {"success": True, "name": "moving_average", "filepath": "...", "message": "..."}
"""

import os
import re
import sys
import json
import requests

# 允许直接运行此文件进行测试
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 从 core 复用 API 配置
try:
    # 直接导入 config 模块，避免 core.__init__ 中的 pydantic_ai 依赖
    _config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core', 'config.py')
    if os.path.exists(_config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", _config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        _API_KEY = config_module.Config.API_KEY
        _API_URL = config_module.Config.API_URL  # 已包含完整路径，不需要拼接
        _MODEL   = config_module.Config.MODEL_NAME_TALK
    else:
        raise ImportError("Config file not found")
except Exception:
    # 如果 core 不可用（独立运行时），使用默认值
    _API_KEY = ""
    _API_URL = "https://api.longcat.chat/openai/v1/chat/completions"
    _MODEL   = "Qwen/Qwen2.5-7B-Instruct"

# 生成的算法文件存放到当前目录（extra_algorithms_fromProjects/）
_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# 引导用户描述算法需求的提示词
# ==============================================================================

USER_GUIDANCE_PROMPT = """
请描述您想要的算法，您可以参考以下方面进行说明：

1. 算法功能：这个算法要做什么？（例：对光谱数据做平滑处理）
2. 输入数据：数据长什么样？（例：dict 含 wavelength 和 intensity 两个列表）
3. 期望输出：希望得到哪些结果？（例：平滑后的强度序列、平滑程度）
4. 可调参数：有哪些可配置的参数？（例：窗口大小 window_size，默认 5）

您不需要提供完整的技术细节，用自然语言描述即可，系统会自动理解并生成代码。

示例描述：
    "我需要一个对数值列表做移动平均的算法，窗口大小可配置，默认 5，
     输出平滑后的序列和各点的原始值与平滑值之差（残差）"
"""


# ==============================================================================
# Step 2：提取算法规格（LLM 调用）
# ==============================================================================

_SPEC_EXTRACT_SYSTEM = """\
你是一个算法规格提取助手。用户会用自然语言描述一个数据处理算法需求。
请从中提取以下信息并以 JSON 格式返回，不要包含任何其他内容（不要用markdown代码块包裹）：
{
  "name": "算法英文标识（小写字母+下划线，如 moving_average，必须简洁且具有描述性）",
  "description": "算法功能的中文简短描述（1-2句话，清晰说明算法的作用）",
  "input_format": "输入 data 参数的格式说明（详细描述数据结构，例如：dict 含有 'values' 列表，或 list 数值序列）",
  "output_fields": ["result 字典中应包含的字段名1", "字段名2"],
  "params": [
    {
      "name": "参数英文名（小写字母+下划线）",
      "type": "float 或 int 或 str 或 bool 或 list",
      "description": "参数作用说明（清晰描述参数的用途）",
      "default": 默认值（必须是合法的 Python 字面量，如 5、1.0、"text"、true、[0, 1]）
    }
  ]
}

注意：
1. name 必须是有效的 Python 标识符，只能包含小写字母、数字和下划线，且不能以数字开头
2. description 要简洁明了，突出算法的核心功能
3. input_format 要详细说明数据结构，便于后续代码生成
4. output_fields 要列出所有期望的输出字段
5. params 中的 default 必须是合法的 JSON 值
"""


def extract_algorithm_spec(user_description: str) -> dict:
    """
    Step 2：将用户的自然语言描述转换为结构化算法规格

    Args:
        user_description: 用户描述算法需求的自然语言文本

    Returns:
        算法规格 dict：{"name", "description", "input_format", "output_fields", "params"}
    """
    raw = _call_llm(_SPEC_EXTRACT_SYSTEM, user_description)
    raw = _strip_markdown_code_block(raw)
    return json.loads(raw)


# ==============================================================================
# Step 3：构建代码生成 Prompt
# ==============================================================================

_CODE_GEN_SYSTEM = """\
你是一位 Python 数据科学专家，精通编写符合接口规范的算法模块。
请严格按照用户要求，输出完整的、可直接运行的 Python 文件代码。

重要要求：
1. 代码必须完整且可直接运行，不要留 TODO 或占位符
2. 算法逻辑必须正确实现，不要使用简化或模拟的实现
3. 只使用标准库、numpy、math，不要导入其他第三方库
4. 代码要有适当的注释，但不要有解释性文字
5. 必须包含完整的测试代码（if __name__ == "__main__" 部分）
6. 输出纯 Python 代码，不要用 markdown 代码块包裹
"""

_CODE_GEN_TEMPLATE = """\
请编写一个 Python 算法文件，满足以下所有要求：

## 算法规格
- 算法唯一标识（name）: {name}
- 功能描述: {description}
- 类名（驼峰命名）: {class_name}
- 输入数据格式: {input_format}
- 输出字段（result 字典应包含）: {output_fields}
- 算法参数:
{params_detail}

## 强制接口规范（必须严格遵守）

### 文件头部（必须原样包含）
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from software.algorithms.base import BaseAlgorithm
import numpy as np  # 如果需要数值计算
```

### 类定义（必须继承 BaseAlgorithm）
```python
class {class_name}(BaseAlgorithm):
    name = "{name}"
    description = "{description}"
    params_schema = {{
        # 根据上方参数规格填写，格式：
        # "param_name": {{"type": "int", "description": "参数说明", "default": 5, "required": False}}
    }}

    def run(self, data, params=None):
        \"\"\"
        执行算法

        Args:
            data: {input_format}
            params: 算法参数字典

        Returns:
            统一格式 dict，包含 success、result、message 字段
        \"\"\"
        params = params or {{}}

        try:
            # 1. 参数提取和验证
            # 从 params 中提取各个参数，使用默认值

            # 2. 输入数据验证
            # 检查 data 格式是否符合要求

            # 3. 算法核心逻辑实现
            # 实现具体的算法功能

            # 4. 构造输出结果
            result = {{
                # 根据 output_fields 构造输出字典
            }}

            return self._build_success(result, "算法执行成功")

        except Exception as e:
            return self._build_error(f"算法执行失败: {{str(e)}}")
```

### 文件末尾（必须包含可运行的测试）
```python
if __name__ == "__main__":
    import json

    algo = {class_name}()
    print(f"算法信息: {{algo.get_info()}}\\n")

    # 测试用例：构造符合 input_format 的示例数据
    test_data = # 填入具体的测试数据
    test_params = {{  # 填入测试参数
    }}

    result = algo.run(data=test_data, params=test_params)
    print("测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

## 重要提示
1. 算法逻辑必须完整实现，不要使用占位符或 TODO
2. 必须正确处理边界情况（空数据、异常值等）
3. 输出的 result 字典必须包含所有 output_fields 中列出的字段
4. 只使用标准库、numpy、math，不要导入其他库
5. 代码必须可以直接运行，测试用例要有实际数据
6. 直接输出 Python 代码，不要用 markdown 代码块包裹

请现在开始编写完整的算法代码：
"""


def build_code_generation_prompt(spec: dict) -> str:
    """
    Step 3：根据算法规格构建完整的代码生成提示词

    Args:
        spec: extract_algorithm_spec() 返回的规格字典

    Returns:
        发送给 LLM 的完整提示词字符串
    """
    class_name = "".join(w.capitalize() for w in spec["name"].split("_"))
    output_fields_str = "、".join(spec.get("output_fields", []))

    params = spec.get("params", [])
    if params:
        lines = []
        for p in params:
            lines.append(
                f"  - {p['name']} ({p['type']}): {p['description']}，默认值 {p['default']}"
            )
        params_detail = "\n".join(lines)
    else:
        params_detail = "  （无额外参数）"

    return _CODE_GEN_TEMPLATE.format(
        name          = spec["name"],
        description   = spec["description"],
        class_name    = class_name,
        input_format  = spec["input_format"],
        output_fields = output_fields_str,
        params_detail = params_detail,
    )


# ==============================================================================
# Step 4：调用 LLM 生成代码
# ==============================================================================

def generate_algorithm_code(spec: dict) -> str:
    """
    Step 4：调用 LLM 生成完整的算法 Python 代码

    Args:
        spec: 算法规格字典

    Returns:
        可直接写入文件的 Python 代码字符串
    """
    prompt = build_code_generation_prompt(spec)
    code = _call_llm(_CODE_GEN_SYSTEM, prompt)
    # 去除可能的 markdown 代码块包裹
    code = _strip_markdown_code_block(code)
    return code


# ==============================================================================
# Step 5：保存文件并更新注册表
# ==============================================================================

def save_algorithm_file(name: str, code: str) -> str:
    """
    Step 5：将生成的算法代码保存到 extra_algorithms_fromProjects/ 目录

    Args:
        name: 算法标识（作为文件名，如 "moving_average" → "moving_average.py"）
        code: 算法 Python 代码

    Returns:
        保存后的文件绝对路径
    """
    filename = f"{name}.py"
    filepath = os.path.join(_SAVE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    return filepath


def update_registry(spec: dict) -> bool:
    """
    更新自定义算法注册表

    Args:
        spec: 算法规格字典

    Returns:
        是否更新成功
    """
    registry_path = os.path.join(_SAVE_DIR, "REGISTRY.json")

    # 读取现有注册表
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {"algorithms": [], "version": "1.0.0", "last_updated": ""}
    else:
        registry = {"algorithms": [], "version": "1.0.0", "last_updated": ""}

    # 检查是否已存在同名算法
    algorithms = registry.get("algorithms", [])
    existing_index = None
    for i, algo in enumerate(algorithms):
        if algo.get("name") == spec["name"]:
            existing_index = i
            break

    # 构造新的算法条目
    new_entry = {
        "name": spec["name"],
        "description": spec["description"],
        "category": "自定义算法",
        "input_type": spec.get("input_format", "未指定"),
        "keywords": [spec["name"]]
    }

    # 更新或添加
    if existing_index is not None:
        algorithms[existing_index] = new_entry
    else:
        algorithms.append(new_entry)

    # 更新注册表
    registry["algorithms"] = algorithms
    registry["last_updated"] = "2026-04-15"

    # 保存注册表
    try:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# ==============================================================================
# 主入口：generate_algorithm()
# ==============================================================================

def generate_algorithm(user_description: str, verbose: bool = True) -> dict:
    """
    主入口：从用户描述自动生成算法文件

    完整流程：
        用户描述 → LLM提取规格 → 构建Prompt → LLM生成代码 → 保存文件

    Args:
        user_description: 用户自然语言描述的算法需求
        verbose         : 是否打印进度（默认 True）

    Returns:
        {
            "success" : bool,
            "name"    : str,         # 算法标识
            "filepath": str,         # 保存路径
            "spec"    : dict,        # 提取的算法规格
            "message" : str          # 说明信息
        }
    """
    def log(msg):
        if verbose:
            print(msg)

    try:
        log("[Step 1/4] 正在从用户描述提取算法规格...")
        spec = extract_algorithm_spec(user_description)
        log(f"           算法名称: {spec['name']}")
        log(f"           功能描述: {spec['description']}")

        log("[Step 2/4] 正在构建代码生成提示词...")
        # （内部步骤，对用户透明）

        log("[Step 3/4] 正在调用 LLM 生成算法代码...")
        code = generate_algorithm_code(spec)
        log(f"           代码已生成，共 {len(code.splitlines())} 行")

        log("[Step 4/4] 正在保存算法文件...")
        filepath = save_algorithm_file(spec["name"], code)
        log(f"           已保存到: {filepath}")

        log("[Step 5/5] 正在更新算法注册表...")
        registry_updated = update_registry(spec)
        if registry_updated:
            log(f"           注册表已更新")
        else:
            log(f"           注册表更新失败（不影响算法使用）")

        log(f"           重新实例化 SoftwareController 后算法将自动注册。")

        return {
            "success" : True,
            "name"    : spec["name"],
            "filepath": filepath,
            "spec"    : spec,
            "message" : f"算法 '{spec['name']}' 已生成并保存，调用 SoftwareController() 后可立即使用",
        }

    except json.JSONDecodeError as e:
        return {
            "success" : False,
            "name"    : "",
            "filepath": "",
            "spec"    : {},
            "message" : f"LLM 返回的规格 JSON 解析失败: {str(e)}，请重试",
        }
    except Exception as e:
        return {
            "success" : False,
            "name"    : "",
            "filepath": "",
            "spec"    : {},
            "message" : f"算法生成失败: {str(e)}",
        }


# ==============================================================================
# 内部工具函数
# ==============================================================================

def _call_llm(system_prompt: str, user_message: str) -> str:
    """调用 LLM API，返回模型回复文本"""
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type" : "application/json",
    }
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(_API_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _strip_markdown_code_block(text: str) -> str:
    """去除 LLM 回复中可能出现的 markdown 代码块包裹"""
    text = text.strip()
    text = re.sub(r"^```(?:python|json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("算法自动生成器 - 交互测试")
    print("=" * 60)
    print(USER_GUIDANCE_PROMPT)

    user_input = input("请输入您的算法需求描述（直接回车使用示例）：\n> ").strip()

    if not user_input:
        user_input = (
            "我需要一个对数值列表做滑动平均的算法，"
            "输入是 dict 含有 'values' 列表，"
            "参数是窗口大小 window_size（整数，默认 5），"
            "输出平滑后的序列 smoothed 和每点的残差 residuals"
        )
        print(f"\n使用示例描述:\n  {user_input}\n")

    result = generate_algorithm(user_input)

    print("\n" + "=" * 60)
    print("生成结果:")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["success"]:
        print(f"\n查看生成的算法文件: {result['filepath']}")
