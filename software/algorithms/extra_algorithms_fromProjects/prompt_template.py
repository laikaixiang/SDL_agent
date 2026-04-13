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
    from core.config import Config
    _API_KEY = Config.API_KEY
    _API_URL = Config.API_URL + "/chat/completions"
    _MODEL   = Config.MODEL_NAME_TALK
except Exception:
    # 如果 core 不可用（独立运行时），使用默认值
    _API_KEY = ""
    _API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
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
请从中提取以下信息并以 JSON 格式返回，不要包含任何其他内容：
{
  "name": "算法英文标识（小写字母+下划线，如 moving_average）",
  "description": "算法功能的中文简短描述（1-2句话）",
  "input_format": "输入 data 参数的格式说明（清晰描述数据结构）",
  "output_fields": ["result 字典中应包含的字段名1", "字段名2"],
  "params": [
    {
      "name": "参数英文名",
      "type": "float 或 int 或 str 或 bool 或 list",
      "description": "参数作用说明",
      "default": 默认值
    }
  ]
}
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
请严格按照用户要求，输出完整的 Python 文件代码。
不要有任何解释文字、注释之外的内容，直接输出可运行的 Python 代码。
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
```
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from software.algorithms.base import BaseAlgorithm
```

### 类定义（必须继承 BaseAlgorithm）
```
class {class_name}(BaseAlgorithm):
    name = "{name}"
    description = "{description}"
    params_schema = {{
        # 根据上方参数规格填写，格式：
        # "param_name": {{"type": "...", "description": "...", "default": ..., "required": False}}
    }}

    def run(self, data, params=None):
        params = params or {{}}
        try:
            # 算法核心逻辑
            result = {{...}}
            return self._build_success(result, "执行成功")
        except Exception as e:
            return self._build_error(f"算法执行失败: {{str(e)}}")
```

### 文件末尾（必须包含可运行的测试）
```
if __name__ == "__main__":
    import json
    algo = {class_name}()
    print(f"算法信息: {{algo.get_info()}}\\n")
    result = algo.run(
        data=<填入符合 input_format 的具体示例数据>,
        params={{<填入参数示例>}}
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

### 其他约束
- run() 返回值必须通过 self._build_success() 或 self._build_error() 构造
- 只使用标准库、numpy、math（无需 import 其他第三方库）
- 代码完整可直接运行，不留 TODO 或占位符

请直接输出完整 Python 代码，不要 markdown 代码块包裹，不要解释文字。
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
# Step 5：保存文件
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
