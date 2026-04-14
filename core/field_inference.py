"""
字段推断模块
负责动态推断提取字段和生成文件名前缀
"""

import json
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field, ValidationError

from .config import Config
from .llm_client import LLMClient


class DynamicFieldsResponse(BaseModel):
    """
    动态字段响应模型
    """
    fields: List[str] = Field(description="推断出的需要提取的数据列名列表")


class FieldInference:
    """
    字段推断类 - 负责动态字段推断和文件名生成

    职责：
    - 根据任务描述推断提取字段
    - 生成英文文件名前缀
    - 处理LLM交互和验证
    """

    def __init__(self):
        """初始化字段推断器"""
        self.config = Config()
        self.llm_client = LLMClient()

    def infer_fields(self, task_description: str) -> Tuple[bool, List[str] | str]:
        """
        从任务描述推断字段

        Args:
            task_description: 任务描述

        Returns:
            (成功状态, 字段列表或错误信息)
        """
        schema_str = json.dumps(DynamicFieldsResponse.model_json_schema(), ensure_ascii=False)

        prompt = (
            f"你是一个文献数据抽取专家。用户希望进行以下信息提取任务：【{task_description}】。\n"
            "请推断为了完成这个任务，最终的数据表格需要包含哪些列名（字段）？\n"
            "🚨 你必须直接输出一个 JSON 对象，不要输出任何 Markdown 标记（如 ```json）、不要输出代码块，也不要输出任何解释性文字。\n"
            "🚨 你的输出必须严格符合以下格式：\n"
            '{"fields": ["列名1", "列名2", "列名3"]}\n'
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        success, result = self.llm_client.call_api_with_validation(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            response_model=DynamicFieldsResponse,
            temperature=0.1,
            max_tokens=1024
        )

        if success:
            return True, result.fields
        else:
            return False, result

    def get_filename_prefix(self, task_description: str) -> str:
        """
        从任务描述生成文件名前缀

        Args:
            task_description: 任务描述

        Returns:
            英文文件名前缀
        """
        prompt = f"将以下提取任务的核心关键词翻译为简短的英文（单词之间用下划线连接），仅输出英文，不要有其他字符。任务：{task_description}"

        messages = [
            {"role": "user", "content": prompt}
        ]

        result = self.llm_client.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )

        if result:
            try:
                content = result['choices'][0]['message']['content'].strip()
                prefix = content.replace(" ", "_").lower()
                # 清理非法字符
                prefix = "".join(c for c in prefix if c.isalnum() or c == "_")
                return prefix if prefix else "extraction_result"
            except:
                pass

        return "extraction_result"

    def validate_fields(self, fields: List[str]) -> bool:
        """
        验证字段列表是否有效

        Args:
            fields: 字段列表

        Returns:
            是否有效
        """
        if not fields or not isinstance(fields, list):
            return False

        # 检查字段是否都是非空字符串
        for field in fields:
            if not field or not isinstance(field, str):
                return False

        return True

    def get_default_fields(self) -> List[str]:
        """
        获取默认字段

        Returns:
            默认字段列表
        """
        return ["钝化剂名称", "原文原句", "作用机理", "文献来源"]

    def format_field_descriptions(self, fields: List[str]) -> Dict[str, str]:
        """
        格式化字段描述

        Args:
            fields: 字段列表

        Returns:
            字段描述字典
        """
        return {field: f"提取：{field}" for field in fields}

    def create_dynamic_model(self, fields: List[str]) -> BaseModel:
        """
        创建动态Pydantic模型

        Args:
            fields: 字段列表

        Returns:
            动态模型类
        """
        from pydantic import create_model, Field
        from typing import Optional

        field_definitions = {
            field: (Optional[str], Field(default="", description=f"提取：{field}"))
            for field in fields
        }

        return create_model('DynamicRecord', **field_definitions)


class IntentRecognizer:
    """
    意图识别器 - 判断用户是想要"设计实验"还是"单步控制硬件"

    职责：
    - 分析用户输入，判断意图类型
    - 返回意图类型和置信度
    - 使用 LLM 进行智能识别，关键词匹配作为后备
    """

    def __init__(self, llm_client: LLMClient = None):
        """
        初始化意图识别器

        Args:
            llm_client: LLM 客户端实例（可选，不传则创建新实例）
        """
        self.llm_client = llm_client or LLMClient()
        self.config = Config()

        # 意图关键词
        self.experiment_keywords = [
            "实验", "设计", "流程", "方案", "规划",
            "步骤", "完整", "自动", "帮我做", "帮我完成"
        ]
        self.hardware_keywords = [
            "移动", "转速", "温度", "测量", "设置",
            "控制", "执行", "启动", "停止", "调整"
        ]

    def recognize(self, user_input: str) -> Tuple[str, float]:
        """
        识别用户意图

        Args:
            user_input: 用户输入文本

        Returns:
            (intent, confidence) 元组
            - intent: "experiment_design" 或 "hardware_control"
            - confidence: 置信度 (0.0-1.0)
        """
        # 首先尝试使用 LLM 进行意图识别
        try:
            intent, confidence = self._recognize_with_llm(user_input)
            return intent, confidence
        except Exception:
            # LLM 调用失败，使用关键词匹配
            return self._recognize_with_keywords(user_input)

    def _recognize_with_llm(self, user_input: str) -> Tuple[str, float]:
        """
        使用 LLM 进行意图识别

        Args:
            user_input: 用户输入文本

        Returns:
            (intent, confidence) 元组
        """
        prompt = f"""
请判断用户的意图是"实验设计"还是"硬件控制"。

用户输入：{user_input}

判断标准：
- 实验设计：用户想要规划一个完整的实验流程，包含多个步骤，需要AI自主选择工具和参数
  特征：描述性需求、多步骤、需要规划、希望AI自动完成
  示例："设计一个旋涂实验"、"帮我做一个温度测试"、"规划一个光谱测量流程"

- 硬件控制：用户想要执行具体的单步硬件操作，明确指定了操作和参数
  特征：具体操作、明确参数、单步执行
  示例："移动到位置A"、"设置转速3000rpm"、"测量光谱"

请只返回 JSON 格式（不要用markdown代码块包裹）：
{{"intent": "experiment_design" 或 "hardware_control", "confidence": 0.0-1.0}}
"""

        messages = [{"role": "user", "content": prompt}]

        result = self.llm_client.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )

        if result:
            content = result['choices'][0]['message']['content'].strip()

            # 去除可能的 markdown 代码块包裹
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
                content = content.rsplit("```", 1)[0]

            data = json.loads(content)
            intent = data.get("intent", "hardware_control")
            confidence = float(data.get("confidence", 0.5))

            return intent, confidence

        # 如果 API 调用失败，抛出异常让外层捕获
        raise Exception("LLM API call failed")

    def _recognize_with_keywords(self, user_input: str) -> Tuple[str, float]:
        """
        使用关键词匹配进行意图识别（后备方案）

        Args:
            user_input: 用户输入文本

        Returns:
            (intent, confidence) 元组
        """
        exp_score = sum(1 for kw in self.experiment_keywords if kw in user_input)
        hw_score = sum(1 for kw in self.hardware_keywords if kw in user_input)

        if exp_score > hw_score:
            intent = "experiment_design"
            confidence = min(0.6 + exp_score * 0.1, 0.9)
        elif hw_score > exp_score:
            intent = "hardware_control"
            confidence = min(0.6 + hw_score * 0.1, 0.9)
        else:
            # 无法判断，默认为硬件控制
            intent = "hardware_control"
            confidence = 0.5

        return intent, confidence