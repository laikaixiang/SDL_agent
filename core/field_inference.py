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