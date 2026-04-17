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




class ExperimentDesignParser:
    """
    实验设计解析器 - 将自然语言转换为实验设计JSON

    职责：
    - 解析用户的实验需求描述
    - 生成统一格式的实验设计JSON
    - 提供实验设计智能体的系统提示词
    """

    # 实验设计智能体系统提示词
    EXPERIMENT_AGENT_SYSTEM_PROMPT = (
        "你是一位经验丰富的材料科学家，专门设计钙钛矿太阳能电池实验。\n"
        "你的任务是根据用户需求，设计详细的实验方案并输出JSON格式。\n\n"
        "可用的实验操作类型：\n"
        "1. spin_coating - 旋涂实验\n"
        "   参数: spin_speed(转速rpm), spin_acc(加速度rpm/s), spin_dur(时长ms), reagent(试剂名), volume(体积μL)\n"
        "2. set_temperature - 温度控制\n"
        "   参数: temperature(温度℃)\n"
        "3. move_robot_arm - 机械臂移动\n"
        "   参数: x, y, z(坐标mm)\n"
        "4. collect_spectrum - 光谱采集\n"
        "   参数: duration(时长秒)\n"
        "5. WAIT - 等待辅助操作\n"
        "   参数: duration(时长ms)\n\n"
        "输出格式要求：\n"
        "🚨 必须输出纯JSON，不要有Markdown标记（如```json）、代码块或解释文字。\n"
        "🚨 JSON格式：\n"
        "{\n"
        '  "experiment_name": "实验名称",\n'
        '  "description": "实验描述",\n'
        '  "steps": [\n'
        '    {"type": "tool", "name": "spin_coating", "params": {...}, "description": "步骤描述"},\n'
        '    {"type": "helper", "name": "WAIT", "params": {"duration": 5000}, "description": "等待5秒"},\n'
        '    ...\n'
        '  ],\n'
        '  "notes": "注意事项"\n'
        "}\n\n"
        "设计原则：\n"
        "- 旋涂步骤必须包含试剂名称和体积\n"
        "- 多步旋涂需要在步骤间添加WAIT\n"
        "- 温度设置应在旋涂前完成\n"
        "- 每个步骤必须有清晰的description说明\n"
    )

    def __init__(self):
        """初始化实验设计解析器"""
        self.config = Config()
        self.llm_client = LLMClient()

    def parse_experiment_design(self, user_description: str) -> Tuple[bool, Dict | str]:
        """
        从用户描述生成实验设计JSON

        Args:
            user_description: 用户的实验需求描述

        Returns:
            (成功状态, JSON字典或错误信息)
        """
        prompt = (
            f"{self.EXPERIMENT_AGENT_SYSTEM_PROMPT}\n\n"
            f"用户需求：{user_description}\n\n"
            "请根据上述需求设计实验方案，直接输出JSON格式。"
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        result = self.llm_client.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.3,
            max_tokens=2048
        )

        if result:
            try:
                content = result['choices'][0]['message']['content'].strip()
                # 清理可能的markdown标记
                content = content.replace("```json", "").replace("```", "").strip()
                experiment_json = json.loads(content)

                # 验证JSON格式
                if self.validate_experiment_json(experiment_json):
                    return True, experiment_json
                else:
                    return False, "生成的JSON格式不符合要求"
            except json.JSONDecodeError as e:
                return False, f"JSON解析失败: {str(e)}"
            except Exception as e:
                return False, f"处理失败: {str(e)}"

        return False, "API调用失败"

    def validate_experiment_json(self, experiment_json: Dict) -> bool:
        """
        验证实验设计JSON的有效性

        Args:
            experiment_json: 实验设计JSON

        Returns:
            是否有效
        """
        # 检查必需字段
        if "steps" not in experiment_json:
            return False

        steps = experiment_json.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            return False

        # 检查每个步骤
        for step in steps:
            if not isinstance(step, dict):
                return False
            if "type" not in step or "name" not in step:
                return False
            if "params" not in step:
                return False

        return True


class AlgorithmParser:
    """
    算法解析器 - 解析用户输入的算法名称

    职责：
    - 从用户输入中识别算法名称
    - 提供算法标签和图标映射
    - 支持关键词匹配和模糊识别
    """

    def __init__(self, llm_client: LLMClient = None):
        """
        初始化算法解析器

        Args:
            llm_client: LLM 客户端实例（可选）
        """
        self.llm_client = llm_client or LLMClient()
        self.config = Config()

        # 算法关键词映射
        self.algo_keywords = {
            'data_statistics': ['统计', 'statistics', '均值', '标准差', '相关性', '方差'],
            'data_normalization': ['归一化', 'normalization', '标准化', 'minmax', 'zscore', '正规化'],
            'spectrum_analysis': ['光谱', 'spectrum', '峰值', '基线', '波长', '吸收']
        }

        # 算法标签映射
        self.algo_tags = {
            'data_statistics': ['数值数据', '多列支持'],
            'data_normalization': ['预处理', '单列'],
            'spectrum_analysis': ['光谱数据', '波长-强度']
        }

        # 算法图标映射
        self.algo_icons = {
            'data_statistics': '📈',
            'data_normalization': '🔧',
            'spectrum_analysis': '🌈'
        }

    def parse(self, user_input: str, available_algorithms: List[Dict]) -> Dict[str, Any]:
        """
        解析用户输入，判断是否指定了算法名称

        Args:
            user_input: 用户输入文本
            available_algorithms: 可用算法列表

        Returns:
            解析结果字典：
            {
                "algorithm_found": bool,
                "algorithm": str (如果找到),
                "description": str (如果找到),
                "params": dict (如果找到),
                "icon": str (如果找到),
                "tags": list (如果找到),
                "available_algorithms": list (如果未找到)
            }
        """
        user_input_lower = user_input.lower().strip()

        # 尝试关键词匹配
        for algo_name, keywords in self.algo_keywords.items():
            if any(kw in user_input_lower for kw in keywords):
                # 在可用算法中查找
                algo_info = self._find_algorithm_info(algo_name, available_algorithms)
                if algo_info:
                    return {
                        "algorithm_found": True,
                        "algorithm": algo_name,
                        "description": algo_info.get('description', ''),
                        "params": algo_info.get('params_schema', {}),
                        "icon": self.get_icon(algo_name),
                        "tags": self.get_tags(algo_name)
                    }

        # 未找到匹配算法，返回所有可用算法（带标签和图标）
        enriched_algorithms = []
        for algo in available_algorithms:
            algo_copy = algo.copy()
            algo_copy['tags'] = self.get_tags(algo['name'])
            algo_copy['icon'] = self.get_icon(algo['name'])
            enriched_algorithms.append(algo_copy)

        return {
            "algorithm_found": False,
            "available_algorithms": enriched_algorithms
        }

    def _find_algorithm_info(self, algo_name: str, available_algorithms: List[Dict]) -> Dict | None:
        """
        在可用算法列表中查找指定算法的信息

        Args:
            algo_name: 算法名称
            available_algorithms: 可用算法列表

        Returns:
            算法信息字典或None
        """
        for algo in available_algorithms:
            if algo.get('name') == algo_name:
                return algo
        return None

    def get_tags(self, algo_name: str) -> List[str]:
        """
        获取算法标签

        Args:
            algo_name: 算法名称

        Returns:
            标签列表
        """
        return self.algo_tags.get(algo_name, ['通用'])

    def get_icon(self, algo_name: str) -> str:
        """
        获取算法图标

        Args:
            algo_name: 算法名称

        Returns:
            图标emoji
        """
        return self.algo_icons.get(algo_name, '📊')