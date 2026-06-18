"""
字段推断模块
负责动态推断提取字段和生成文件名前缀
"""

import json
import os
import re
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
        self.llm_client = LLMClient(
            api_key=self.config.TALK_API_KEY,
            api_url=self.config.TALK_API_URL,
        )

    def infer_fields(self, task_description: str, history: list = None) -> Tuple[bool, List[str] | str]:
        """
        从任务描述推断字段

        Args:
            task_description: 任务描述
            history: 对话历史 [{role: str, content: str}, ...]，用于修改/补充场景的记忆

        Returns:
            (成功状态, 字段列表或错误信息)
        """
        schema_str = json.dumps(DynamicFieldsResponse.model_json_schema(), ensure_ascii=False)

        from prompts import create_prompt_manager
        pm = create_prompt_manager(lang='zh')
        prompt = pm.get("field_inference_infer_fields", task_description=task_description, schema_str=schema_str)

        messages = (history or []) + [
            {"role": "user", "content": prompt}
        ]

        success, result = self.llm_client.call_api_with_validation(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            response_model=DynamicFieldsResponse,
            temperature=0.1,
            max_tokens=None,
        )

        if success:
            # 兜底: 强制包含 grounding 字段 (Step 1)
            if "原文原句" not in result.fields:
                result.fields.append("原文原句")
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
        from prompts import create_prompt_manager
        pm = create_prompt_manager(lang='zh')
        prompt = pm.get("field_inference_filename_prefix", task_description=task_description)

        messages = [
            {"role": "user", "content": prompt}
        ]

        result = self.llm_client.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
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




class ExperimentDesignAgent:
    """
    Experiment Design Agent - Converts natural language to experiment design JSON (Approach 2)

    Responsibilities:
    - Parse user's experiment requirements
    - Generate unified format experiment design JSON
    - Dynamically generate system prompts from registries (hardware tools + software algorithms + helper operations)

    This is the Approach 2 implementation using JSON + prompt-based method.
    Does not require Function Calling support, works with any LLM.
    """

    def __init__(self):
        """Initialize experiment design agent"""
        self.config = Config()
        self.llm_client = LLMClient(
            api_key=self.config.EXPERIMENT_API_KEY,
            api_url=self.config.EXPERIMENT_API_URL,
            extra_body=self.config.get_extra_body('EXPERIMENT'),
        )
        self.hardware_registry = self._load_hardware_registry()
        self.software_registry = self._load_software_registry()
        self.helper_registry = self._get_helper_registry()
        self.system_prompt = self._generate_system_prompt()

    def _load_hardware_registry(self) -> Dict:
        """从hardware/tools/REGISTRY.json加载硬件工具注册表"""
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        registry_path = os.path.join(project_root, "hardware", "tools", "REGISTRY.json")

        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[警告] 硬件工具注册表未找到: {registry_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"[错误] 硬件工具注册表JSON解析失败: {e}")
            return {}

    def _load_software_registry(self) -> List[Dict]:
        """从software模块加载算法注册表"""
        try:
            # TODO: 解耦后从software/algorithms/REGISTRY.json读取
            # 当前通过SoftwareController动态获取
            from software.software_controller import SoftwareController
            controller = SoftwareController()
            return controller.list_algorithms()
        except Exception as e:
            print(f"[警告] 软件算法注册表加载失败: {e}")
            return []

    def _get_helper_registry(self) -> Dict:
        """获取辅助操作注册表（内置定义）"""
        return {
            "WAIT": {
                "name": "WAIT",
                "description": "等待辅助操作 - 暂停指定时长",
                "params": {
                    "duration": {
                        "type": "int",
                        "description": "等待时长(ms)",
                        "required": True,
                        "default": 5000
                    }
                }
            },
            "LOOP": {
                "name": "LOOP",
                "description": "循环辅助操作 - 重复执行步骤（count模式）或按范围迭代（start/stop/step模式）",
                "params": {
                    "count": {
                        "type": "int",
                        "description": "循环次数（简单模式，与start/stop互斥）",
                        "required": False,
                        "default": 3
                    },
                    "var": {
                        "type": "str",
                        "description": "循环变量名，默认 _i",
                        "required": False,
                        "default": "_i"
                    },
                    "start": {
                        "type": "int",
                        "description": "起始值（范围模式，与count互斥）",
                        "required": False
                    },
                    "stop": {
                        "type": "int",
                        "description": "结束值（范围模式，不包含）",
                        "required": False
                    },
                    "step": {
                        "type": "int",
                        "description": "步长，默认 1",
                        "required": False,
                        "default": 1
                    }
                }
            },
            "GROUP": {
                "name": "GROUP",
                "description": "分组辅助操作 - 将多个步骤组合",
                "params": {}
            },
            "CONDITION": {
                "name": "CONDITION",
                "description": "条件辅助操作 - 根据条件执行",
                "params": {
                    "condition": {
                        "type": "str",
                        "description": "条件表达式",
                        "required": True
                    }
                }
            },
            "END": {
                "name": "END",
                "description": "结束标记 - 标记LOOP/GROUP/CONDITION的结束",
                "params": {}
            },
            "USER_INPUT": {
                "name": "USER_INPUT",
                "description": "用户输入 - 运行时请求用户输入",
                "params": {
                    "prompt": {
                        "type": "str",
                        "description": "提示信息",
                        "required": True
                    },
                    "var_name": {
                        "type": "str",
                        "description": "变量名",
                        "required": True
                    }
                }
            }
        }

    @staticmethod
    def _format_params_sig(params: dict) -> str:
        """将参数定义格式化为函数签名风格: param1: type, param2: type = default"""
        parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "str")
            default = pinfo.get("default")
            required = pinfo.get("required", False)
            if not required and default is not None:
                parts.append(f"{pname}: {ptype} = {default}")
            elif not required:
                parts.append(f"{pname}: {ptype} = ?")
            else:
                parts.append(f"{pname}: {ptype}")
        return ", ".join(parts) if parts else "无参数"

    def _generate_system_prompt(self) -> str:
        """动态生成系统提示词（函数签名格式，紧凑）"""
        hardware_tools_desc = []
        for idx, (name, info) in enumerate(self.hardware_registry.items(), 1):
            sig = self._format_params_sig(info.get("params", {}))
            hardware_tools_desc.append(f"{idx}. {name}({sig}) — {info['description']}")

        software_tools_desc = []
        for idx, algo in enumerate(self.software_registry, 1):
            sig = self._format_params_sig(algo.get("params_schema", {}))
            label = algo.get('chinese_name', algo['name'])
            software_tools_desc.append(f"{idx}. {algo['name']}({sig}) — {label}: {algo['description']}")

        helper_tools_desc = []
        for idx, (name, info) in enumerate(self.helper_registry.items(), 1):
            sig = self._format_params_sig(info.get("params", {}))
            helper_tools_desc.append(f"{idx}. {name}({sig}) — {info['description']}")

        from prompts import create_prompt_manager
        pm = create_prompt_manager(lang='zh')
        prompt = pm.get(
            "experiment_design_system",
            hardware_tools_desc="\n".join(hardware_tools_desc),
            software_tools_desc="\n".join(software_tools_desc) if software_tools_desc else "暂无可用算法",
            helper_tools_desc="\n".join(helper_tools_desc),
        )

        return prompt

    def parse_experiment_design(self, user_description: str) -> Tuple[bool, Dict | str]:
        """
        从用户描述生成实验设计JSON

        Args:
            user_description: 用户的实验需求描述

        Returns:
            (成功状态, JSON字典或错误信息)
        """
        from prompts import create_prompt_manager
        pm = create_prompt_manager(lang='zh')
        prompt = pm.get(
            "experiment_design_user",
            system_prompt=self.system_prompt,
            user_description=user_description,
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        result = self.llm_client.call_api(
            model=self.config.EXPERIMENT_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=None,
        )

        if result:
            try:
                content = result['choices'][0]['message']['content'].strip()
                print(f"[实验设计] LLM原始输出({len(content)}字符): {content[:300]}...")

                experiment_json = self._parse_experiment_json(content)

                if experiment_json is None:
                    diag = (
                        f"JSON解析失败: 无法从LLM响应提取JSON。"
                        f"长度={len(content)} 字符,首字符={content[:30]!r},"
                        f"首{{位置={content.find('{')},末}}位置={content.rfind('}')}。"
                        f"原始输出: {content[:200]}"
                    )
                    print(f"[实验设计] {diag}")
                    return False, diag

                # 规范化变量：为缺少 type 的变量从 default_value 推断类型
                self._normalize_variables(experiment_json)

                if self.validate_experiment_json(experiment_json):
                    return True, experiment_json
                else:
                    return False, "生成的JSON格式不符合要求"
            except Exception as e:
                import traceback
                traceback.print_exc()
                return False, f"处理失败: {str(e)}"

        return False, "API调用失败"

    @staticmethod
    def _sse_event(event_type: str, data) -> str:
        """将事件格式化为 SSE data 行"""
        payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        return f"data: {payload}\n\n"

    def parse_experiment_design_stream(self, user_description: str):
        """
        流式生成实验设计JSON，yield SSE 事件字符串（已格式化为 data: ...\\n\\n）

        Yields:
            SSE 事件字符串:
            - thinking_start: 思考开始 {"type": "thinking_start", "data": ""}
            - thinking_delta: 思考内容增量 {"type": "thinking_delta", "data": "..."}
            - thinking_end: 思考结束 {"type": "thinking_end", "data": "..."}
            - chunk: LLM输出的文本片段 {"type": "chunk", "data": "..."}
            - complete: 最终结果 {"type": "complete", "data": {...}}
            - error: 错误信息 {"type": "error", "data": "..."}
        """
        from prompts import create_prompt_manager
        pm = create_prompt_manager(lang='zh')
        prompt = pm.get(
            "experiment_design_user",
            system_prompt=self.system_prompt,
            user_description=user_description,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            typed_stream = self.llm_client.stream_typed(
                model=self.config.EXPERIMENT_MODEL_NAME,
                messages=messages,
                temperature=0.3,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield self._sse_event("error", f"API调用失败: {str(e)}")
            return

        thinking_started = False
        thinking_buf = ""
        full_content = ""

        try:
            for chunk_type, chunk_text in typed_stream:
                if chunk_type == 'reasoning':
                    if not thinking_started:
                        thinking_started = True
                        thinking_buf = ""
                        print("[实验设计-流式] 思考开始")
                        yield self._sse_event("thinking_start", "")
                    thinking_buf += chunk_text
                    yield self._sse_event("thinking_delta", thinking_buf)

                elif chunk_type == 'content':
                    if thinking_started:
                        print(f"[实验设计-流式] 思考结束 ({len(thinking_buf)} 字符)")
                        yield self._sse_event("thinking_end", thinking_buf)
                        thinking_started = False
                    full_content += chunk_text
                    yield self._sse_event("chunk", chunk_text)
        except GeneratorExit:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            if thinking_started:
                yield self._sse_event("thinking_end", thinking_buf)
            yield self._sse_event("error", f"流式读取失败: {str(e)}")
            return

        # Flush any remaining thinking (unlikely but safe)
        if thinking_started:
            yield self._sse_event("thinking_end", thinking_buf)

        content = full_content.strip()
        print(f"[实验设计-流式] LLM原始输出({len(content)}字符): {content[:300]}...")

        experiment_json = self._parse_experiment_json(content)

        if experiment_json is None:
            diag = (
                f"JSON解析失败: 无法从LLM响应提取JSON。"
                f"长度={len(content)} 字符,首字符={content[:30]!r},"
                f"首{{位置={content.find('{')},末}}位置={content.rfind('}')}。"
                f"原始输出: {content[:200]}"
            )
            print(f"[实验设计-流式] {diag}")
            yield self._sse_event("error", diag)
            return

        # 规范化变量：为缺少 type 的变量从 default_value 推断类型
        self._normalize_variables(experiment_json)

        if not self.validate_experiment_json(experiment_json):
            yield self._sse_event("error", "生成的JSON格式不符合要求")
            return

        import datetime
        experiment_json['created_at'] = datetime.datetime.now().isoformat()

        from experiment.format import ExperimentFormatConverter
        converter = ExperimentFormatConverter()
        visual_data = converter.json_to_visual(experiment_json)

        yield self._sse_event("complete", {
            "experiment_json": experiment_json,
            "visual_data": visual_data,
            "reply": (
                f"✅ 已生成实验设计方案：{experiment_json.get('experiment_name', '未命名实验')}\n\n"
                f"{experiment_json.get('description', '')}\n\n"
                f"共 {len(experiment_json.get('steps', []))} 个步骤，已推送到实验流程画布。"
                + (f"\n\n📊 包含 {len(experiment_json.get('variables', {}))} 个可配置变量，可在变量栏中修改默认值或导入CSV批量执行。"
                   if experiment_json.get('variables') else "")
            )
        })

    @staticmethod
    def _normalize_variables(experiment_json: dict) -> dict:
        """
        规范化 variables 字段：补 name、补 type、自动声明 LOOP 迭代变量

        AI 的 prompt 不要求输出 type，因此需要在后端补全。
        同时扫描步骤中的 LOOP helper，将其 var 自动加入 variables 列表。
        直接修改传入的 dict（浅层），同时返回该 dict。
        """
        from core.variable_resolver import VariableResolver
        variables = experiment_json.get("variables")
        if not variables or not isinstance(variables, dict):
            variables = {}
            experiment_json["variables"] = variables

        # 自动声明 LOOP 迭代变量
        steps = experiment_json.get("steps", [])
        for step in steps:
            if step.get("type") == "helper" and step.get("name") == "LOOP":
                params = step.get("params", {})
                loop_var = params.get("var", "_i")
                if loop_var and isinstance(loop_var, str) and loop_var not in variables:
                    start_val = params.get("start")
                    stop_val = params.get("stop")
                    step_val = params.get("step", 1)
                    # loop 变量：default_value=start, constraints=start-stop
                    var_def = {
                        "name": loop_var,
                        "type": "int",
                        "default_value": start_val if start_val is not None else 0,
                    }
                    constraints = {}
                    if start_val is not None:
                        constraints["min"] = start_val
                    if stop_val is not None:
                        constraints["max"] = stop_val
                    if step_val != 1:
                        constraints["step"] = step_val
                    if constraints:
                        var_def["constraints"] = constraints
                    variables[loop_var] = var_def

        # 规范化已有变量
        for var_name, var_def in list(variables.items()):
            if isinstance(var_def, dict):
                # 补 name 字段（变量名是 key，前端需要 name 字段）
                if "name" not in var_def:
                    var_def["name"] = var_name
                # 补 type 字段（从 default_value 推断）
                if "type" not in var_def:
                    dv = var_def.get("default_value")
                    var_def["type"] = VariableResolver._infer_type(dv)
                if "constraints" not in var_def:
                    var_def["constraints"] = {}

        return experiment_json

    def _parse_experiment_json(self, content: str):
        """多策略解析实验设计JSON,按从最直接到最宽松的顺序尝试"""
        if not content:
            return None

        # 策略1: 标准markdown清理
        cleaned = content.replace("```json", "").replace("```", "").strip()
        parsed = self._try_json_loads(cleaned)
        if parsed is not None:
            return parsed

        # 策略2: 提取最外层{...}块
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            block = content[start:end + 1]
            block = block.replace("```json", "").replace("```", "").strip()
            parsed = self._try_json_loads(block)
            if parsed is not None:
                return parsed

        # 策略3: 修复常见 LLM JSON 错误(尾随逗号、单引号、Python None/True/False)后再尝试
        repaired = self._repair_common_json_errors(cleaned)
        parsed = self._try_json_loads(repaired)
        if parsed is not None:
            return parsed

        # 策略4: 提取最外层 {...} 块后再次修复
        if start != -1 and end != -1 and end > start:
            repaired_block = self._repair_common_json_errors(block)
            parsed = self._try_json_loads(repaired_block)
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _try_json_loads(text: str):
        """静默的 json.loads 包装,失败返回 None"""
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _repair_common_json_errors(text: str) -> str:
        """修复 LLM 常见的 JSON 格式问题:
        - Python 风格 None/True/False → JSON null/true/false
        - 单引号字符串 → 双引号
        - 尾部逗号 (在 ] 或 } 之前)
        """
        if not text:
            return text
        # Python 字面量 → JSON
        text = re.sub(r'\bNone\b', 'null', text)
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        # 尾随逗号: ,] 或 ,}
        text = re.sub(r',(\s*[\]\}])', r'\1', text)
        return text

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