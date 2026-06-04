"""
核心 - 统一工具执行器 (core/agent_tools.py)
===========================================

UnifiedToolExecutor 合并 hardware/tools/, software/algorithms/ 和内置
agent 工具到统一的 OpenAI tools 格式，供 AgentLoop 使用。

组件:
    AgentTool dataclass        — 统一工具描述（含 func / category / dangerous）
    BUILTIN_TOOLS              — ask_user（AgentLoop 会在 dispatch 时拦截）
    scan_hardware_tools()      — 从 hardware.ToolRegistry 导入硬件工具
    scan_software_algorithms() — 从 SoftwareController.list_algorithms() 导入软件算法
    UnifiedToolExecutor        — dispatch / build_openai_tools / is_hardware_tool / get / names
    create_main_executor()     — 工厂函数，合并全部三类工具

使用示例::

    from core.agent_tools import create_main_executor

    executor = create_main_executor()
    openai_tools = executor.build_openai_tools()
    result = executor.dispatch("spin_coating", {"spin_speed": 3000, ...})
"""

import json
from dataclasses import dataclass
from typing import Callable


# =============================================================================
# AgentTool 数据类
# =============================================================================

@dataclass
class AgentTool:
    """统一工具描述，包含 OpenAI JSON Schema 和执行函数引用"""
    name: str
    description: str
    parameters: dict              # OpenAI JSON Schema object
    required: list[str]           # 必填参数名
    func: Callable[[dict], str]   # 接收 args dict，返回结果字符串
    category: str                 # "builtin" | "hardware" | "software"
    dangerous: bool = False       # 是否为危险操作（硬件工具默认 True）


# =============================================================================
# 参数 schema 转换工具
# =============================================================================

_TYPE_MAP = {
    "int":   "integer",
    "float": "number",
    "str":   "string",
    "bool":  "boolean",
    "list":  "array",
    "dict":  "object",
}


def _param_to_json_schema(param_def: dict) -> dict:
    """将单个参数定义转为 JSON Schema property"""
    ptype = param_def.get("type", "string")
    schema_type = _TYPE_MAP.get(ptype, "string")
    prop = {"type": schema_type, "description": param_def.get("description", "")}
    if "default" in param_def:
        prop["default"] = param_def["default"]
    if ptype == "list":
        prop["items"] = {"type": "string"}
    return prop


def _params_to_json_schema(params: dict) -> dict:
    """将 Registry 格式的 params dict 转为 OpenAI JSON Schema"""
    properties = {}
    required_list = []
    for name, param_def in params.items():
        properties[name] = _param_to_json_schema(param_def)
        if param_def.get("required"):
            required_list.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required_list:
        schema["required"] = required_list
    return schema


# =============================================================================
# BUILTIN_TOOLS --- 内置 agent 工具
# =============================================================================

def _ask_user_func(args: dict) -> str:
    """No-op: AgentLoop 在 dispatch 时检测此返回值并拦截，暂停等待用户输入"""
    return "__ASK_USER_PENDING__"


def _resolve_pdf_path(pdf_path: str, cfg) -> str:
    """
    灵活解析 LLM 传来的 PDF 路径，处理以下情况：
    - 绝对路径
    - 'PDF_TARGET/foo.pdf' 形式（PDF_FOLDER 基名前缀，会导致双重拼接）
    - 'dialogue data/PDF_TARGET/foo.pdf' 形式
    - 仅文件名（在 PDF_FOLDER 下查找）
    - literature_registry 兜底（用于 sanitize 重命名后的文件名）
    """
    if not pdf_path:
        return pdf_path
    import os as _os

    if _os.path.isabs(pdf_path):
        return pdf_path

    # 去掉前导的 PDF_FOLDER 基名 + 分隔符，防止双重拼接
    folder_basename = _os.path.basename(cfg.PDF_FOLDER.rstrip("/\\"))
    for prefix in (folder_basename + "/", folder_basename + "\\"):
        if pdf_path.startswith(prefix):
            pdf_path = pdf_path[len(prefix):]
            break

    # 1) 尝试 PDF_FOLDER + 原路径
    resolved = _os.path.join(cfg.PDF_FOLDER, pdf_path)
    if _os.path.isfile(resolved):
        return resolved

    # 2) 兜底：literature_registry 用 current_filename 查
    try:
        import sqlite3 as _sqlite
        if _os.path.isfile(cfg.LITERATURE_REGISTRY_DB_PATH):
            with _sqlite.connect(cfg.LITERATURE_REGISTRY_DB_PATH) as conn:
                row = conn.execute(
                    "SELECT current_filename FROM literature_registry WHERE current_filename = ?",
                    (_os.path.basename(pdf_path),),
                ).fetchone()
                if row:
                    candidate = _os.path.join(cfg.PDF_FOLDER, row[0])
                    if _os.path.isfile(candidate):
                        return candidate
    except Exception:
        pass

    return resolved


def _list_available_pdfs(cfg) -> str:
    """列出 PDF_FOLDER 中可用 PDF 文件名（用于错误信息）"""
    import os as _os
    try:
        files = sorted(
            f for f in _os.listdir(cfg.PDF_FOLDER) if f.lower().endswith(".pdf")
        )
    except Exception:
        return "(无法列举)"
    if not files:
        return "(无文件)"
    shown = files[:10]
    suffix = f" ... (共 {len(files)} 个)" if len(files) > 10 else ""
    return "\n  - ".join([""] + shown) + suffix


def _search_literature_func(args: dict) -> str:
    """封装 SemanticSearch.search() 为工具函数"""
    from extract.semantic_search import SemanticSearch
    from extract.embedding_service import create_embedding_service
    from extract.vector_store import ChromaVectorStore
    from core.config import Config as _Config
    import os as _os
    import sqlite3 as _sqlite
    _cfg = _Config()
    emb = create_embedding_service()
    vs = ChromaVectorStore(persist_dir=_cfg.CHROMADB_PERSIST_DIR)
    sqlite_path = _os.path.join(_cfg.CHROMADB_PERSIST_DIR, "page_metadata.db")
    ss = SemanticSearch(emb, vs, sqlite_path)
    query = args.get("query", "")
    top_k = int(args.get("top_k", 10) or 10)
    results = ss.search(query, top_k=top_k)
    if not results:
        return "未找到相关文献"

    # 从 literature_registry 按 current_filename 反查真实标题
    title_map: dict = {}
    try:
        if _os.path.isfile(_cfg.LITERATURE_REGISTRY_DB_PATH):
            with _sqlite.connect(_cfg.LITERATURE_REGISTRY_DB_PATH) as conn:
                for row in conn.execute(
                    "SELECT current_filename, title FROM literature_registry"
                ).fetchall():
                    if row[0] and row[1]:
                        title_map[row[0]] = row[1]
    except Exception:
        pass

    lines = []
    for i, r in enumerate(results[:top_k], 1):
        # SemanticSearch 返回 pdf_name/similarity,不是 title/score
        pdf_name = r.get("pdf_name", "")
        title = title_map.get(pdf_name, "") or pdf_name or "未知"
        score = r.get("similarity", 0) or 0
        lines.append(f"{i}. {title} (相关度: {score:.2f})")
    return "\n".join(lines)


def _design_experiment_func(args: dict) -> str:
    """封装 ExperimentDesignAgent 为工具函数"""
    from core.field_inference import ExperimentDesignAgent
    agent = ExperimentDesignAgent()
    description = args.get("description", "")
    success, result = agent.parse_experiment_design(description)
    if not success:
        return f"实验设计失败: {result}"
    import json as _json
    return _json.dumps(result, ensure_ascii=False, indent=2)


def _generate_algorithm_func(args: dict) -> str:
    """封装 AlgorithmGuide 为工具函数"""
    from extract.algorithm_guide import AlgorithmGuide
    from core.config import Config as _Cfg
    guide = AlgorithmGuide()
    description = args.get("description", "")
    # Start the guide
    result = guide.start(description)
    if result.get("error"):
        return f"算法生成失败: {result['error']}"
    import json as _json
    return _json.dumps(result, ensure_ascii=False)


def _extract_from_pdf_func(args: dict) -> str:
    """同步提取 PDF 中指定页面的结构化数据（使用 VL 模型）"""
    from core.extract_manager import PDFProcessor
    from core.config import Config as _Cfg
    from core.llm_client import LLMClient
    from prompts import create_prompt_manager as _get_pm
    import json as _json
    import os as _os

    _cfg = _Cfg()
    pdf_path = args.get("pdf_path", "")
    task_description = args.get("task_description", "")
    fields = args.get("fields", None)
    pages = args.get("pages", None)

    pdf_path = _resolve_pdf_path(pdf_path, _cfg)

    if not _os.path.isfile(pdf_path):
        available = _list_available_pdfs(_cfg)
        return _json.dumps(
            {"error": f"PDF 文件不存在: {pdf_path}\n可用的文件:{available}"},
            ensure_ascii=False,
        )

    processor = PDFProcessor()

    # Get PDF info
    info = processor.get_pdf_info(pdf_path)
    if not info:
        return _json.dumps({"error": f"无法读取 PDF: {pdf_path}"}, ensure_ascii=False)

    num_pages = info.get("num_pages", 0)
    filename = _os.path.basename(pdf_path)

    # Determine pages to extract
    if pages and isinstance(pages, list):
        target_pages = [p - 1 for p in pages if 1 <= p <= num_pages]  # convert to 0-based
    else:
        # Default: first 3 pages
        target_pages = list(range(min(3, num_pages)))

    if not target_pages:
        return _json.dumps({"error": "没有有效的目标页面"}, ensure_ascii=False)

    # Infer fields if not provided
    if not fields or not isinstance(fields, list) or len(fields) == 0:
        from core.field_inference import FieldInference
        fi = FieldInference()
        ok, inferred = fi.infer_fields(task_description)
        if ok and isinstance(inferred, list):
            fields = inferred
        else:
            fields = ["提取结果"]

    # Prepare VL LLM client
    vl_client = LLMClient(
        api_key=_cfg.VL_API_KEY,
        api_url=_cfg.VL_API_URL,
        extra_body=_cfg.get_extra_body("VL"),
    )

    pm = _get_pm(lang="zh")
    example_item = {f: "提取的内容" for f in fields}
    example_json = _json.dumps({"data": [example_item]}, ensure_ascii=False)

    sys_prompt = pm.get(
        "extraction_system_vision",
        task_description=task_description,
        fields=str(fields),
        example_json=example_json,
    )

    all_records: list[dict] = []
    errors: list[str] = []

    for page_idx in target_pages:
        page_num = page_idx + 1  # 1-based for display
        try:
            img_b64 = processor.pdf_page_to_image(pdf_path, page_idx)
            if not img_b64:
                errors.append(f"第 {page_num} 页: 图片转换失败")
                continue

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]},
            ]

            result = vl_client.call_api(
                model=_cfg.MODEL_NAME_VL,
                messages=messages,
                temperature=0.1,
                timeout=120,
            )

            if result:
                content = result["choices"][0]["message"]["content"].strip()
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = _json.loads(content)
                    data_list = parsed.get("data", [parsed] if isinstance(parsed, dict) else [])
                    for item in data_list:
                        if isinstance(item, dict):
                            item["_source_doc"] = filename
                            item["_source_page"] = page_num
                            all_records.append(item)
                except _json.JSONDecodeError:
                    # Partial extraction: store raw text as a record
                    all_records.append({
                        "_source_doc": filename,
                        "_source_page": page_num,
                        "_raw": content[:500],
                    })
            else:
                errors.append(f"第 {page_num} 页: API 调用失败")
        except Exception as e:
            errors.append(f"第 {page_num} 页: {str(e)}")

    return _json.dumps({
        "filename": filename,
        "total_pages": num_pages,
        "extracted_pages": len(target_pages),
        "records": len(all_records),
        "fields": fields,
        "data": all_records[:20],  # limit to 20 records to avoid huge responses
        "errors": errors if errors else None,
    }, ensure_ascii=False, indent=2)


def _preview_pdf_page_func(args: dict) -> str:
    """获取PDF页面预览（base64图片）"""
    from core.extract_manager import PDFProcessor
    from core.config import Config as _Cfg
    import os as _os
    _cfg = _Cfg()
    pdf_path = args.get("pdf_path", "")
    page_num = args.get("page_num", 1)

    pdf_path = _resolve_pdf_path(pdf_path, _cfg)

    if not _os.path.isfile(pdf_path):
        available = _list_available_pdfs(_cfg)
        return f"PDF 文件不存在: {pdf_path}\n可用的文件:{available}"

    processor = PDFProcessor()
    try:
        info = processor.get_pdf_info(pdf_path)
        if not info:
            return f"无法读取 PDF: {pdf_path}"
        total = info.get("num_pages", info.get("total_pages", 0)) or 0
        if page_num < 1 or page_num > total:
            return f"错误: 页码 {page_num} 超出范围 (1-{total})"

        image_b64 = processor.pdf_page_to_image(pdf_path, page_num)
        if not image_b64:
            return f"无法生成第 {page_num} 页的预览图: {pdf_path}"
        return f"PDF: {_os.path.basename(pdf_path)}, 第{page_num}/{total}页, 图片已加载 ({len(image_b64)} chars base64)"
    except Exception as e:
        return f"预览失败: {str(e)}"


BUILTIN_TOOLS: list[AgentTool] = [
    AgentTool(
        name="ask_user",
        description=(
            "向用户提问以澄清意图、确认危险操作或在多个策略中选择。"
            "当指令不够明确或存在多种可行方案时，应使用此工具请求用户确认。"
            "注意：不要使用此工具询问已由其他工具参数明确指定的范围或取值。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "向用户提出的问题"},
                "options": {"type": "string", "description": "可选的 JSON 数组字符串"},
            },
            "required": ["question"],
        },
        required=["question"],
        func=_ask_user_func,
        category="builtin",
    ),
    AgentTool(
        name="search_literature",
        description="在文献库中进行语义搜索，返回相关文献列表及相似度评分。用于查找特定研究方向的PDF文献。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询关键词"},
                "top_k": {"type": "integer", "description": "返回结果数量，默认 10"},
            },
            "required": ["query"],
        },
        required=["query"],
        func=_search_literature_func,
        category="builtin",
    ),
    AgentTool(
        name="design_experiment",
        description="根据自然语言描述自动生成实验设计方案（JSON格式，含步骤列表）。用于实验规划和设计。",
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "实验需求的自然语言描述"},
            },
            "required": ["description"],
        },
        required=["description"],
        func=_design_experiment_func,
        category="builtin",
    ),
    AgentTool(
        name="generate_algorithm",
        description="根据需求描述自动生成Python数据分析算法代码。用于创建新的数据处理算法。",
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "算法需求的自然语言描述"},
            },
            "required": ["description"],
        },
        required=["description"],
        func=_generate_algorithm_func,
        category="builtin",
    ),
    AgentTool(
        name="extract_from_pdf",
        description="从指定PDF文件中提取结构化实验数据。给定PDF路径和提取任务描述，返回提取的CSV数据摘要。",
        parameters={
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "PDF文件的完整路径或相对于PDF_TARGET的路径"},
                "task_description": {"type": "string", "description": "提取任务描述，如'提取钙钛矿钝化剂的名称和效率'"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "可选，指定提取的字段列表"},
                "pages": {"type": "array", "items": {"type": "integer"}, "description": "可选，指定提取的页码范围(从1开始)"},
            },
            "required": ["pdf_path", "task_description"],
        },
        required=["pdf_path", "task_description"],
        func=_extract_from_pdf_func,
        category="extraction",
    ),
    AgentTool(
        name="preview_pdf_page",
        description="获取PDF指定页面的预览信息（页码、总页数）。用于在提取之前确认PDF内容和页面范围。",
        parameters={
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "PDF文件路径"},
                "page_num": {"type": "integer", "description": "页码，从1开始，默认1"},
            },
            "required": ["pdf_path"],
        },
        required=["pdf_path"],
        func=_preview_pdf_page_func,
        category="extraction",
    ),
]


# =============================================================================
# 硬件工具扫描与分发
# =============================================================================

def _dispatch_hardware(name: str, args: dict) -> str:
    """
    分发硬件工具调用

    从 hardware.ToolRegistry 查找工具定义，按 registry params 顺序构建
    kwargs（对缺失的 optional 参数填入 default），调用实际函数。

    Args:
        name: 工具名称
        args: LLM 传入的参数 dict

    Returns:
        硬件工具执行结果字符串
    """
    from hardware import ToolRegistry

    entry = ToolRegistry.get_tool(name)
    if entry is None:
        return f"错误: 未找到硬件工具 '{name}'"

    params_def = entry.get("params", {})
    kwargs = {}

    for param_name, param_def in params_def.items():
        if param_name in args:
            kwargs[param_name] = args[param_name]
        elif not param_def.get("required", False):
            # 可选参数：有 default 就填 default
            if "default" in param_def:
                kwargs[param_name] = param_def["default"]
        # required 但未提供：不填 kwargs，让函数自身处理（触发 TypeError）

    try:
        func = entry["function"]
        result = func(**kwargs)
        return str(result)
    except Exception as e:
        return f"硬件工具 '{name}' 执行错误: {str(e)}"


def scan_hardware_tools() -> list[AgentTool]:
    """
    扫描 hardware.ToolRegistry 中所有已注册工具，转为 AgentTool 列表

    注意: hardware.__init__.py 在 import 时已调用 discover_tools()，
    因此导入时 ToolRegistry 已填充完毕。

    Returns:
        AgentTool 列表，每个工具的 category="hardware", dangerous=True
    """
    from hardware import ToolRegistry

    tools: list[AgentTool] = []
    entries = ToolRegistry.get_all()

    for name, entry in entries.items():
        params_def = entry.get("params", {})
        schema = _params_to_json_schema(params_def)
        required_list = schema.get("required", [])

        # 闭包捕获 tool_name，避免循环变量延迟绑定问题
        def _make_func(tool_name: str):
            def _func(args: dict) -> str:
                return _dispatch_hardware(tool_name, args)
            return _func

        tool = AgentTool(
            name=name,
            description=entry.get("description", ""),
            parameters=schema,
            required=required_list,
            func=_make_func(name),
            category="hardware",
            dangerous=True,
        )
        tools.append(tool)

    return tools


# =============================================================================
# 软件算法扫描与分发
# =============================================================================

def _dispatch_software(name: str, args: dict) -> str:
    """
    分发软件算法调用

    创建 SoftwareController 实例，将 args 中的 data 字段提取为算法输入，
    其余字段作为算法参数传入。

    Args:
        name: 算法名称
        args: LLM 传入的参数 dict（包含 data 和算法参数）

    Returns:
        JSON 字符串，格式: {"success": bool, "algorithm": str, "result": ..., "message": str}
    """
    from software.software_controller import SoftwareController

    controller = SoftwareController()
    args_copy = dict(args)  # 不修改调用方的 dict
    data = args_copy.pop("data", None)
    params = args_copy if args_copy else None

    try:
        result = controller.run_algorithm(name, data=data, params=params)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "algorithm": name,
            "result": None,
            "message": f"算法 '{name}' 执行异常: {str(e)}",
        }, ensure_ascii=False)


def scan_software_algorithms() -> list[AgentTool]:
    """
    扫描 SoftwareController 中所有已注册算法，转为 AgentTool 列表

    自动向每个算法的 OpenAI schema 中添加 "data" 必填字段
    （算法执行必需的输入数据）。

    Returns:
        AgentTool 列表，每个工具的 category="software", dangerous=False
    """
    from software.software_controller import SoftwareController

    controller = SoftwareController()
    algos = controller.list_algorithms()

    tools: list[AgentTool] = []
    for algo in algos:
        params_schema = algo.get("params_schema", {})
        schema = _params_to_json_schema(params_schema)

        # 软件算法需要输入数据 —— 向 schema 中添加 data 字段
        schema.setdefault("properties", {})
        schema["properties"]["data"] = {
            "type": "object",
            "description": "算法的输入数据（dict / list / 由具体算法定义格式）",
        }
        required_list = schema.get("required", [])
        required_list.append("data")
        schema["required"] = required_list

        name = algo["name"]

        def _make_func(tool_name: str):
            def _func(args: dict) -> str:
                return _dispatch_software(tool_name, args)
            return _func

        tool = AgentTool(
            name=name,
            description=algo.get("description", ""),
            parameters=schema,
            required=required_list,
            func=_make_func(name),
            category="software",
            dangerous=False,
        )
        tools.append(tool)

    return tools


# =============================================================================
# UnifiedToolExecutor --- 统一工具执行器
# =============================================================================

class UnifiedToolExecutor:
    """
    统一工具执行器

    合并内置工具 / 硬件工具 / 软件算法，提供统一的 dispatch + 查询接口。
    AgentLoop 通过此对象与所有工具交互。

    使用示例::

        from core.agent_tools import create_main_executor
        exec = create_main_executor()

        # 构建 OpenAI tools 参数
        openai_tools = exec.build_openai_tools()

        # LLM 返回 tool_call 后分发执行
        result = exec.dispatch("spin_coating", {"spin_speed": 3000, ...})

        # 查询
        exec.is_hardware_tool("drop")  # → True
        exec.get("data_statistics")     # → AgentTool | None
        exec.names                      # → ["ask_user", "drop", ...]
    """

    def __init__(self, tools: list[AgentTool]):
        self._tools: dict[str, AgentTool] = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        """所有已注册工具的名称列表"""
        return list(self._tools.keys())

    def build_openai_tools(self) -> list[dict]:
        """
        构建 OpenAI tools 参数格式

        Returns:
            [{"type":"function","function":{"name":...,"description":...,"parameters":...}}, ...]
        """
        result: list[dict] = []
        for tool in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result

    def dispatch(self, name: str, arguments: dict) -> str:
        """
        按名称查找工具并执行

        Args:
            name: 工具名称
            arguments: 工具参数 dict

        Returns:
            工具执行结果字符串
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"错误: 未找到工具 '{name}'"
        try:
            return tool.func(arguments)
        except Exception as e:
            return f"工具 '{name}' 执行错误: {str(e)}"

    def is_hardware_tool(self, name: str) -> bool:
        """判断给定名称是否对应一个硬件工具"""
        tool = self._tools.get(name)
        return tool is not None and tool.category == "hardware"

    def get(self, name: str) -> AgentTool | None:
        """按名称查找 AgentTool，未找到返回 None"""
        return self._tools.get(name)


# =============================================================================
# Factory
# =============================================================================

def create_main_executor() -> UnifiedToolExecutor:
    """
    工厂函数：扫描硬件工具 + 软件算法 + 合并内置工具，创建 UnifiedToolExecutor

    启动时打印工具扫描摘要:
        [AgentTools] Found N hardware tools: [...]
        [AgentTools] Found N software algorithms: [...]
        [AgentTools] Total: N tools

    Returns:
        已填充所有工具的 UnifiedToolExecutor 实例
    """
    hw_tools = scan_hardware_tools()
    hw_names = [t.name for t in hw_tools]
    print(f"[AgentTools] Found {len(hw_tools)} hardware tools: {hw_names}")

    sw_tools = scan_software_algorithms()
    sw_names = [t.name for t in sw_tools]
    print(f"[AgentTools] Found {len(sw_tools)} software algorithms: {sw_names}")

    all_tools = list(BUILTIN_TOOLS) + hw_tools + sw_tools
    print(f"[AgentTools] Total: {len(all_tools)} tools")

    return UnifiedToolExecutor(all_tools)
