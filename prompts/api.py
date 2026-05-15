"""
Prompts API — Flask Blueprint

提供 prompt 管理的 RESTful 接口:
- GET  /api/prompts              列出所有 prompt 元信息
- GET  /api/prompts/<name>       获取单个 prompt 详情
- PUT  /api/prompts/<name>       修改 prompt（写入 overrides/）
- POST /api/prompts/<name>/reset 重置单个 prompt
- POST /api/prompts/reload       重新加载全部
- POST /api/prompts/optimize     LLM 优化建议
- POST /api/prompts/test         试跑测试
"""

from flask import Blueprint, request, jsonify
from . import create_prompt_manager
from .manager import NoSuchPromptError, MissingVariableError

prompts_bp = Blueprint("prompts", __name__)


def _get_manager():
    """获取全局 PromptManager"""
    return create_prompt_manager()


def _get_optimizer():
    """懒加载 PromptOptimizer"""
    from .optimizer import PromptOptimizer
    from core.config import Config
    from core.llm_client import LLMClient

    manager = _get_manager()
    config = Config()
    llm_client = LLMClient()
    return PromptOptimizer(manager, llm_client)


# ═══════════════════════════════════════════════════════════════
# 列表
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts", methods=["GET"])
def list_prompts():
    """列出所有 prompt 元信息"""
    manager = _get_manager()
    category = request.args.get("category")
    prompts = manager.list_all(category=category)
    return jsonify({"success": True, "data": prompts, "total": len(prompts)})


# ═══════════════════════════════════════════════════════════════
# 详情
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts/<name>", methods=["GET"])
def get_prompt(name: str):
    """获取单个 prompt 完整信息"""
    manager = _get_manager()
    try:
        meta = manager.get_meta(name)
        return jsonify({"success": True, "data": meta})
    except NoSuchPromptError as e:
        return jsonify({"success": False, "error": str(e)}), 404


# ═══════════════════════════════════════════════════════════════
# 修改
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts/<name>", methods=["PUT"])
def update_prompt(name: str):
    """修改 prompt，写入 overrides/"""
    manager = _get_manager()
    data = request.get_json(silent=True) or {}

    try:
        kwargs = {}
        if "template" in data:
            kwargs["template"] = data["template"]
        if "variables" in data:
            kwargs["variables"] = data["variables"]
        if "description" in data:
            kwargs["description"] = data["description"]

        manager.update(name, **kwargs)
        return jsonify({"success": True, "message": f"Prompt '{name}' 已更新"})
    except NoSuchPromptError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# 重置
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts/<name>/reset", methods=["POST"])
def reset_prompt(name: str):
    """重置单个 prompt 到原始版本"""
    manager = _get_manager()
    try:
        manager.reset(name)
        return jsonify({"success": True, "message": f"Prompt '{name}' 已重置"})
    except NoSuchPromptError as e:
        return jsonify({"success": False, "error": str(e)}), 404


# ═══════════════════════════════════════════════════════════════
# 全部重新加载
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts/reload", methods=["POST"])
def reload_prompts():
    """重新加载所有 prompt"""
    manager = _get_manager()
    manager.reload()
    prompts = manager.list_all()
    return jsonify({
        "success": True,
        "message": f"已重新加载 {len(prompts)} 个 prompt",
        "total": len(prompts),
    })


# ═══════════════════════════════════════════════════════════════
# LLM 优化
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts/optimize", methods=["POST"])
def optimize_prompt():
    """请求 LLM 优化指定 prompt"""
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    requirements = data.get("requirements")
    test_inputs = data.get("test_inputs")

    if not name or not requirements:
        return jsonify({
            "success": False,
            "error": "缺少必填字段: name, requirements",
        }), 400

    try:
        optimizer = _get_optimizer()
        result = optimizer.optimize(name, requirements, test_inputs)
        return jsonify({
            "success": True,
            "data": {
                "name": name,
                "original": result.original,
                "optimized": result.optimized,
                "changes_summary": result.changes_summary,
            },
        })
    except NoSuchPromptError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

@prompts_bp.route("/api/prompts/test", methods=["POST"])
def test_prompt():
    """用测试输入跑 prompt"""
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    variables = data.get("variables", {})
    user_content = data.get("user_content")

    if not name:
        return jsonify({"success": False, "error": "缺少必填字段: name"}), 400

    manager = _get_manager()
    try:
        rendered = manager.get(name, **variables)
        response_data = {
            "rendered_prompt": rendered,
            "llm_response": None,
        }

        if user_content:
            from core.config import Config
            from core.llm_client import LLMClient
            config = Config()
            llm = LLMClient()
            messages = [
                {"role": "system", "content": rendered},
                {"role": "user", "content": user_content},
            ]
            success, result = llm.call_api_with_validation(
                model=config.MODEL_NAME_TALK,
                messages=messages,
                response_model=None,
                temperature=0.1,
                max_tokens=None,
            )
            # call_api fallback
            if result is None:
                result = llm.call_api(
                    model=config.MODEL_NAME_TALK,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=None,
                )
            response_data["llm_response"] = str(result) if result else ""

        return jsonify({"success": True, "data": response_data})
    except MissingVariableError as e:
        return jsonify({"success": False, "error": str(e), "missing": e.missing}), 400
    except NoSuchPromptError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
