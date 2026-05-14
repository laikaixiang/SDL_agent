"""
实验执行器 - 执行实验方案并管理硬件/算法调用

职责：
- 执行JSON格式的实验方案
- 调用硬件工具（旋涂、温控、机械臂等）
- 调用数据分析算法
- 实验方案验证
- 实时进度反馈
"""
import json
import os
import time
from typing import Callable, Optional


class ExperimentExecutor:
    """
    实验执行器

    职责：
    - 解析并执行JSON格式的实验方案
    - 调用硬件工具和数据分析算法
    - 实验方案验证
    - 提供实时进度反馈
    """

    def __init__(self, software_manager: Optional["SoftwareManager"] = None, hardware_agent=None):
        # 辅助操作映射
        self.helper_map = {
            "WAIT":       self._execute_wait,
            "LOOP":       self._execute_loop,
            "GROUP":      self._execute_group,
            "CONDITION":  self._execute_condition,
            "END":        self._execute_end,
            "USER_INPUT": self._execute_user_input,
        }

        self._software_manager = software_manager
        self._hardware_agent = hardware_agent
        if self._hardware_agent is None:
            from core.hardware_controller import HardwareAgent
            self._hardware_agent = HardwareAgent()

    # ========== 执行方法 ==========

    def execute_plan(self, plan_json: dict, progress_callback: Optional[Callable] = None) -> dict:
        """
        执行实验方案

        Args:
            plan_json: 实验方案JSON
                格式: {
                    "experiment_name": "...",
                    "description": "...",
                    "steps": [
                        {
                            "step_number": 1,
                            "description": "...",
                            "action": "spin_coating",
                            "params": {...}
                        },
                        ...
                    ],
                    "notes": "..."
                }
            progress_callback: 进度回调函数
                签名: callback(step_number, status, message)
                status: "running" | "completed" | "error"

        Returns:
            dict: 执行结果
                {
                    "success": bool,
                    "results": [
                        {
                            "step": int,
                            "action": str,
                            "result": str,
                            "success": bool
                        },
                        ...
                    ],
                    "error": str | None
                }
        """
        results = []

        try:
            steps = plan_json.get("steps", [])

            if not steps:
                return {
                    "success": False,
                    "results": [],
                    "error": "实验方案中没有步骤"
                }

            print(f"[执行器] 开始执行实验: {plan_json.get('experiment_name', '未命名')}")
            print(f"[执行器] 共 {len(steps)} 个步骤")

            # 逐步执行
            for idx, step in enumerate(steps):
                step_num = step.get("step_number", idx + 1)

                # 支持两种格式：旧格式用action，新格式用type+name
                step_type = step.get("type", "tool")
                action = step.get("action") or step.get("name")
                params = step.get("params", {})
                description = step.get("description", "")

                print(f"[执行器] 步骤 {step_num}: {description} ({action})")

                if progress_callback:
                    progress_callback(step_num, "running", f"正在执行: {description}")

                # 处理软件算法步骤
                if step_type == "software":
                    sw_result = self._execute_software_algorithm(step)
                    is_success = sw_result.get("success", False)
                    result_msg = sw_result.get("message", "算法执行完成" if is_success else "算法执行失败")
                    results.append({
                        "step": step_num,
                        "action": action,
                        "description": description,
                        "result": result_msg,
                        "detail": sw_result.get("result"),
                        "success": is_success
                    })
                    if progress_callback:
                        progress_callback(step_num, "completed" if is_success else "error", result_msg)
                    print(f"[执行器] 步骤 {step_num} {'成功' if is_success else '失败'}: {result_msg}")
                    continue

                # 处理辅助操作（如WAIT）
                if step_type == "helper":
                    if action in self.helper_map:
                        try:
                            result = self.helper_map[action](params)
                            results.append({
                                "step": step_num,
                                "action": action,
                                "description": description,
                                "result": result,
                                "success": True
                            })
                            if progress_callback:
                                progress_callback(step_num, "completed", result)
                            print(f"[执行器] 步骤 {step_num} 成功: {result}")
                        except Exception as e:
                            error_msg = f"执行失败: {str(e)}"
                            results.append({
                                "step": step_num,
                                "action": action,
                                "description": description,
                                "result": error_msg,
                                "success": False
                            })
                            if progress_callback:
                                progress_callback(step_num, "error", error_msg)
                            print(f"[执行器] 步骤 {step_num} 异常: {e}")
                    continue

                # 执行工具操作 - 通过 HardwareAgent 统一入口
                if self._hardware_agent.is_known_tool(action):
                    try:
                        agent_result = self._hardware_agent.execute_tool_call({
                            "name": action,
                            "params": params
                        })
                        result = agent_result.get("result", "") or agent_result.get("message", "")
                        is_success = agent_result.get("status") == "success"

                        results.append({
                            "step": step_num,
                            "action": action,
                            "description": description,
                            "result": result,
                            "success": is_success
                        })

                        if progress_callback:
                            status = "completed" if is_success else "error"
                            progress_callback(step_num, status, result)

                        print(f"[执行器] 步骤 {step_num} {'成功' if is_success else '失败'}: {result}")

                    except Exception as e:
                        error_msg = f"执行失败: {str(e)}"
                        results.append({
                            "step": step_num,
                            "action": action,
                            "description": description,
                            "result": error_msg,
                            "success": False
                        })

                        if progress_callback:
                            progress_callback(step_num, "error", error_msg)

                        print(f"[执行器] 步骤 {step_num} 异常: {e}")
                else:
                    error_msg = f"未知操作类型: {action}"
                    results.append({
                        "step": step_num,
                        "action": action,
                        "description": description,
                        "result": error_msg,
                        "success": False
                    })

                    if progress_callback:
                        progress_callback(step_num, "error", error_msg)

                    print(f"[执行器] 步骤 {step_num} 错误: {error_msg}")

            # 如果有旋涂步骤，发送启动指令
            has_spin_coating = any(r["action"] == "spin_coating" for r in results)
            if has_spin_coating:
                print("[执行器] 检测到旋涂步骤，发送启动指令...")
                start_agent_result = self._hardware_agent.execute_tool_call({
                    "name": "start_experiment",
                    "params": {}
                })
                start_result = start_agent_result.get("result", "")
                results.append({
                    "step": "final",
                    "action": "start_experiment",
                    "description": "启动实验序列",
                    "result": start_result,
                    "success": "成功" in start_result or "已发送" in start_result
                })

                if progress_callback:
                    progress_callback(0, "info", start_result)

                print(f"[执行器] 启动指令: {start_result}")

            # 判断整体是否成功
            all_success = all(r["success"] for r in results)

            return {
                "success": all_success,
                "results": results,
                "error": None
            }

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[执行器] 执行异常:")
            print(error_detail)

            return {
                "success": False,
                "results": results,
                "error": str(e)
            }

    def _check_success(self, result: str) -> bool:
        """
        检查执行结果是否成功

        Args:
            result: 执行结果字符串

        Returns:
            bool: 是否成功
        """
        # 失败关键词
        fail_keywords = ["失败", "错误", "异常", "Error", "Failed", "missing"]
        # 成功关键词
        success_keywords = ["成功", "已发送", "已设置", "已移动", "已启动", "✅"]

        result_lower = result.lower()

        # 先检查失败关键词
        if any(keyword in result for keyword in fail_keywords):
            return False

        # 再检查成功关键词
        if any(keyword in result for keyword in success_keywords):
            return True

        # 默认认为成功（如果没有明确的失败标志）
        return True

    def _execute_wait(self, params: dict) -> str:
        """执行等待操作"""
        duration_ms = params.get("duration", 1000)
        duration_s = duration_ms / 1000.0
        time.sleep(duration_s)
        return f"✅ 等待 {duration_s} 秒完成"

    def _execute_loop(self, params: dict) -> str:
        """LOOP 步骤：当前仅记录循环次数，嵌套步骤由上层调用方处理"""
        iterations = params.get("iterations", 1)
        return f"🔁 循环标记 {iterations} 次（嵌套步骤需在上层展开）"

    def _execute_group(self, params: dict) -> str:
        """GROUP 步骤：步骤组标记，实际步骤由上层调用方处理"""
        name = params.get("name", "步骤组")
        return f"📦 进入步骤组: {name}"

    def _execute_condition(self, params: dict) -> str:
        """CONDITION 步骤：条件判断标记，分支执行由上层调用方处理"""
        condition = params.get("condition", "")
        return f"🔀 条件判断: {condition}（分支步骤需在上层展开）"

    def _execute_end(self, params: dict) -> str:
        """END 步骤：结束点标记，标志最近的 LOOP 或 GROUP 结束"""
        return f"🏁 结束点标记（标志循环/组结束）"

    def _execute_user_input(self, params: dict) -> str:
        """
        USER_INPUT 步骤：用户输入标记

        注意：当前为简化实现，仅返回标记信息。
        完整实现需要：
        1. 通过 progress_callback 发送 "user_input_required" 状态
        2. 前端弹出输入框并通过 API 提交用户输入
        3. 后端暂停执行等待用户响应
        4. 接收到输入后恢复执行并将输入值存储到变量
        """
        prompt = params.get("prompt", "请输入参数")
        variable_name = params.get("variable_name", "user_value")
        return f"✋ 用户输入标记: {prompt} (变量: {variable_name})"

    def _get_software_manager(self):
        """延迟加载SoftwareManager"""
        if self._software_manager is None:
            from software.software_manager import SoftwareManager
            self._software_manager = SoftwareManager()
        return self._software_manager

    def _execute_software_algorithm(self, step: dict) -> dict:
        """
        执行软件算法步骤

        step 字段：
            name        : 算法名（必填，对应 REGISTRY.json 中的 name）
            params      : 算法参数（可选，传给 algorithm.run()）
            input_file  : 输入 CSV/数据文件路径（可选）
            output_file : 结果保存路径（可选，不填则不保存）
            user_params : 用户在前端填写的额外参数（可选，与 params 合并）
        """
        algo_name = step.get("name", "")
        params = dict(step.get("params") or {})
        user_params = step.get("user_params") or {}
        params.update(user_params)

        input_file = step.get("input_file")
        output_file = step.get("output_file")

        mgr = self._get_software_manager()

        # 读取输入数据
        if input_file:
            if not os.path.exists(input_file):
                return {"success": False, "message": f"输入文件不存在: {input_file}", "result": None}
            try:
                data = mgr._read_csv_as_columns(input_file)
            except Exception as e:
                return {"success": False, "message": f"读取输入文件失败: {e}", "result": None}
        else:
            data = {}

        result = mgr.run_algorithm(algo_name, data, params)

        # 保存到指定输出路径
        if output_file and result.get("success"):
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["output_file"] = output_file

        return result

    # ========== 验证方法 ==========

    def validate_plan(self, plan_json: dict) -> tuple[bool, str]:
        """
        验证实验方案的有效性

        Args:
            plan_json: 实验方案JSON

        Returns:
            tuple: (是否有效, 错误信息)
        """
        # 检查必需字段
        if "steps" not in plan_json:
            return False, "缺少 'steps' 字段"

        steps = plan_json.get("steps", [])
        if not steps:
            return False, "步骤列表为空"

        # 检查每个步骤
        for i, step in enumerate(steps):
            # 支持两种格式：旧格式用action，新格式用type+name
            step_type = step.get("type", "tool")
            action = step.get("action") or step.get("name")

            if not action:
                return False, f"步骤 {i+1} 缺少操作类型（action或name字段）"

            if "params" not in step:
                return False, f"步骤 {i+1} 缺少 'params' 字段"

            # 检查操作类型是否支持
            if step_type == "helper":
                if action not in self.helper_map:
                    return False, f"步骤 {i+1} 的辅助操作 '{action}' 不支持"
            elif step_type == "software":
                pass  # 算法名在运行时由 SoftwareManager 校验
            elif not self._hardware_agent.is_known_tool(action):
                return False, f"步骤 {i+1} 的操作类型 '{action}' 不支持"

            # 检查旋涂步骤的试剂是否存在
            if action == "spin_coating":
                reagent = step["params"].get("reagent")
                if not reagent:
                    return False, f"步骤 {i+1} 缺少试剂名称"

                # 检查试剂是否存在
                reagent_pos = self._hardware_agent.check_reagent(reagent)
                if reagent_pos[:2] != "BP":
                    return False, f"步骤 {i+1} 的试剂 '{reagent}' 不存在或未配置"

        return True, ""
