"""
实验执行器 - 根据JSON方案执行实验
"""
import json
from typing import Dict, Callable, Optional

from hardware.tools import (
    execute_spin_coating,
    execute_set_temperature,
    execute_move_robot_arm,
    execute_start_experiment,
    execute_collect_spectrum,
    find_reagent
)


class ExperimentExecutor:
    """
    实验执行器

    职责：
    - 解析JSON格式的实验方案
    - 按步骤顺序执行实验操作
    - 提供实时进度反馈
    - 记录执行结果
    """

    def __init__(self):
        # 操作类型映射到执行函数
        self.action_map = {
            "spin_coating": self._execute_spin_coating,
            "set_temperature": self._execute_set_temperature,
            "move_robot_arm": self._execute_move_robot_arm,
            "collect_spectrum": self._execute_collect_spectrum,
        }

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
            for step in steps:
                step_num = step.get("step_number", 0)
                action = step.get("action")
                params = step.get("params", {})
                description = step.get("description", "")

                print(f"[执行器] 步骤 {step_num}: {description} ({action})")

                if progress_callback:
                    progress_callback(step_num, "running", f"正在执行: {description}")

                # 执行操作
                if action in self.action_map:
                    try:
                        result = self.action_map[action](params)
                        is_success = self._check_success(result)

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
                start_result = execute_start_experiment()
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

    def _execute_spin_coating(self, params: dict) -> str:
        """执行旋涂操作"""
        return execute_spin_coating(
            spin_speed=params.get("spin_speed", 3000),
            spin_acc=params.get("spin_acc", 1000),
            spin_dur=params.get("spin_dur", 30000),
            reagent=params.get("reagent", ""),
            volume=params.get("volume", 10)
        )

    def _execute_set_temperature(self, params: dict) -> str:
        """执行温度设置"""
        return execute_set_temperature(params.get("temperature", 25))

    def _execute_move_robot_arm(self, params: dict) -> str:
        """执行机械臂移动"""
        return execute_move_robot_arm(
            x=params.get("x", 0),
            y=params.get("y", 0),
            z=params.get("z", 0)
        )

    def _execute_collect_spectrum(self, params: dict) -> str:
        """执行光谱采集"""
        return execute_collect_spectrum(params.get("duration", 60))

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
            if "action" not in step:
                return False, f"步骤 {i+1} 缺少 'action' 字段"

            action = step["action"]
            if action not in self.action_map:
                return False, f"步骤 {i+1} 的操作类型 '{action}' 不支持"

            if "params" not in step:
                return False, f"步骤 {i+1} 缺少 'params' 字段"

            # 检查旋涂步骤的试剂是否存在
            if action == "spin_coating":
                reagent = step["params"].get("reagent")
                if not reagent:
                    return False, f"步骤 {i+1} 缺少试剂名称"

                # 检查试剂是否存在
                reagent_pos = find_reagent(reagent)
                if reagent_pos[:2] != "BP":
                    return False, f"步骤 {i+1} 的试剂 '{reagent}' 不存在或未配置"

        return True, ""
