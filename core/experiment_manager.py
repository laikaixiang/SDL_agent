"""
实验管理器 - 实验方案的执行、验证和格式转换

职责：
- 执行JSON格式的实验方案
- JSON与图形化格式的双向转换
- 实验方案验证
- 实时进度反馈
"""
import json
import os
import time
from typing import Dict, Callable, Optional, List

from hardware.tools import (
    execute_spin_coating,
    execute_set_temperature,
    execute_move_robot_arm,
    execute_start_experiment,
    execute_collect_spectrum,
    find_reagent
)
from core.software_manager import SoftwareManager


class ExperimentManager:
    """
    实验管理器

    职责：
    - 解析并执行JSON格式的实验方案
    - JSON ↔ 图形化格式双向转换
    - 实验方案验证
    - 提供实时进度反馈
    """

    def __init__(self, software_manager: Optional["SoftwareManager"] = None):
        # 操作类型映射到执行函数
        self.action_map = {
            "spin_coating": self._execute_spin_coating,
            "set_temperature": self._execute_set_temperature,
            "move_robot_arm": self._execute_move_robot_arm,
            "collect_spectrum": self._execute_collect_spectrum,
        }

        # 辅助操作映射
        self.helper_map = {
            "WAIT":      self._execute_wait,
            "LOOP":      self._execute_loop,
            "GROUP":     self._execute_group,
            "CONDITION": self._execute_condition,
        }

        self._software_manager = software_manager

    # ========== 格式转换方法 ==========

    def json_to_visual(self, experiment_json: dict) -> dict:
        """
        将标准JSON格式转换为前端可视化格式

        Args:
            experiment_json: 标准实验JSON
                {
                    "experiment_name": "实验名称",
                    "description": "描述",
                    "steps": [
                        {"type": "tool", "name": "spin_coating", "params": {...}, "description": "..."},
                        {"type": "helper", "name": "WAIT", "params": {"duration": 5000}, "description": "..."}
                    ],
                    "notes": "注意事项"
                }

        Returns:
            dict: 前端可视化格式
                {
                    "experiment_name": "实验名称",
                    "created_at": "2026-04-17T...",
                    "description": "描述",
                    "nodes": [
                        {
                            "id": "node_1",
                            "type": "spin_coating",
                            "label": "旋涂",
                            "params": {...},
                            "description": "..."
                        },
                        {
                            "id": "node_2",
                            "type": "wait",
                            "label": "等待5秒",
                            "params": {"duration": 5000},
                            "description": "..."
                        }
                    ],
                    "edges": [
                        {"from": "node_1", "to": "node_2"}
                    ],
                    "notes": "注意事项"
                }
        """
        nodes = []
        edges = []
        steps = experiment_json.get("steps", [])

        # 转换步骤为节点
        for idx, step in enumerate(steps):
            node_id = f"node_{idx + 1}"
            step_type = step.get("type", "tool")
            step_name = step.get("name", "")

            # 生成节点标签
            if step_type == "helper" and step_name == "WAIT":
                duration_s = step.get("params", {}).get("duration", 1000) / 1000.0
                label = f"等待{duration_s}秒"
            elif step_type == "software":
                label = f"算法:{step_name}"
            else:
                label = self._get_action_label(step_name)

            nodes.append({
                "id": node_id,
                "type": step_name.lower(),
                "label": label,
                "params": step.get("params", {}),
                "description": step.get("description", "")
            })

            # 创建边（连接到下一个节点）
            if idx > 0:
                edges.append({
                    "from": f"node_{idx}",
                    "to": node_id
                })

        return {
            "experiment_name": experiment_json.get("experiment_name", "未命名实验"),
            "created_at": experiment_json.get("created_at", ""),
            "description": experiment_json.get("description", ""),
            "nodes": nodes,
            "edges": edges,
            "notes": experiment_json.get("notes", "")
        }

    def visual_to_json(self, visual_data: dict) -> dict:
        """
        将前端可视化格式转换为标准JSON格式

        Args:
            visual_data: 前端可视化格式
                {
                    "experiment_name": "实验名称",
                    "nodes": [...],
                    "edges": [...],
                    "description": "描述",
                    "notes": "注意事项"
                }

        Returns:
            dict: 标准实验JSON
        """
        nodes = visual_data.get("nodes", [])
        edges = visual_data.get("edges", [])

        # 构建节点顺序（根据edges）
        node_order = self._build_node_order(nodes, edges)

        # 转换节点为步骤
        steps = []
        for node_id in node_order:
            node = next((n for n in nodes if n["id"] == node_id), None)
            if not node:
                continue

            node_type = node.get("type", "")

            # 判断步骤类型
            if node_type == "wait":
                step_type = "helper"
                step_name = "WAIT"
            elif node_type in ("loop", "group", "condition"):
                step_type = "helper"
                step_name = node_type.upper()
            elif node_type.startswith("software:") or node.get("step_type") == "software":
                # 支持 type="software:algo_name" 或 step_type 字段标记
                step_type = "software"
                step_name = node_type.replace("software:", "") or node.get("algo_name", node_type)
            else:
                step_type = "tool"
                step_name = node_type

            step = {
                "type":        step_type,
                "name":        step_name,
                "params":      node.get("params", {}),
                "description": node.get("description", "")
            }
            # software 步骤透传 input_file / output_file
            if step_type == "software":
                if node.get("input_file"):
                    step["input_file"] = node["input_file"]
                if node.get("output_file"):
                    step["output_file"] = node["output_file"]
            steps.append(step)

        return {
            "experiment_name": visual_data.get("experiment_name", "未命名实验"),
            "description": visual_data.get("description", ""),
            "steps": steps,
            "notes": visual_data.get("notes", "")
        }

    def _build_node_order(self, nodes: List[dict], edges: List[dict]) -> List[str]:
        """
        根据边构建节点执行顺序

        Args:
            nodes: 节点列表
            edges: 边列表

        Returns:
            List[str]: 节点ID的执行顺序
        """
        # 构建邻接表
        graph = {node["id"]: [] for node in nodes}
        in_degree = {node["id"]: 0 for node in nodes}

        for edge in edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            if from_node and to_node:
                graph[from_node].append(to_node)
                in_degree[to_node] += 1

        # 拓扑排序
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果有节点未被访问（存在环），按原始顺序返回
        if len(result) != len(nodes):
            return [node["id"] for node in nodes]

        return result

    def _get_action_label(self, action_name: str) -> str:
        """获取操作的中文标签"""
        labels = {
            "spin_coating":    "旋涂",
            "set_temperature": "温度控制",
            "move_robot_arm":  "机械臂移动",
            "collect_spectrum":"光谱采集",
            "WAIT":            "等待",
            "LOOP":            "循环",
            "GROUP":           "步骤组",
            "CONDITION":       "条件判断",
        }
        return labels.get(action_name, action_name)

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

                # 执行工具操作
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

    def _get_software_manager(self) -> SoftwareManager:
        if self._software_manager is None:
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
        import csv as _csv

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
            elif action not in self.action_map:
                return False, f"步骤 {i+1} 的操作类型 '{action}' 不支持"

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
