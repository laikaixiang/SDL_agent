"""
实验格式转换器 - JSON与可视化格式的双向转换

职责：
- JSON → Visual（前端图形化格式）
- Visual → JSON（标准实验格式）
- 拓扑排序（根据edges确定执行顺序）
"""
from typing import List


class ExperimentFormatConverter:
    """
    实验格式转换器

    职责：
    - 将标准JSON格式转换为前端可视化格式
    - 将前端可视化格式转换为标准JSON格式
    - 拓扑排序确定节点执行顺序
    """

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
