import subprocess
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_c_executable(command, args):
    """
    调用底层 C/C++ 编译后的可执行文件 (例如: ./motor_control)
    """
    try:
        # 构建执行命令列表
        cmd_list = [f"./{command}"] + [str(a) for a in args]
        logger.info(f"正在执行底层硬件指令: {' '.join(cmd_list)}")

        # subprocess.run 会阻塞等待执行完成，并获取输出结果
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        logger.error(f"硬件执行失败: {e.stderr}")
        return {"status": "error", "message": e.stderr}
    except FileNotFoundError:
        return {"status": "error", "message": f"找不到 C 语言可执行文件: {command}"}


def execute_llm_hardware_command(llm_json_response):
    """
    接收并解析大模型传来的 JSON 指令，分发给不同的底层模块。
    大模型的 Prompt 应该强制要求输出固定格式的 JSON，例如：
    {"action": "set_temperature", "params": {"target": 25.5}}
    """
    try:
        # 提取 LLM 输出的纯 JSON 部分 (防止带 markdown ```json)
        clean_text = llm_json_response.replace("```json", "").replace("```", "").strip()
        cmd_data = json.loads(clean_text)

        action = cmd_data.get("action")
        params = cmd_data.get("params", {})

        # 🚗 路由分发器：根据 action 选择调用的硬件脚本
        if action == "set_temperature":
            # 假设你有一个控制温度的 C 程序：temp_ctrl
            target = params.get("target", 25.0)
            return call_c_executable("temp_ctrl", ["--set", target])

        elif action == "move_robot_arm":
            # 假设你有一个控制机械臂的 python 脚本：arm_ctrl.py
            x, y, z = params.get("x", 0), params.get("y", 0), params.get("z", 0)
            # 调用 Python 脚本
            res = subprocess.run(["python", "arm_ctrl.py", str(x), str(y), str(z)], capture_output=True, text=True)
            return {"status": "success", "output": res.stdout}

        else:
            return {"status": "error", "message": f"未知的硬件操作指令: {action}"}

    except json.JSONDecodeError:
        return {"status": "error", "message": "大模型返回的不是合法的 JSON 格式"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 本地测试用的 Mock 示例
if __name__ == "__main__":
    mock_llm_output = '{"action": "set_temperature", "params": {"target": 35.0}}'
    print(execute_llm_hardware_command(mock_llm_output))