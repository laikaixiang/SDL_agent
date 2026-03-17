import subprocess
import json
import logging

# ai与自动化平台需要通过emqx服务器通信，仅能传输简单的开始/结束指令和一些指定的参数，还没有做平台组件的运动控制协议
import threading
import paho.mqtt.client as mqtt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自动化平台emqx客户端
local_client = MQTTConnector()
topic = "do_experiment"
# 自动化平台试剂瓶位置:溶液的配置文件地址，在实验室电脑里
json_path = "bin\\Debug\\net8.0-windows\\reagent_layout.json"

# ai客户端配置网络配置
class Client_Conf:
    """
    Class saving the configurations of the agent client, including client id, username, password, server ip and port
    """
    def __init__(self):
        self.client_id = "bibilabu"
        self.usr_name = "agent"
        self.password = "s208ht"
        self.ip = "192.168.120.129"
        self.port = 1883

# emqx服务器连接、发布消息
class MQTTConnector:
    """emqx server connection class"""
    def __init__(self):
        self.client_config = Client_Conf()
        self.client = None
        self.is_connected = False
        self.connect_event = threading.Event()

    def on_connect(self, client, userdata, flags, rc):
        """Connection recall"""
        if rc == 0:
            print('Connected to emqx server')
            self.is_connected = True
        else:
            print(f'Connection failed! RC: {rc}')
            self.is_connected = False

        self.connect_event.set()

    def connect(self, timeout=5) -> bool:
        """
        Connect the emqx server. Automatically initiallizes mqtt.Client() class object
        :param timeout: how long the thread waits for connection recall. if timeout, connection is considered fail
        :return: True if connection success, False if failed
        """
        self.client = mqtt.Client()
        self.client.username_pw_set(username=self.client_config.usr_name, password=self.client_config.password)
        self.client.on_connect = self.on_connect

        # reset connection status
        self.is_connected = False
        self.connect_event.clear()

        try:
            self.client.connect(self.client_config.ip, self.client_config.port, 60)
            self.client.loop_start()

            # waiting for recall. if connection established, returns True
            if self.connect_event.wait(timeout):
                return self.is_connected
            else:
                print("Timeout waiting for connection.")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def check_connect(self) -> bool:
        """
        Check connection with emqx server.
        :return: The current connection state. True if agent_client is already or successfully connected, otherwise, False
        """
        if self.is_connected:
            return True
        else:
            return False

    def publish(self, topic:str, msg:str):
        """Publish string data"""
        self.client.publish(topic, msg)

# 获取自动化平台试剂瓶的位置，找到特定溶液对应的平台中的位置
def get_reagents(name:str, path = json_path) -> str:
    """
    Search through reagent_layout.json to find if the reagent we need is already loaded onto the experiment platform.
    :param name: the reagent we want to use
    :param path: the path of reagent_layout.json
    :return: The position in the form of "BPxx" of the reagent on the platform if reagent found, otherwise, raise an error
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            points = data.get("Points", {})
            for point_id, info in points.items():
                reagent_name = info.get("name", "")

                if reagent_name == name:
                    return point_id
            raise Exception("Reagent is missing")
    except Exception as e:
        print(f"Error occurred: {e}")
        return e

# 指示自动化平台进行一次原位旋涂实验，可以传给平台的参数有转速，加速度，旋涂时间和溶液体积
def do_experiment(spin_speed:int, spin_acc:int, spin_dur:int, reagent:str, volume:int) -> str:
    """
    Tell the platform to conduct a single round of an in-situ spin coating experiment. This function will send all the
    parameters you have set one by one to the emqx server, and then it will pass them on to the platform to start the
    experiment.
    :param spin_speed: spin speed for spin coating, max 6000rpm
    :param spin_acc: acceleration of the spin coater, must be integer and default 1000rpm/s
    :param spin_dur: spin duration for spin coating in ms
    :param reagent: Name of the reagent to be used this round.
    :param volume: The volume of the reagent to be dispensed onto substrate
    :return: Whether there is any errors. No errors will return "experiment start"
    """
    try:
        reagent_pos = get_reagents(reagent)
        if reagent_pos[:2] != "BP":
            raise Exception("Reagent is missing")
        print(reagent_pos)

        if local_client.is_connected:
            local_client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
            return "experiment start"
        else:
            connect_state = local_client.connect()
            if connect_state:
                local_client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
                return "experiment start"
            else:
                raise Exception("Connect server failed")
    except Exception as e:
        print(f"Error occurred: {e}")
        return e

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
            if msg == "experiment start":
                return {"status": "success", "output": msg}
            else:
                return {"status": "error", "message": msg"}

        elif action == "do_experiment":
            # 自动化实验平台执行一次原位旋涂实验，要求JSON输出格式为：
            # {"action": "do_experiment", "params": {"spin_speed":int, "spin_acc":int, "spin_dur":int, "reagent":str, "volume":int}
            spin_speed = params.get("spin_speed")
            spin_acc = params.get("spin_acc")
            spin_dur = params.get("spin_dur")
            reagent = params.get("reagent")
            volume = params.get("volume")
            msg = do_experiment(spin_speed, spin_acc, spin_dur, reagent, volume)
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
