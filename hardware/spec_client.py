"""
光谱仪数据采集客户端 - 通过 MQTT 订阅光谱仪实时数据，汇总为 DataFrame
来源：AutonomousPlatform/spec_client.py，封装为 SpectrometerClient 类
"""

import threading
import paho.mqtt.client as mqtt
import numpy as np
import pandas as pd

from .visualization import save_fig


class SpectrometerClient:
    """
    光谱仪 MQTT 数据采集客户端

    该类封装了通过 MQTT 协议从光谱仪采集实时数据的完整逻辑。
    内部维护一个状态机来控制数据记录的启停。

    Attributes:
        broker_ip (str)   : EMQX/MQTT 代理服务器的 IP 地址
        port (int)        : MQTT 代理服务器端口号，默认 1883
        client_id (str)   : MQTT 客户端标识符，用于在服务器端区分不同客户端
        username (str)    : MQTT 认证用户名
        password (str)    : MQTT 认证密码
        output_dir (str)  : 生成的图表文件的保存目录

    内部属性:
        _counts (list)         : 缓冲区 - 暂存每次收到的光谱计数数据
        _wavelength (list)     : 缓冲区 - 暂存每次收到的波长数据
        _time (list)           : 缓冲区 - 暂存每次收到的时间戳
        _df (pd.DataFrame)     : 汇总数据表，最终包含所有采集到的数据
        _running_state (int)   : 状态机变量（0=阻塞等待, 1=正在记录, -1=已退出）
        _lock (threading.Lock) : 线程锁，保护对共享数据的并发访问
        _client (mqtt.Client)  : paho-mqtt 客户端实例
        _thread (threading.Thread) : 后台采集线程
    """

    def __init__(
        self,
        broker_ip: str = "192.168.120.129",
        port: int = 1883,
        client_id: str = "987zyx",
        username: str = "s208",
        password: str = "s208ht",
        output_dir: str = "figures",
    ):
        """
        初始化光谱仪客户端

        Args:
            broker_ip  : MQTT 代理服务器 IP 地址，默认 "192.168.120.129"
            port       : MQTT 端口号，默认 1883
            client_id  : 客户端唯一标识符，默认 "987zyx"
            username   : 认证用户名，默认 "s208"
            password   : 认证密码，默认 "s208ht"
            output_dir : 图表输出目录，默认 "figures"
        """
        # MQTT 连接参数
        self.broker_ip = broker_ip       # EMQX 服务器地址
        self.port = port                 # MQTT 端口
        self.client_id = client_id       # 客户端ID（在EMQX中可见）
        self.username = username          # MQTT 认证用户名
        self.password = password          # MQTT 认证密码
        self.output_dir = output_dir     # 图表文件保存路径

        # ---------- 数据缓冲区 ----------
        # 缓冲区用于暂存两次 "record" 命令之间收到的原始数据
        self._counts = []       # 光谱计数缓冲区，每个元素是一组数值（bytes列表）
        self._wavelength = []   # 波长缓冲区，每个元素是一组数值（bytes列表）
        self._time = []         # 时间戳缓冲区，每个元素是一个浮点数

        # ---------- 汇总数据 ----------
        # 每次收到 "record" 命令时，缓冲区数据会被追加到这个 DataFrame 中
        self._df = pd.DataFrame({"counts": [], "wavelength": [], "time": []})

        # ---------- 状态与线程控制 ----------
        # _running_state 是核心状态变量：
        #   0  = 阻塞/等待中（等待 "continue" 命令唤醒）
        #   1  = 正在记录数据
        #   -1 = 已退出（收到 "quit" 命令）
        self._running_state = 0
        self._lock = threading.Lock()     # 线程锁，保护 _counts/_wavelength/_time/_df 的并发访问
        self._client = None               # paho-mqtt 客户端实例（在 _run_loop 中创建）
        self._thread = None               # 后台采集线程引用

    def _on_connect(self, client, userdata, flags, rc):
        """
        MQTT 连接成功回调函数

        连接成功后自动订阅光谱仪相关的所有主题。

        Args:
            client   : mqtt.Client 实例
            userdata : 用户自定义数据（未使用）
            flags    : 连接标志（未使用）
            rc       : 连接结果代码，0 表示成功，非 0 表示失败
        """
        if rc == 0:
            print("SpectrometerClient: 已连接到 MQTT 代理服务器")
            # 订阅光谱仪发布数据的所有主题
            client.subscribe("counts")       # 光谱计数数据
            client.subscribe("wavelength")   # 波长数据
            client.subscribe("control")      # 控制命令（continue/record/stop/quit）
            client.subscribe("time")         # 时间戳数据
        else:
            print(f"SpectrometerClient: 连接失败，返回码 RC={rc}")

    def _on_message(self, client, userdata, msg):
        """
        MQTT 消息接收回调函数

        根据当前状态机状态（_running_state）和收到的消息主题（msg.topic）
        执行相应的数据处理或状态转换。

        Args:
            client   : mqtt.Client 实例
            userdata : 用户自定义数据（未使用）
            msg      : 收到的 MQTT 消息对象，包含 .topic（主题）和 .payload（内容）
        """
        with self._lock:
            # ===== 正在记录状态 (state=1) =====
            if self._running_state == 1:
                if msg.topic == "counts":
                    # 收到光谱计数数据，按空格分割后存入缓冲区
                    data = msg.payload.split()
                    self._counts.append(data)

                elif msg.topic == "wavelength":
                    # 收到波长数据，按空格分割后存入缓冲区
                    data = msg.payload.split()
                    self._wavelength.append(data)

                elif msg.topic == "time":
                    # 收到时间戳数据，转为浮点数后存入缓冲区
                    data = float(msg.payload)
                    self._time.append(data)

                elif msg.topic == "control":
                    if msg.payload == b"record":
                        # "record" 命令：将缓冲区数据写入 DataFrame，然后清空缓冲区
                        new_data = pd.DataFrame({
                            "counts": self._counts,
                            "wavelength": self._wavelength,
                            "time": self._time,
                        })
                        self._df = pd.concat([self._df, new_data], ignore_index=True)
                        self._counts.clear()
                        self._wavelength.clear()
                        self._time.clear()

                    elif msg.payload == b"stop":
                        # "stop" 命令：暂停记录，进入阻塞状态
                        self._running_state = 0

                    elif msg.payload == b"quit":
                        # "quit" 命令：终止采集
                        self._running_state = -1

            # ===== 阻塞/等待状态 (state=0) =====
            elif self._running_state == 0:
                if msg.topic == "control":
                    if msg.payload == b"continue":
                        # "continue" 命令：唤醒客户端，开始新一轮数据记录
                        self._running_state = 1
                        # 清空 DataFrame，为新一轮采集做准备
                        self._df = pd.DataFrame({
                            "counts": [],
                            "wavelength": [],
                            "time": [],
                        })
                        # 向控制主题发送 "next" 表示客户端已准备好接收数据
                        client.publish("control", "next")

                    elif msg.payload == b"quit":
                        # "quit" 命令：终止采集
                        self._running_state = -1

    def _run_loop(self):
        """
        内部主循环，在后台线程中运行

        负责：
        1. 创建并配置 MQTT 客户端
        2. 连接到 MQTT 代理服务器
        3. 持续等待数据和控制命令
        4. 当数据采集结束（stop）时，调用可视化模块生成图表
        5. 当收到退出命令（quit）时，断开连接并退出循环
        """
        # 创建 MQTT 客户端并设置认证信息
        self._client = mqtt.Client()
        self._client.username_pw_set(username=self.username, password=self.password)
        self._client.on_connect = self._on_connect     # 注册连接回调
        self._client.on_message = self._on_message     # 注册消息回调

        # 连接到 MQTT 服务器（keepalive=60秒）
        self._client.connect(self.broker_ip, self.port, 60)
        # 启动 MQTT 客户端网络循环（后台线程处理消息收发）
        self._client.loop_start()

        while True:
            if self._running_state == -1:
                # 退出状态：停止网络循环并断开连接
                self._client.loop_stop()
                print("SpectrometerClient: 已断开连接")
                break

            elif self._running_state == 1:
                # 记录状态：等待直到状态变为阻塞（数据采集完成）或退出
                while True:
                    if self._running_state == 0:
                        # 数据采集完成，生成可视化图表
                        print("SpectrometerClient: 数据接收完成")
                        print(f"DataFrame 形状: {self._df.shape}")
                        if not self._df.empty:
                            save_fig(self._df, output_dir=self.output_dir)
                        break
                    elif self._running_state == -1:
                        break

    def start_collection(self):
        """
        启动光谱仪数据采集

        创建后台线程连接 MQTT 并开始监听光谱仪数据。
        调用后客户端处于阻塞状态（state=0），等待光谱仪控制端发送 "continue" 命令
        才会正式开始记录数据。

        注意：重复调用不会创建多个线程，已有线程运行时会直接返回。

        使用示例::

            spec = SpectrometerClient()
            spec.start_collection()
            # 此时客户端已在后台运行，等待光谱仪控制端的 "continue" 命令
        """
        if self._thread and self._thread.is_alive():
            print("SpectrometerClient: 采集线程已在运行中，无需重复启动")
            return

        self._running_state = 0  # 重置为阻塞状态
        # daemon=True 表示主程序退出时自动终止此线程
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop_collection(self) -> pd.DataFrame:
        """
        停止数据采集并返回已收集的全部数据

        将状态设为退出（state=-1），等待后台线程结束，然后返回 DataFrame 副本。

        Returns:
            pd.DataFrame: 包含 'counts', 'wavelength', 'time' 三列的数据表。
                - counts     : 光谱计数数据（每行是一组数值的列表）
                - wavelength : 波长数据（每行是一组数值的列表）
                - time       : 对应的时间戳（浮点数）

        使用示例::

            final_data = spec.stop_collection()
            print(final_data.head())
        """
        self._running_state = -1  # 通知后台线程退出
        if self._thread:
            self._thread.join(timeout=5)  # 等待线程结束，最多等 5 秒
        return self._df.copy()

    def get_latest_data(self) -> pd.DataFrame:
        """
        获取当前已采集的数据（不停止采集）

        在采集过程中可以随时调用此方法查看已收集到的数据。
        返回的是 DataFrame 的副本，不会影响采集过程。

        Returns:
            pd.DataFrame: 当前已采集数据的副本

        使用示例::

            # 采集进行中，查看目前收集了多少数据
            current_df = spec.get_latest_data()
            print(f"已采集 {len(current_df)} 条记录")
        """
        return self._df.copy()

    def is_collecting(self) -> bool:
        """
        检查是否正在采集数据

        Returns:
            bool: True 表示正在记录数据（state=1），False 表示处于等待或已退出状态
        """
        return self._running_state == 1
