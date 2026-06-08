import threading
import time
import paho.mqtt.client as mqtt


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


class MQTTConnector:
    """emqx server connection class"""
    SUBSCRIBE_TOPICS = {
        "platform_respond": 2,
        "do_experiment": 0,
    }

    def __init__(self):
        self.client_config = Client_Conf()
        self.client = None
        self.is_connected: bool = False
        self.message_received: str = "none"
        self.connect_event = threading.Event()
        self._loop_started = False
        self._msg_lock = threading.Lock()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print('Connected to emqx server')
            self.is_connected = True
            for topic, qos in self.SUBSCRIBE_TOPICS.items():
                result = client.subscribe(topic, qos)
                if result[0] == mqtt.MQTT_ERR_SUCCESS:
                    print(f"Subscribed to '{topic}' with QoS {qos}")
                else:
                    print(f"Subscribe '{topic}' failed, rc={result[0]}")
        else:
            print(f'Connection failed! RC: {rc}')
            self.is_connected = False

        self.connect_event.set()

    def on_disconnect(self, client, userdata, rc):
        print(f"Disconnected from broker (rc={rc})")
        self.is_connected = False

    def on_message(self, client, userdata, message):
        """Message callback"""
        try:
            payload_str = message.payload.decode("utf-8")
        except Exception:
            payload_str = message.payload.decode("utf-8", errors="replace")
        with self._msg_lock:
            self.message_received = payload_str
        # print(f"Received from {message.topic}: {payload_str}")

    def connect(self, timeout=5) -> bool:
        """
        Connect the emqx server. Reuses existing client when possible.
        :param timeout: how long the thread waits for connection recall
        :return: True if connection success, False if failed
        """
        if self.is_connected and self.client is not None:
            return True

        if self.client is None:
            self.client = mqtt.Client(client_id=self.client_config.client_id)
            self.client.username_pw_set(
                username=self.client_config.usr_name,
                password=self.client_config.password
            )
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message

        # reset connection status
        self.is_connected = False
        self.connect_event.clear()

        try:
            self.client.connect(self.client_config.ip, self.client_config.port, 60)
            if not self._loop_started:
                self.client.loop_start()
                self._loop_started = True

            # waiting for recall. if connection established, returns True
            if self.connect_event.wait(timeout):
                return self.is_connected
            else:
                print("Timeout waiting for connection.")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def disconnect(self):
        """Clean up network loop and disconnect"""
        if self.client is not None:
            try:
                if self._loop_started:
                    self.client.loop_stop()
                    self._loop_started = False
                self.client.disconnect()
            except Exception as e:
                print(f"Error during disconnect: {e}")
            self.is_connected = False

    def check_connect(self) -> bool:
        """Check connection with emqx server."""
        return self.is_connected

    def publish(self, topic: str, msg: str):
        """Publish string data"""
        if self.client is None or not self.is_connected:
            print(f"[MQTT] Cannot publish to '{topic}': client not connected")
            return
        info = self.client.publish(topic, msg, qos=2)
        try:
            info.wait_for_publish(timeout=2)
            # print("message sent")
        except Exception:
            pass

    def get_message(self):
        """Return latest received message and reset buffer"""
        with self._msg_lock:
            msg = self.message_received
            self.message_received = "none"
            return msg

    def listen_to_message(self, matchstr: str, timeout: float = None):
        """
        Block until a message matching matchstr is received.
        :param matchstr: the exact payload string to match
        :param timeout: optional timeout in seconds
        :return: the matching message, or None if timeout
        """
        start = time.time()
        while True:
            with self._msg_lock:
                if self.message_received == matchstr:
                    self.message_received = "none"
                    return matchstr
            if timeout is not None and (time.time() - start) > timeout:
                return None
            time.sleep(0.02)
