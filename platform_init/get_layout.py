from typing import *
import json

from .registry import register_tool
from .utils import get_mqtt_client
import vision.platform_scan as scan


@register_tool(
    name="scan_layout",
    description="开始扫描平台，获取试剂瓶托盘和玻璃片托盘分别在滴液枪坐标系和机械臂坐标系中的位置，结果保留在vision/blocks.json中",
    params={},
)
def scan_layout() -> None:
    """
    Starts the dispenser and the camera to detect tray distribution of the platform
    """
    # TODO: Use the dispenser to do the whole platform scan process with two for loops
    try:
      client = get_mqtt_client()
      for block_id in range(1, 7):
          if client.is_connected:
              client.publish(experiment_topic, "scan")
          else:
              connect_state = local_client.connect()
              if connect_state:
                  client.publish(experiment_topic, "scan")
          pos_conf = scan.run_scan(block_id)
          results = pos_conf[block_id]
          substrate_pos = results["substrate_trays"]
          bottle_pos = results["bottle_trays"]
          if len(substrate_pos) > 0:
              for pos in substrate_pos:
                  scan.write_tray_pos(float(pos[0]), float(pos[1]), block_id, "substrate_trays")
          if len(bottle_pos) > 0:
              for pos in bottle_pos:
                  scan.write_tray_pos(float(pos[0]), float(pos[1]), block_id, "bottle_trays")
    except Exeption as e:
      return 
