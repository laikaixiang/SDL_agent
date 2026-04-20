"""
导出硬件工具注册表到 REGISTRY.json

运行此脚本以更新 hardware/tools/REGISTRY.json 文件
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入所有工具以触发注册
from hardware.tools import (
    execute_spin_coating,
    execute_set_temperature,
    execute_move_robot_arm,
    execute_start_experiment,
    execute_collect_spectrum,
    ToolRegistry
)

if __name__ == "__main__":
    # 导出到 REGISTRY.json
    output_path = os.path.join("hardware", "tools", "REGISTRY.json")
    ToolRegistry.export_to_json(output_path)
    print(f"✅ 注册表已导出到: {output_path}")
    print(f"📊 共注册 {len(ToolRegistry.get_all())} 个工具:")
    for name in ToolRegistry.get_all().keys():
        print(f"   - {name}")
