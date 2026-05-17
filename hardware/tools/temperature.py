"""
温度控制工具
"""

from .registry import register_tool


@register_tool(
    name="set_temperature",
    description="设置加热台温度",
    params={
        "target": {"type": "float", "description": "目标温度(℃)", "required": True}
    }
)
def set_temperature(target: float) -> str:
    """
    底层同步函数：设置加热台温度

    当前为模拟实现（返回确认消息），实际部署时需取消注释
    subprocess调用以执行真实的C/C++温控程序。

    Args:
        target : 目标温度值（单位：℃）

    Returns:
        str: 温度设置结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        # cmd_list = ["./temp_ctrl", "--set", str(target)]
        # res = subprocess.run(cmd_list, capture_output=True, text=True)
        # return res.stdout.strip()
        return f"硬件加热台温度已成功设置为 {target} ℃"
    except Exception as e:
        return f"温度设置失败: {str(e)}"
