"""
光谱数据 3D 可视化 - 将光谱仪 DataFrame 绘制为曲面图，支持保存文件或返回 base64
来源：AutonomousPlatform/test.py
"""

import os
import base64
import io
import numpy as np
import pandas as pd

# 使用非交互式后端 Agg，避免在无显示器的服务器上报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 导入即注册 3D 投影


def save_fig(
    df: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "plot.png",
) -> str:
    """
    根据光谱仪采集的 DataFrame 绘制 3D 曲面图并保存为 PNG 文件

    处理流程：
    1. 从 DataFrame 中提取时间轴、波长轴、计数矩阵
    2. 用 np.meshgrid 构建二维网格坐标
    3. 绘制 3D 曲面图（使用 viridis 配色方案）
    4. 保存到指定路径

    Args:
        df         : 包含 'counts', 'wavelength', 'time' 三列的 DataFrame
                     - df['time']          : 一维数组，形状 (N,)，每个元素是一个时间戳
                     - df['wavelength'][0] : 一维数组，形状 (M,)，表示所有波长刻度
                     - df['counts']        : N 行数据，每行是一个长度为 M 的数值列表
        output_dir : 图表输出目录路径，默认 "figures"，不存在时自动创建
        filename   : 输出文件名，默认 "plot.png"

    Returns:
        str: 保存的文件完整路径（如 "figures/plot.png"）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # ---- 数据提取与重塑 ----

    # unique_times: 时间轴数据，形状 (N,)，N 为采集的时间点数量
    unique_times = np.array(df["time"])

    # unique_wavelengths: 波长轴数据，形状 (M,)，M 为每次采集的波长点数量
    # 取第 0 行的 wavelength，因为所有行的波长刻度相同
    unique_wavelengths = np.array(df.loc[0, "wavelength"], dtype=np.float64)

    # total_counts: 光谱计数矩阵，先提取为二维数组，形状 (N, M)
    total_counts = np.array(
        [df.loc[i, "counts"] for i in range(df["counts"].shape[0])],
        dtype=np.float64,
    )

    # X, Y: 通过 meshgrid 构建二维网格坐标，形状均为 (N, M)
    # X 对应波长维度，Y 对应时间维度
    X, Y = np.meshgrid(unique_wavelengths, unique_times)

    # Z: 光谱计数的二维矩阵，形状 (N, M)，与 X、Y 匹配
    Z = total_counts.reshape(unique_times.shape[0], unique_wavelengths.shape[0])

    # ---- 绘制 3D 曲面图 ----

    fig = plt.figure(figsize=(12, 8))                    # 创建 12x8 英寸画布
    ax = fig.add_subplot(111, projection="3d")           # 添加 3D 坐标轴

    # 绘制曲面，cmap="viridis" 是一种从深蓝到亮黄的渐变配色
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",       # 颜色映射方案（可替换为 "jet", "plasma", "coolwarm" 等）
        alpha=0.8,            # 透明度（0=完全透明, 1=完全不透明）
        linewidth=0.1,        # 网格线宽度
        antialiased=True,     # 开启抗锯齿
    )

    # ---- 图表标签与美化 ----

    ax.set_xlabel("Wavelength", fontsize=12, labelpad=10)   # X 轴标签：波长
    ax.set_ylabel("Time", fontsize=12, labelpad=10)         # Y 轴标签：时间
    ax.set_zlabel("Counts", fontsize=12, labelpad=10)       # Z 轴标签：光谱计数

    ax.set_title(
        "3D Surface Plot of Counts vs Wavelength & Time",   # 图表标题
        fontsize=14,
        pad=20,
    )

    # 添加颜色条（colorbar），显示 Z 轴数值与颜色的对应关系
    fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, label="Counts")

    # 设置初始观察视角：仰角 30 度，方位角 45 度
    ax.view_init(elev=30, azim=45)

    # ---- 保存与清理 ----

    plt.tight_layout()          # 自动调整子图间距
    plt.savefig(output_path)    # 保存为 PNG 文件
    plt.close(fig)              # 关闭图形，释放内存

    return output_path


def fig_to_base64(df: pd.DataFrame) -> str:
    """
    将光谱仪数据绘制为 3D 曲面图，并返回 base64 编码的 PNG 图片字符串

    与 save_fig() 的区别：
    - save_fig() 将图片保存到磁盘文件
    - fig_to_base64() 将图片存入内存缓冲区，返回 base64 字符串
      适合通过 WebSocket 或 HTTP API 直接发送给前端浏览器展示

    前端使用方式：
        收到 base64 字符串后，前端可以直接用 <img src="data:image/png;base64,{字符串}"> 展示

    Args:
        df : 包含 'counts', 'wavelength', 'time' 三列的 DataFrame（格式同 save_fig）

    Returns:
        str: PNG 图片的 base64 编码字符串（不含 data URI 前缀）
    """
    # ---- 数据提取（与 save_fig 相同） ----

    unique_times = np.array(df["time"])                     # 时间轴，形状 (N,)
    unique_wavelengths = np.array(
        df.loc[0, "wavelength"], dtype=np.float64           # 波长轴，形状 (M,)
    )
    total_counts = np.array(
        [df.loc[i, "counts"] for i in range(df["counts"].shape[0])],
        dtype=np.float64,                                   # 计数矩阵，形状 (N, M)
    )

    X, Y = np.meshgrid(unique_wavelengths, unique_times)    # 网格坐标
    Z = total_counts.reshape(
        unique_times.shape[0], unique_wavelengths.shape[0]  # 重塑为 (N, M)
    )

    # ---- 绘图（与 save_fig 相同） ----

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        alpha=0.8,
        linewidth=0.1,
        antialiased=True,
    )

    ax.set_xlabel("Wavelength", fontsize=12, labelpad=10)
    ax.set_ylabel("Time", fontsize=12, labelpad=10)
    ax.set_zlabel("Counts", fontsize=12, labelpad=10)
    ax.set_title(
        "3D Surface Plot of Counts vs Wavelength & Time",
        fontsize=14,
        pad=20,
    )
    fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, label="Counts")
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()

    # ---- 写入内存缓冲区并编码为 base64 ----

    buf = io.BytesIO()                # 创建内存字节流（代替磁盘文件）
    plt.savefig(buf, format="png")    # 将图片写入内存
    plt.close(fig)                    # 关闭图形，释放内存
    buf.seek(0)                       # 将读取指针移回开头

    # 将二进制内容编码为 base64 字符串并返回
    return base64.b64encode(buf.read()).decode("utf-8")
