"""
内置算法包 (software/algorithms/default/)
=========================================

包含开箱即用的数据处理算法：
    data_statistics    - 描述性统计（均值、标准差、分位数、相关性矩阵）
    data_normalization - 数据归一化（Min-Max、Z-Score、Robust）
    spectrum_analysis  - 光谱分析（最高峰、FWHM、峰面积）

SoftwareController 会自动扫描并注册此目录下所有继承 BaseAlgorithm 的类。
"""
