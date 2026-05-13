import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from software.algorithms.base import BaseAlgorithm
import numpy as np

class NormalizeNormalDistribution(BaseAlgorithm):
    name = "normalize_normal_distribution"
    description = "将输入数据转换为标准正态分布（均值为0，标准差为1），适用于数据标准化处理，常用于机器学习模型输入前的预处理。"
    params_schema = {
        "method": {"type": "str", "description": "标准化方法，可选 'z_score'（使用均值和标准差标准化）或 'min_max'（使用最小最大值标准化，但会改变分布形态）", "default": "z_score", "required": False}
    }
    input_format = "list[float] or np.ndarray"
    output_fields = ["normalized_data", "mean", "std"]

    def run(self, data, params=None):
        # 参数提取
        method = params.get("method") if params else self.params_schema["method"]["default"]

        # 输入验证
        if not data:
            return self._build_error("输入数据为空")
        
        if not isinstance(data, (list, np.ndarray)):
            return self._build_error(f"输入数据格式不正确，期望 {self.input_format}，实际为 {type(data)}")
        
        data = np.array(data)
        if data.size == 0:
            return self._build_error("输入数据为空")
        
        # 核心逻辑
        normalized_data = None
        mean = 0.0
        std = 1.0
        
        if method == "z_score":
            # Z-score标准化
            mean = np.mean(data)
            std = np.std(data)
            
            # 处理标准差为0的情况（所有值相同）
            if std == 0:
                normalized_data = np.zeros_like(data)
            else:
                normalized_data = (data - mean) / std
                
        elif method == "min_max":
            # Min-Max标准化
            min_val = np.min(data)
            max_val = np.max(data)
            
            # 处理最大最小值相同的情况（所有值相同）
            if max_val == min_val:
                normalized_data = np.zeros_like(data)
            else:
                normalized_data = (data - min_val) / (max_val - min_val)
        else:
            return self._build_error(f"不支持的标准化方法: {method}")
        
        # 构造输出
        result = {
            "normalized_data": normalized_data.tolist() if isinstance(normalized_data, np.ndarray) else normalized_data,
            "mean": float(mean),
            "std": float(std)
        }
        
        return self._build_success(result, "标准化成功")

if __name__ == "__main__":
    import json
    algo = NormalizeNormalDistribution()
    print(f"算法信息: {algo.get_info()}\n")
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_params = {"method": "z_score"}
    result = algo.run(data=test_data, params=test_params)
    print("测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))