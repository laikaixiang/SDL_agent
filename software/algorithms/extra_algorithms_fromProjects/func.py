import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from software.algorithms.base import BaseAlgorithm
import numpy as np

class Func(BaseAlgorithm):
    name = "func"
    description = "func 算法用于对输入数据进行处理并生成期望的输出结果，适用于数据转换、特征提取或结果计算等场景。"
    params_schema = {
        "scale_factor": {"type": "float", "description": "缩放因子，用于调整输出结果的幅度", "default": 1.0, "required": False},
        "mode": {"type": "string", "description": "处理模式，可选值为 'mean', 'sum', 'max'", "default": "mean", "required": False}
    }
    input_format = "list"
    output_fields = ["output"]

    def run(self, data, params=None):
        # 参数提取
        scale_factor = params.get("scale_factor", self.params_schema["scale_factor"]["default"])
        mode = params.get("mode", self.params_schema["mode"]["default"])
        
        # 输入验证
        if not isinstance(data, dict) or 'input' not in data:
            return self._build_error("输入数据必须为包含 'input' 字段的字典")
        
        input_data = data['input']
        if not isinstance(input_data, list):
            return self._build_error("输入数据的 'input' 字段必须为列表")
        
        if len(input_data) == 0:
            return self._build_error("输入数据不能为空列表")
        
        # 核心逻辑
        try:
            # 转换为 numpy 数组进行处理
            arr = np.array(input_data)
            
            # 处理模式
            if mode == "mean":
                result_value = np.mean(arr)
            elif mode == "sum":
                result_value = np.sum(arr)
            elif mode == "max":
                result_value = np.max(arr)
            else:
                return self._build_error(f"不支持的模式: {mode}")
            
            # 应用缩放因子
            result_value = result_value * scale_factor
            
        except Exception as e:
            return self._build_error(f"数据处理过程中发生错误: {str(e)}")
        
        # 构造输出
        result = {
            "output": float(result_value)
        }
        
        return self._build_success(result, "处理成功")

if __name__ == "__main__":
    import json
    algo = Func()
    print(f"算法信息: {algo.get_info()}\n")
    test_data = {"input": [1, 2, 3, 4, 5]}
    test_params = {"scale_factor": 2.0, "mode": "sum"}
    result = algo.run(data=test_data, params=test_params)
    print("测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))