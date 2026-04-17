import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from software.algorithms.base import BaseAlgorithm
import numpy as np

class BayesianOptimization(BaseAlgorithm):
    name = "bayesian_optimization"
    chinese_name = "贝叶斯优化"
    description = "贝叶斯优化算法，根据历史工艺参数和对应输出，预测下一轮最优的工艺参数组合。"
    params_schema = {
        "bounds": {"type": "list", "description": "每个输入参数的上下限，格式为 [[min1, max1], [min2, max2], ...]，若未提供则使用输入数据中每列的最小值和最大值。", "default": None, "required": False},
        "n_samples": {"type": "int", "description": "下一轮探索的工艺参数组合数量，即生成多少个候选参数点。", "default": 10, "required": False}
    }

    def run(self, data, params=None):
        """
        执行算法

        Args:
            data: 输入 data 为二维列表或 numpy 数组，每行表示一次实验，最后一列为输出值，其余列为输入的工艺参数。输入数据至少包含两列，最后一列为输出，其余为输入参数。
            params: 算法参数字典

        Returns:
            统一格式 dict，包含 success、result、message 字段
        """
        params = params or {}

        try:
            # 1. 参数提取和验证
            bounds = params.get("bounds")
            n_samples = params.get("n_samples", 10)

            # 2. 输入数据验证
            if not isinstance(data, (list, np.ndarray)):
                raise ValueError("输入数据必须是列表或numpy数组")
            
            data = np.array(data)
            if data.ndim != 2:
                raise ValueError("输入数据必须是二维数组")
            
            if data.shape[1] < 2:
                raise ValueError("输入数据至少需要两列，最后一列为输出，其余为输入参数")
            
            # 分离输入参数和输出值
            X = data[:, :-1]  # 所有列除了最后一列
            y = data[:, -1]   # 最后一列是输出值
            
            # 如果没有提供bounds，则使用数据中的最小值和最大值
            if bounds is None:
                bounds = [[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])]
            else:
                # 验证bounds的格式
                if len(bounds) != X.shape[1]:
                    raise ValueError("bounds的长度必须与输入参数的数量一致")
                for i, (min_val, max_val) in enumerate(bounds):
                    if min_val >= max_val:
                        raise ValueError(f"bounds中第{i}个参数的最小值必须小于最大值")

            # 3. 算法核心逻辑实现
            # 使用高斯过程回归进行贝叶斯优化
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            from scipy.optimize import minimize

            # 创建高斯过程回归模型
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)

            # 拟合模型
            gpr.fit(X, y)

            # 定义获取下一个点的函数（使用期望改进）
            def expected_improvement(x, gpr, X_train, y_train, xi=0.01):
                # 计算预测值和标准差
                mu, sigma = gpr.predict(x.reshape(1, -1), return_std=True)
                # 获取当前最优值
                y_best = np.max(y_train)
                # 计算期望改进
                with np.errstate(divide='ignore'):
                    z = (mu - y_best - xi) / sigma
                    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei  # 最大化期望改进，所以取负值用于最小化

            # 生成候选点
            next_params = []
            next_params_scores = []
            
            # 从每个维度的边界内随机采样n_samples个点
            for _ in range(n_samples):
                # 从每个参数的边界内随机采样
                x_sample = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(X.shape[1])])
                next_params.append(x_sample.tolist())
                
                # 计算该点的期望改进
                ei = expected_improvement(x_sample, gpr, X, y)
                next_params_scores.append(float(ei))

            # 4. 构造输出结果
            result = {
                "next_params": next_params,
                "next_params_scores": next_params_scores
            }

            return self._build_success(result, "算法执行成功")

        except Exception as e:
            return self._build_error(f"算法执行失败: {str(e)}")

if __name__ == "__main__":
    import json
    from scipy.stats import norm

    algo = BayesianOptimization()
    print(f"算法信息: {algo.get_info()}\n")

    # 测试用例：构造符合 input_format 的示例数据
    # 3个输入参数，1个输出参数
    test_data = [
        [1.0, 2.0, 3.0, 10.0],
        [2.0, 3.0, 4.0, 15.0],
        [3.0, 4.0, 5.0, 20.0],
        [4.0, 5.0, 6.0, 25.0],
        [5.0, 6.0, 7.0, 30.0]
    ]
    test_params = {
        "bounds": [[0.5, 5.5], [1.5, 6.5], [2.5, 7.5]],
        "n_samples": 5
    }

    result = algo.run(data=test_data, params=test_params)
    print("测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))