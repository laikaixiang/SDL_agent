"""
算法基类 (software/algorithms/base.py)
====================================

定义所有算法的统一调用接口。

所有算法必须继承 BaseAlgorithm 并实现 run() 方法，
保证 SoftwareController 可以动态加载并统一调用。

接口规范：
    - run(data, params) → dict  （统一调用接口）
    - get_info() → dict          （元数据查询）

返回格式规范（run 方法必须返回）：
    {
        "success"  : bool,   # 是否成功
        "algorithm": str,    # 算法名称（self.name）
        "result"   : Any,    # 算法输出，内容由各算法自定义
        "message"  : str     # 说明或错误信息
    }
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAlgorithm(ABC):
    """
    算法基类 - 所有算法必须继承此类并实现 run() 方法

    类属性（子类必须覆写）：
        name        : 算法唯一标识符，用于路由（英文、小写、下划线）
        description : 算法功能描述（中文可读）
        params_schema: 参数定义字典，格式见下方说明

    params_schema 格式示例：
        {
            "window_size": {
                "type"       : "int",
                "description": "滑动窗口大小",
                "default"    : 5,
                "required"   : False
            },
            "method": {
                "type"       : "str",
                "description": "归一化方法：'minmax' 或 'zscore'",
                "default"    : "minmax",
                "required"   : False
            }
        }
    """

    name: str = ""
    chinese_name: str = ""
    description: str = ""
    params_schema: dict = {}

    @abstractmethod
    def run(self, data: Any, params: dict = None) -> dict:
        """
        统一算法调用接口（子类必须实现）

        Args:
            data  : 输入数据，可以是 dict / list / numpy array / DataFrame
                    具体类型由各算法说明
            params: 算法参数字典，键名和类型参见 params_schema；
                    未传时使用 params_schema 中的 default 值

        Returns:
            dict: 固定格式
                {
                    "success"  : bool,
                    "algorithm": str,
                    "result"   : Any,
                    "message"  : str
                }

        示例（子类实现）::

            def run(self, data, params=None):
                params = params or {}
                try:
                    # ... 算法逻辑 ...
                    return {
                        "success"  : True,
                        "algorithm": self.name,
                        "result"   : {...},
                        "message"  : "分析完成"
                    }
                except Exception as e:
                    return {
                        "success"  : False,
                        "algorithm": self.name,
                        "result"   : None,
                        "message"  : f"算法执行失败: {str(e)}"
                    }
        """
        pass

    def get_info(self) -> dict:
        """
        返回算法元数据，供前端展示可用算法列表

        Returns:
            dict: {"name", "description", "params_schema"}
        """
        return {
            "name": self.name,
            "chinese_name": self.chinese_name or self.name,
            "description": self.description,
            "params_schema": self.params_schema,
        }

    def _build_error(self, message: str) -> dict:
        """构造统一错误返回（子类可调用的辅助方法）"""
        return {
            "success": False,
            "algorithm": self.name,
            "result": None,
            "message": message,
        }

    def _build_success(self, result: Any, message: str = "执行成功") -> dict:
        """构造统一成功返回（子类可调用的辅助方法）"""
        return {
            "success": True,
            "algorithm": self.name,
            "result": result,
            "message": message,
        }
