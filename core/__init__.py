"""
核心业务逻辑模块包
包含所有业务逻辑的实现，每个.py文件负责单一功能模块

注意：PDFProcessor 和 ExtractionEngine 已移至 extract/ 包，
     通过 core/extract_manager.py 门面访问：
       from core.extract_manager import PDFProcessor, ExtractionEngine
"""

# 模块导出
from .config import Config
from .llm_client import LLMClient
from .field_inference import FieldInference, AlgorithmParser, ExperimentDesignAgent
from .hardware_controller import HardwareController
from .task_manager import TaskManager
from .software_manager import SoftwareManager
from .adaptive_stream import AdaptiveStreamHandler

__all__ = [
    'Config',
    'LLMClient',
    'FieldInference',
    'AlgorithmParser',
    'ExperimentDesignAgent',
    'HardwareController',
    'TaskManager',
    'SoftwareManager',
    'AdaptiveStreamHandler',
]
