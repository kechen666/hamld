# src/eamld/__init__.py

# 导入必要的模块和功能
from .logging_config import setup_logger

from .contraction_strategy import (
    DetectorErrorModelHypergraph,
    ConnectivityGraph,
    GreedyMLDOrderFinder,
    ParallelGreedyMLDOrderFinder,
    ContractionTree,
    SliceFinder,
    ContractionStrategy
)

from .contraction_executor import ContractionExecutor

from .eamld import EAMLD

# from .benchmark import contraction_strategy_benchmarking

from .sample_decoder import Sample_Decoder

from .circuit_generator import CircuitGenerator

# 设置包的版本
__version__ = "0.0.1"

# 定义模块的公共 API，这样外部用户可以直接通过 `import eamld` 使用这些功能
__all__ = [
    'DetectorErrorModelHypergraph',
    'ConnectivityGraph',
    'GreedyMLDOrderFinder',
    'ParallelGreedyMLDOrderFinder',
    'ContractionTree',
    'SliceFinder',
    'contraction_strategy_benchmarking',
    'Sample_Decoder',
    'EAMLD',
    'ContractionStrategy'# 如果需要在外部直接使用此功能
    'CircuitGenerator',
]
