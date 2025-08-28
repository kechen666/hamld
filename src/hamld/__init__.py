# src/hamld/__init__.py

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

from .hamld import HAMLD

# from .benchmark import contraction_strategy_benchmarking

from .sample_decoder import Sample_Decoder

from .circuit_generator import CircuitGenerator

# C++ version: if you want to use it, please uncomment the lines below.
# 如果你希望使用C++的高性能版本，请参考cpp文件，将其编译并放在在当前文件夹下，同时取消下述代码的注释。
# from .hamld_pybind11 import HAMLDCpp_from_file

# 设置包的版本
__version__ = "0.0.0"

# 定义模块的公共 API，这样外部用户可以直接通过 `import hamld` 使用这些功能
__all__ = [
    'DetectorErrorModelHypergraph',
    'ConnectivityGraph',
    'GreedyMLDOrderFinder',
    'ParallelGreedyMLDOrderFinder',
    'ContractionTree',
    'SliceFinder',
    'contraction_strategy_benchmarking',
    'Sample_Decoder',
    'HAMLD',
    'ContractionStrategy'# 如果需要在外部直接使用此功能
    'CircuitGenerator',
]
