from typing import List, Tuple, Set, Optional
import logging
from eamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
from eamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph
from eamld.contraction_strategy.contraction_tree import ContractionTree
from eamld.contraction_strategy.mld_order_finder import GreedyMLDOrderFinder
from eamld.contraction_strategy.slice_finder import SliceFinder
from eamld.logging_config import setup_logger
from stim import DetectorErrorModel

# 设置 logging 配置，放在模块级别
logger = setup_logger("./contraction_strategy/contraction_strategy", log_level=logging.DEBUG)

SUPPORT_ORDER_STRATEGY = {'mld', 'greedy'}
SUPPORT_SLICE_STRATEGY = {'parallelism', 'memory'}

class ContractionStrategy:
    def __init__(self, order: Optional[List[str]] = None, sliced_hyperedges: Optional[List[Tuple[str]]] = None):
        """
        收缩策略类，提供整体接口获取收缩策略。
        
        Args:
            order (Optional[List[str]]): 节点的收缩顺序。如果没有提供，则使用默认的序号策略。默认为None。
            sliced_hyperedges (Optional[List[Tuple[str]]]): 需要切片并行的超边列表。如果没有，则不进行切片。默认为None。
        """
        self.order: List[str] = order if order else []
        self.sliced_hyperedges: List[Tuple[str]] = sliced_hyperedges if sliced_hyperedges else []
        self.contraction_cost: int = 0  
        self.contraction_width: int = 0
        self.sliced_contraction_cost: int = 0  
        self.sliced_contraction_width: int = 0

    def __str__(self) -> str:
        """
        返回一个字符串表示 ContractionStrategy 实例的当前状态。
        
        Returns:
            str: 表示 ContractionStrategy 的字符串。
        """
        return (
            f"ContractionStrategy(\n"
            f"  order: {self.order},\n"
            f"  sliced_hyperedges: {self.sliced_hyperedges},\n"
            f"  contraction_cost: {self.contraction_cost},\n"
            f"  contraction_width: {self.contraction_width},\n"
            f"  sliced_contraction_cost: {self.sliced_contraction_cost},\n"
            f"  sliced_contraction_width: {self.sliced_contraction_width}\n"
            f")"
        )

    def get_contraction_strategy_from_slice_finder(self, slice_finder: SliceFinder) -> None:
        """
        从SliceFinder中获取收缩策略, 包括顺序、切片、成本。
        
        Args:
            slice_finder (SliceFinder): 一个SliceFinder对象，用来获取节点顺序和切片策略。
        """
        self.order = slice_finder.contraction_tree.order
        self.contraction_cost = slice_finder.contraction_tree.contraction_cost
        self.contraction_width = slice_finder.contraction_tree.contraction_width
        if slice_finder.sliced_hyperedges == []:
            self.sliced_contraction_cost = self.contraction_cost
            self.sliced_contraction_width = self.contraction_width
            self.sliced_hyperedges = []
        else:
            self.order, self.sliced_hyperedges, self.sliced_contraction_cost, self.sliced_contraction_width = slice_finder.get_contraction_strategy()
    
    def get_contraction_strategy_from_error_model(
        self,
        detector_error_model: Optional[DetectorErrorModel] = None,
        order_strategy: Optional[str] = None,
        perform_slicing: bool = False,
        slice_strategy: Optional[str] = None,
        sliced_hyperedge_count: Optional[int] = None,
        sliced_contraction_width: Optional[int] = None,
        update_order: bool = False
    ) -> None:
        """
        根据输入的错误模型、收缩顺序、切片策略以及其他参数获取收缩策略。
        
        Args:
            detector_error_model (DetectorErrorModel): 错误模型。
            order_strategy (Optional[str]): 收缩顺序。默认为None。
            slice_strategy (Optional[str]): 切片策略。默认为None。
            perform_slicing (bool): 是否执行切片。默认为False。
            update_order (bool): 是否更新顺序。默认为False。
        """
        # 基于错误模型、顺序和切片策略来选择合适的策略
        hypergraph = DetectorErrorModelHypergraph(detector_error_model=detector_error_model, have_logical_observable=True)
        connectivity_graph = ConnectivityGraph()
        connectivity_graph.hypergraph_to_connectivity_graph(hypergraph)
        
        if order_strategy == 'greedy':
            # 使用贪心策略寻找收缩顺序
            order_finder = GreedyMLDOrderFinder(connectivity_graph)
            self.order = order_finder.find_order()
            
            # 获取收缩树的成本和宽度
            contraction_tree = ContractionTree(self.order, detector_error_model=detector_error_model)
            self.contraction_cost, self.contraction_width, _ = contraction_tree.get_contraction_cost_information()
            self.sliced_contraction_cost = self.contraction_cost
            self.sliced_contraction_width = self.contraction_width
        elif order_strategy == 'mld':
            detector_number = detector_error_model.num_detectors
            self.order = [f"D{i}" for i in range(detector_number)]
            contraction_tree = ContractionTree(self.order, detector_error_model=detector_error_model)
            self.contraction_cost, self.contraction_width, _ = contraction_tree.get_contraction_cost_information()
            self.sliced_contraction_cost = None
            self.sliced_contraction_width = None
        
        # 默认传统的MLD方法不进行切片
        if perform_slicing and order_strategy != 'mld':
            # 如果需要切片，创建SliceFinder并进行切片操作
            slice_finder = SliceFinder(contraction_tree)
            if slice_strategy == 'parallelism':
                if sliced_hyperedge_count is None:
                    raise ValueError("For 'parallelism' slice strategy, sliced_hyperedge_count must be provided.")
                slice_finder.slice_based_on_parallelism(sliced_hyperedge_count, update_order)
                logger.info(f"Performing slicing with parallelism strategy, slicing {sliced_hyperedge_count} hyperedges.")
            elif slice_strategy == 'memory':
                if sliced_contraction_width is None:
                    raise ValueError("For 'memory' slice strategy, sliced_contraction_width must be provided.")
                slice_finder.slice_based_on_memory(sliced_contraction_width, update_order)
                logger.info(f"Performing slicing with memory strategy, maximum contraction width: {sliced_contraction_width}.")
            else:
                raise ValueError(f"Unsupported slice strategy '{slice_strategy}' for slicing.")
            
            # 获取切片后的收缩策略
            self.order, self.sliced_hyperedges, self.sliced_contraction_cost, self.sliced_contraction_width = slice_finder.get_contraction_strategy()
        else:
            # 如果不进行切片，清空切片超边
            self.sliced_hyperedges = []
        
    def get_contraction_strategy(self, 
                                slice_finder: Optional[SliceFinder] = None, 
                                detector_error_model: Optional[DetectorErrorModel] = None,
                                order_strategy: Optional[str] = None,
                                perform_slicing: bool = False,
                                slice_strategy: Optional[str] = None,
                                sliced_hyperedge_count: Optional[int] = None,
                                sliced_contraction_width: Optional[int] = None,
                                update_order: bool = False) -> None:
        """
        获取收缩策略，支持从 SliceFinder 获取或从错误模型获取收缩策略。

        根据不同的输入条件（例如：收缩顺序策略、切片策略等），动态生成收缩策略。
        
        Args:
            slice_finder (Optional[SliceFinder]): 提供收缩顺序和切片策略的 SliceFinder 对象。
            detector_error_model (Optional[DetectorErrorModel]): 错误模型，用于指导收缩策略。
            order_strategy (Optional[str]): 收缩顺序策略，当前支持 "greedy"和 "mld"，其中mld指的是按序列顺序。
            perform_slicing (bool): 是否进行切片。如果为 True，按照切片策略执行切片。
            slice_strategy (Optional[str]): 切片策略，支持 "parallelism"（并行度）和 "memory"（内存要求）。
            sliced_hyperedge_count (Optional[int]): 若切片策略为 "parallelism"，指定并行度。
            sliced_contraction_width (Optional[int]): 若切片策略为 "memory"，指定最大内存要求。
            update_order (bool): 是否在每次切片后更新收缩顺序。

        Raises:
            ValueError: 如果提供了不支持的收缩顺序策略或切片策略。
        """
        # 处理SliceFinder或错误模型的输入
        if slice_finder is not None:
            # 从SliceFinder获取收缩策略
            self.get_contraction_strategy_from_slice_finder(slice_finder)
        elif detector_error_model is not None:
            # 默认使用'greedy'收缩顺序策略
            if order_strategy is None:
                order_strategy = 'greedy'
            
            # 验证收缩顺序策略是否合法
            if order_strategy not in SUPPORT_ORDER_STRATEGY:
                raise ValueError(f"Order strategy '{order_strategy}' is not supported. Supported strategies: {SUPPORT_ORDER_STRATEGY}.")
            
            # 验证切片策略是否合法
            if slice_strategy is not None:
                if slice_strategy not in SUPPORT_SLICE_STRATEGY:
                    raise ValueError(f"Slice strategy '{slice_strategy}' is not supported. Supported strategies: {SUPPORT_SLICE_STRATEGY}.")
            
            # 获取收缩策略
            self.get_contraction_strategy_from_error_model(
                detector_error_model,
                order_strategy,
                perform_slicing,
                slice_strategy,
                sliced_hyperedge_count,
                sliced_contraction_width,
                update_order
            )
        else:
            raise ValueError("Either slice_finder or detector_error_model must be provided.")