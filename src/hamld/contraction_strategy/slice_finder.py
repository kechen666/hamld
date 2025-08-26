from typing import List, Tuple, Set
import logging
from hamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
from hamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph
from hamld.contraction_strategy.contraction_tree import ContractionTree
from hamld.contraction_strategy.mld_order_finder import GreedyMLDOrderFinder
from hamld.logging_config import setup_logger
import networkx as nx
import matplotlib.pyplot as plt

# 设置 logging 配置，放在模块级别
logger = setup_logger("./contraction_strategy/slice_finder", log_level=logging.INFO)

class SliceFinder:
    def __init__(self, contraction_tree: ContractionTree):
        """
        初始化 SliceFinder 实例。

        Args:
            contraction_tree (ContractionTree): 用于计算边收缩和相关操作的收缩树。
        """
        # slice information
        self.max_parallelism: int = 0  # 最大并行度
        self.memory_limit: int = 0  # 内存限制
        self.sliced_hyperedges: List[Tuple[str]] = []  # 切片后的超边列表
        self.sliced_contraction_cost: int = 0  # 收缩代价
        self.sliced_contraction_width: int = 0  # 收缩宽度
        
        # 初始化信息
        self.contraction_tree: ContractionTree = contraction_tree
        self.order: List[str] = contraction_tree.order
        self.hypergraph: DetectorErrorModelHypergraph = DetectorErrorModelHypergraph(self.contraction_tree.detector_error_model, have_logical_observable=True)
        self.connectivity_graph: ConnectivityGraph = ConnectivityGraph()

    def get_sliced_contraction_cost_information(self)->Tuple[int, int]:
        """ 取出收缩过程中的成本参数

        Returns:
            Tuple[int, int]: 切片之后, 每个子任务的收缩成本相关参数
        """
        return self.sliced_contraction_cost, self.sliced_contraction_width

    def slice_based_on_parallelism(self, sliced_hyperedge_count: int = 5, updated_order: bool = False) -> Tuple[int, int, List[Tuple[str]]]:
        """
        基于并行度限制选择切片超边。

        Args:
            sliced_hyperedge_count (int): 每个 slice 的超边数量，控制并行度上限。
            updated_order (bool): 是否在每次切片之后更新超图并重新计算 order。

        Returns:
            Tuple[int, int, List[Tuple[str]]]: (最终收缩成本, 最大收缩规模, 切片后的超边列表)。
        """
        if not updated_order:
            logger.debug(f"updated_order: {updated_order}")
            # 根据当前 order 选择并排序超边收缩，选择前 sliced_hyperedge_count 个
            sorted_edge_contractions = sorted(self.contraction_tree.edge_contractions, 
                                            key=lambda x: x[2] - x[1], reverse=True)[:sliced_hyperedge_count]
            self.sliced_hyperedges = [edge_contraction[0] for edge_contraction in sorted_edge_contractions]
            # 更新超图
            self.hypergraph = self.hypergraph.slice_hyperedges(self.sliced_hyperedges)
            self.connectivity_graph = ConnectivityGraph()
            self.connectivity_graph.hypergraph_to_connectivity_graph(self.hypergraph)
        else:
            # 如果需要更新 order，则逐步选择边进行切片
            for _ in range(sliced_hyperedge_count):
                # 基于当前超图计算边收缩，并选择收缩代价最大的边
                edge_contractions = ContractionTree.compute_edge_contractions(
                    order=self.order, hyperedges=self.hypergraph.get_hyperedges()
                )
                sorted_edge_contractions = sorted(edge_contractions, 
                                                key=lambda x: x[2] - x[1], reverse=True)[:sliced_hyperedge_count]
                sorted_edges = [edge_contraction[0] for edge_contraction in sorted_edge_contractions]
                # 筛选出不在已选择的 sliced_hyperedges 中的边
                remaining_edges = [edge for edge in sorted_edges if edge not in self.sliced_hyperedges]
                # 选择最前面的边
                selected_edge = [remaining_edges[0]]
                self.sliced_hyperedges.append(selected_edge)

                # 更新超图
                self.hypergraph = self.hypergraph.slice_hyperedges(selected_edge)
                logger.debug(f"slice step: {_}, selected_edge: {selected_edge}")
                # 更新收缩顺序（order）
                self.connectivity_graph = ConnectivityGraph()
                self.connectivity_graph.hypergraph_to_connectivity_graph(self.hypergraph)
                # self.connectivity_graph.draw()
                
                order_finder = GreedyMLDOrderFinder(self.connectivity_graph)
                self.order = order_finder.find_order()
                
                logger.debug(f"slice step: {_}, new order: {self.order}")
                logger.debug(f"slice step: {_}, sorted_edge_contractions: {sorted_edge_contractions}")
                logger.debug(f"new hypergraph hyperedges number: {self.hypergraph.get_hyperedges_number()}")
        
        logger.debug(f"self.sliced_hyperedges: {self.sliced_hyperedges}")
        logger.debug(f"after sliced hypergraph.get_hyperedges_number(): {self.hypergraph.get_hyperedges_number()}")
        # 计算基于切片超边后的收缩代价和收缩宽度
        self.sliced_contraction_cost, self.sliced_contraction_width = SliceFinder.compute_node_contraction_cost_by_sliced_hypergraph(
            self.order, self.hypergraph, []
        )
        logger.debug(f"sliced_contraction_cost: {self.sliced_contraction_cost}")
        logger.debug(f"sliced_contraction_width: {self.sliced_contraction_width}")
        return self.sliced_contraction_cost, self.sliced_contraction_width, self.sliced_hyperedges


    def slice_based_on_memory(self, sliced_contraction_width: int = 20, updated_order: bool = False) -> List[Tuple[str]]:
        """
        基于内存限制选择切片超边。

        Args:
            contraction_width (int): 每次切片后的收缩宽度。
            updated_order (bool): 是否在每次切片后更新 order。

        Returns:
            List[Tuple[str]]: 切片后的超边列表。
        """
        logger.info(f"The {self.slice_based_on_memory.__name__} function is pending implementation.")
        pass  # 根据具体实现补充内存限制逻辑

    def get_contraction_strategy(self):
        """
        获取收缩策略，用于指导 HAMLD 执行收缩操作。
        
        该方法返回当前收缩策略的相关信息，包括节点收缩顺序、切片超边、单次收缩成本以及切片后的收缩宽度。
        该方法是HAMLD执行收缩操作的关键部分, 具体的策略会根据输入的错误模型或其他参数动态生成。

        Returns:
            ContractionStrategy: 收缩策略对象，包含以下信息：
                - order (List[str]): 节点的收缩顺序。
                - sliced_hyperedges (List[Tuple[str]]): 切片后的超边列表。
                - sliced_contraction_cost (int): 单次收缩的成本。
                - sliced_contraction_width (int): 切片后的收缩宽度。
        """
        logger.info(f"Calling {self.get_contraction_strategy.__name__} - Pending implementation: "
                    f"Returning current strategy with order, sliced hyperedges, and cost/width information.")
        
        return (self.order, self.sliced_hyperedges, self.sliced_contraction_cost, self.sliced_contraction_width)
    
    @staticmethod
    def compute_node_contraction_cost_by_sliced_hypergraph(
        order: List[str], 
        hypergraph: DetectorErrorModelHypergraph, 
        sliced_hyperedges: List[Tuple[str]]
    ) -> Tuple[int, int]:
        """
        计算切片后的连通图上，按照给定顺序进行节点收缩的总代价和最大宽度。

        Args:
            order (List[str]): 节点收缩顺序列表。
            hypergraph (DetectorErrorModelHypergraph): 输入的超图。
            sliced_hyperedges (List[Tuple[str]]): 需要从超图中切片的超边列表。

        Returns:
            Tuple[int, int]: 总收缩代价和最大收缩宽度。
        """
        contraction_cost = 0
        contraction_width = 0
        current_contracted_nodes: Set[str] = set()
        current_prob_dist_related_nodes: Set[str] = set()
        
        try:
            # 如果切片的超边为空，则直接更新超图
            sliced_hypergraph = hypergraph.slice_hyperedges(sliced_hyperedges) if sliced_hyperedges else hypergraph
        except ValueError as e:
            logger.error(f"Error while slicing hyperedges: {e}")
            raise  # Reraise the exception to signal failure

        # 初始化连通图
        connectivity_graph = ConnectivityGraph()
        connectivity_graph.hypergraph_to_connectivity_graph(sliced_hypergraph)

        # 按照顺序进行节点收缩
        for step, current_contraction_node in enumerate(order):
            # 收缩节点并更新相关的概率分布节点
            current_contracted_nodes, current_prob_dist_related_nodes = ContractionTree.contract_node_and_update_prob_related_nodes(
                current_contraction_node, current_contracted_nodes, current_prob_dist_related_nodes, connectivity_graph
            )
            
            # 计算当前概率分布的维度
            current_prob_dist_dimension = len(current_prob_dist_related_nodes) - len(current_contracted_nodes)
            
            # 计算当前节点收缩的代价：2^当前概率分布维度
            current_contraction_cost = 2 ** current_prob_dist_dimension
            logger.debug(f"Step {step}: current_contraction_cost = {current_contraction_cost} (prob_dist_dimension = {current_prob_dist_dimension})")

            # 累加收缩代价
            contraction_cost += current_contraction_cost

            # 更新最大收缩宽度
            contraction_width = max(contraction_width, current_prob_dist_dimension)

        logger.debug(f"Total contraction cost: {contraction_cost}, Maximum contraction width: {contraction_width}")
        return contraction_cost, contraction_width
