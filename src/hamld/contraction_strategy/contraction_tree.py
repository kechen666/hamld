from typing import List, Dict, Tuple, Set
import stim
import logging
from hamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
from hamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph

from hamld.logging_config import setup_logger
import networkx as nx
import matplotlib.pyplot as plt

# 设置 logging 配置，放在模块级别
logger = setup_logger("./contraction_strategy/contraction_tree", log_level=logging.INFO)

# TODO: 目前直接利用detector_error_model转化为contraction tree，后续考虑利用hypergraph作为输入，或者只用connectivity graph作为输入来只计算收缩成本。
class ContractionTree:
    def __init__(self, order: List[str], detector_error_model: stim.DetectorErrorModel = None, 
                 hypergraph: DetectorErrorModelHypergraph = None, 
                 connectivity_graph: ConnectivityGraph = None, 
                 connectivity_graph_have_logical_observable: bool = False):
        """
        初始化收缩树。

        Args:
            order (List[str]): 收缩节点的顺序列表。
            detector_error_model (stim.DetectorErrorModel): 输入的检测器误差模型。
            hypergraph (DetectorErrorModelHypergraph): 输入的超图。
            connectivity_graph (ConnectivityGraph): 输入的连通图。
            connectivity_graph_have_logical_observable: 是否包含逻辑可观测值，默认不包含？估计可以会和实际的存在一点偏差，因为没有考虑逻辑可观测值的影响。
        """
        # 检查输入参数并构建必要的图结构
        if connectivity_graph is None:
            if hypergraph is None:
                if detector_error_model is None:
                    raise ValueError("必须提供至少一个参数：detector_error_model、hypergraph或connectivity_graph")
                hypergraph = DetectorErrorModelHypergraph(detector_error_model, have_logical_observable=True)
            connectivity_graph = ConnectivityGraph()
            connectivity_graph.hypergraph_to_connectivity_graph(hypergraph, connectivity_graph_have_logical_observable)
            
        self.order: List[str] = order
        self.step_number: int = len(order)  # 总步骤数
        self.detector_error_model = detector_error_model
        self.hypergraph = hypergraph
        self.connectivity_graph = connectivity_graph
        
        
        # 边收缩信息
        self.edge_contractions: List[Tuple[Tuple[str], int, int]] = []  # 边收缩信息（例如 (('D0', 'D2'), 3, 5)）
        self.edge_contraction_steps: Dict[Tuple[str], int] = {}  # 边收缩步骤，用字典表示，便于查找操作

        # 收缩成本
        self.contraction_cost: int = 0
        self.contraction_width: int = 0  # 收缩宽度
        self.contraction_cost_each_step: Dict[str, int] = {}  # 每步的收缩成本
        

        # 树结构
        self.root: str = order[0]
        self.tree_graph: nx.Graph = nx.Graph()
        self.tree_graph.add_nodes_from([f"D{i}" for i in range(self.step_number)])

        for step in range(self.step_number):
            stem_node = f"#{step}"
            self.tree_graph.add_node(stem_node)
            if step == 0:
                self.tree_graph.add_edge(order[0], stem_node)
                self.tree_graph.nodes[order[0]]['parent'] = stem_node
            elif step >= 1:
                self.tree_graph.add_edge(order[step], stem_node)
                self.tree_graph.add_edge(f"#{step-1}", stem_node)
                self.tree_graph.nodes[order[step]]['parent'] = stem_node
                self.tree_graph.nodes[f"#{step-1}"]['parent'] = stem_node
        
        # compute some information
        self.compute_node_contraction_cost()
        self.compute_edge_contraction_information()
        
    def get_contraction_cost_information(self)->Tuple[int, int, Dict[str, int]]:
        """ 取出收缩过程中的成本参数

        Returns:
            Tuple[int, int, Dict[str, int]]: 收缩成本相关参数
        """
        return self.contraction_cost, self.contraction_width, self.contraction_cost_each_step
    
    def get_contractions_hyperedge_information(self)->Tuple[List[Tuple[Tuple[str], int, int]], Dict[Tuple[str], int]]:
        """获取收缩过程中的超边情况

        Returns:
            Tuple[List[Tuple[Tuple[str], int, int]], Dict[Tuple[str], int]]: 超边收缩情况
        """
        return self.edge_contractions, self.edge_contraction_steps

    def is_brach(self, edge: Tuple[str]):
        """判断当前边是不是分支

        Args:
            edge (Tuple[str]): tree graph 的某个边
        """
        # 检查边的两个端点是否包含 "D"
        return any("D" in node for node in edge)

    def is_stem(self, edge: Tuple[str]):
        """判断当前边是不是主干

        Args:
            edge (Tuple[str]): tree graph 的某个边
        """
        # 检查边的两个端点是否包含 "#"
        return  all(node.startswith("#") for node in edge)

    def is_leaf(self, node: str) -> bool:
        """
        判断节点是否是叶节点。

        Args:
            node (str): 节点名称。
        
        Returns:
            bool: 如果是叶节点返回 True，否则返回 False。
        """
        return "#" not in node
    
    def is_root(self, node: str) -> bool:
        """
        判断节点是否是root（开始收缩的节点）。

        Args:
            node (str): 节点名称。
        
        Returns:
            bool: 如果是root节点返回 True，否则返回 False。
        """
        return node == self.root
    
    def is_stem_node(self, node: str) -> bool:
        """
        判断节点是否是非叶子节点，主干节点。

        Args:
            node (str): 节点名称。
        
        Returns:
            bool: 如果是非叶子节点，主干节点，返回 True，否则返回 False。
        """
        return "#" in node
    
    def compute_node_contraction_cost(self) -> Tuple[int, int]:
        """
        计算每个节点的收缩成本并更新总收缩成本和最大收缩宽度。
        """
        current_prob_dist_dimension: int = 0
        current_contracted_nodes: Set[str] = set()
        current_prob_dist_related_nodes: Set[str] = set()
        self.contraction_cost = 0

        # connectivity_graph = ConnectivityGraph()
        # connectivity_graph.dem_to_connectivity_graph(self.detector_error_model)
        connectivity_graph = self.connectivity_graph
        
        for step in range(self.step_number):
            current_contraction_node = self.order[step]
            current_contracted_nodes, current_prob_dist_related_nodes = ContractionTree.contract_node_and_update_prob_related_nodes(
                current_contraction_node, current_contracted_nodes, current_prob_dist_related_nodes, connectivity_graph
            )
            current_prob_dist_dimension = len(current_prob_dist_related_nodes) - len(current_contracted_nodes)
            # current_contraction_cost = 2^current_prob_dist_dimension, detail can see the paper.
            current_contraction_cost = 2 ** current_prob_dist_dimension
            self.contraction_cost_each_step[current_contraction_node] = current_contraction_cost
            
            # save information in each stem node.
            self.tree_graph.nodes[f"#{step}"]["contraction_cost"] = current_contraction_cost

            self.contraction_cost += current_contraction_cost
            if current_prob_dist_dimension > self.contraction_width:
                self.contraction_width = current_prob_dist_dimension
        return self.contraction_cost, self.contraction_width
    
    @staticmethod
    def contract_node_and_update_prob_related_nodes(
        node_to_contract: str, contracted_nodes: Set[str], prob_related_nodes: Set[str], connectivity_graph: ConnectivityGraph
    ) -> Tuple[Set[str], Set[str]]:
        """
        收缩节点并更新相关节点集。

        Args:
            node_to_contract (str): 要收缩的节点名称。
            contracted_nodes (Set[str]): 已收缩节点集合。
            prob_related_nodes (Set[str]): 与收缩节点相关的节点集合。
            connectivity_graph (ConnectivityGraph): 连接性图，用于获取邻居节点。

        Returns:
            Tuple[Set[str], Set[str]]: 更新后的已收缩节点和相关节点集合。
        """
        # 将当前收缩节点添加到相关节点集合中
        contracted_nodes.add(node_to_contract)
        prob_related_nodes.add(node_to_contract)

        # 获取当前收缩节点的邻居节点
        neighboring_nodes = connectivity_graph.neighbors(node_to_contract)

        # 将邻居节点添加到相关节点集合
        prob_related_nodes.update(neighboring_nodes)

        return contracted_nodes, prob_related_nodes

    def compute_edge_contraction_information(self):
        """计算收缩过程中边的相关信息。

        更新边收缩信息 `self.edge_contractions` 和 `self.edge_contraction_steps`。 
        `self.edge_contractions` 存储了每条边及其收缩过程中的最小和最大步骤； 
        `self.edge_contraction_steps` 存储每条边在某步骤收缩的状态。

        """
        # 创建超图并获取所有超边
        hypergraph = self.hypergraph
        # hypergraph = DetectorErrorModelHypergraph(self.detector_error_model, have_logical_observable=True)
        hyperedges = hypergraph.get_hyperedges()
        contracted_hyperedges: Set[Tuple[str]] = set()

        # 遍历每个步骤进行边收缩
        for step in range(self.step_number):
            current_contraction_node = self.order[step]
            logger.debug(f"当前收缩节点: {current_contraction_node}")

            # 找到包含当前节点且尚未收缩的超边
            current_contraction_hyperedges = [
                hyperedge for hyperedge in hyperedges
                if current_contraction_node in hyperedge and hyperedge not in contracted_hyperedges
            ]

            # 更新已收缩的超边集合
            contracted_hyperedges.update(current_contraction_hyperedges)

            logger.debug(f"当前收缩的超边: {current_contraction_hyperedges}")

            # 更新边收缩步骤
            for hyperedge in current_contraction_hyperedges:
                self.edge_contraction_steps[hyperedge] = step
            
            # 将边收缩信息添加到tree graph的边中。
            self.tree_graph[self.order[step]][f"#{step}"]["step"] = step
            self.tree_graph[self.order[step]][f"#{step}"]["contraction_hyperedges"] = current_contraction_hyperedges

        self.edge_contractions = ContractionTree.compute_edge_contractions(self.order, hyperedges)

        logger.debug(f"边收缩信息: {self.edge_contractions}")
        
    @staticmethod
    def compute_edge_contractions(order:List[str], hyperedges: List[Tuple[str]]) -> List[Tuple[Tuple[str], int, int]]:
        # 创建节点步数映射，L0 对应 -1
        node_map_step = {order[step]: step for step in range(len(order))}
        node_map_step["L0"] = -1
        logger.debug(f"节点到步数的映射: {node_map_step}")

        # 将超边映射为数字，并保存其中的（初始节点，最小值，最大值）
        edge_contractions = ContractionTree.map_edges_to_numbers(node_map_step, hyperedges)
        return edge_contractions

    @staticmethod
    def map_edges_to_numbers(node_mapping: Dict[str, int], edge_list: List[Tuple[str]]) -> List[Tuple[Tuple[str], int, int]]:
        """
        将节点边列表映射为数字，并保存其中的（初始节点，最小值，最大值）。

        Args:
            node_mapping (Dict[str, int]): 节点名称到数字的映射字典。
            edge_list (List[Tuple[str]]): 包含节点对的边列表，例如 [('D0', 'D3'), ('D3',), ('D3', 'D7')]。

        Returns:
            List[Tuple[Tuple[str], int, int]]: 每个边及其最小、最大值的三元组。
        """
        result = []

        for edge in edge_list:
            # 将节点名称映射为数字，排除 'L0' 节点（如果存在）
            mapped_values = [node_mapping[node] for node in edge if node_mapping.get(node, -1) != -1]

            # 获取最小值和最大值
            if mapped_values:  # 确保 mapped_values 不为空
                min_value = min(mapped_values)
                max_value = max(mapped_values)
                result.append((edge, min_value, max_value))

        return result


    def draw_contraction_tree(self, layout: str = 'spring') -> None:
        """
        绘制收缩树图形，支持不同布局方式。

        Args:
            layout (str): 布局方式，默认为 'spring'，可选择 'circular', 'spectral' 等。
        """
        plt.figure(figsize=(12, 8))

        if layout == 'spring':
            pos = nx.spring_layout(self.tree_graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.tree_graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.tree_graph)
        else:
            logger.warning(f"Unknown layout type '{layout}', defaulting to 'spring'.")
            pos = nx.spring_layout(self.tree_graph)

        nx.draw(self.tree_graph, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=10, font_weight='bold')
        plt.title('Contraction Tree Representation')
        plt.show()
