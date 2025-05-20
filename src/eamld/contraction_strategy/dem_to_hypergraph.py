import stim
import networkx as nx
# import hypernetx as hnx
import matplotlib.pyplot as plt
import logging
from decimal import Decimal
from typing import List, Union, Tuple, Dict
from eamld.logging_config import setup_logger
import copy

# 设置 logging 配置
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger = setup_logger("./contraction_strategy/dem_to_hypergraph", log_level=logging.WARNING)

class DetectorErrorModelHypergraph:
    def __init__(self, detector_error_model: stim.DetectorErrorModel, have_logical_observable: bool = False):
        """_summary_

        Args:
            detector_error_model (stim.DetectorErrorModel): Stim生成的检测器错误模型.
            have_logical_observable (bool, optional): 是否有logical_observable作为超图节点. Defaults to False.
            
        Note:
            这里的have_logical_observable只是删除L0节点, 不会删除边, 在构造连通图中, 这些边可能有用。
        """
        self.detector_error_model = detector_error_model
        self.nodes: List[str]
        self.hyperedges: List[Tuple[str]]
        self.weights: List[Decimal]
        self.have_logical_observable: bool = have_logical_observable
        
        self.detector_number: int = None
        self.logical_observable_number: int = None
        
        # 初始化超图结构
        self.nodes, self.hyperedges, self.weights = self.detector_error_model_to_hypergraph(detector_error_model)

    def get_nodes(self, have_logical_observable = True) -> List[str]:
        """返回节点列表"""
        if have_logical_observable:
            return self.nodes
        else:
            return [node for node in self.nodes if not node.startswith('L')]

    def get_hyperedges(self) -> List[Tuple[str]]:
        """返回超边列表"""
        return self.hyperedges
    
    def get_weights(self) -> List[Decimal]:
        """返回每条超边的权重"""
        return self.weights
    
    def get_nodes_number(self, have_logical_observable = True) -> int:
        """返回节点数量"""
        if have_logical_observable:
            return len(self.nodes)
        else:
            return len([node for node in self.nodes if not node.startswith('L')])
    
    def get_hyperedges_number(self) -> int:
        """返回超边数量"""
        return len(self.hyperedges)
    
    def get_hyperedges_weights_dict(self, hyperedges = None) -> Dict[Tuple[str], Decimal]:
        if len(self.hyperedges) != len(self.weights):
            raise ValueError("DetectorErrorModelHypergraph/get_hyperedges_weights_dict is error. In the hypergraph, the number of hyperedges is not necessarily equal to the number of weights.")
        hyperedges_weights_dict:Dict[Tuple[str], Decimal] = {}
        if hyperedges is None:
            for i in range(len(self.hyperedges)):
                hyperedges_weights_dict[self.hyperedges[i]] = self.weights[i]
        else:
            for hyperedge in hyperedges:
                hyperedges_weights_dict[hyperedge] = self.weights[self.hyperedges.index(hyperedge)]
        return hyperedges_weights_dict

    def get_sub_hypergraph(self, nodes: List[str]) -> "DetectorErrorModelHypergraph":
        """获取由指定节点构成的子超图。

        Args:
            nodes (List[str]): 要包含在子超图中的节点列表

        Returns:
            DetectorErrorModelHypergraph: 包含指定节点及其相连超边的新超图实例
        """
        # 确保输入节点都是当前超图中的节点
        invalid_nodes = set(nodes) - set(self.nodes)
        if invalid_nodes:
            raise ValueError(f"Nodes {invalid_nodes} not found in the hypergraph.")
        
        # 获取所有包含至少一个输入节点的超边
        sub_hyperedges = []
        sub_weights = []
        for edge, weight in zip(self.hyperedges, self.weights):
            if any(node in nodes for node in edge):
                sub_hyperedges.append(edge)
                sub_weights.append(weight)
        
        # 获取子超图中实际存在的节点(超边中出现的节点)
        actual_nodes = set()
        for edge in sub_hyperedges:
            actual_nodes.update(edge)
        
        # 创建新的超图实例
        new_hypergraph = type(self)(self.detector_error_model)
        new_hypergraph.nodes = sorted(actual_nodes)
        new_hypergraph.hyperedges = sub_hyperedges
        new_hypergraph.weights = sub_weights
        new_hypergraph.have_logical_observable = self.have_logical_observable
        
        return new_hypergraph
    
    def get_priority_sub_hypergraph(self, nodes: List[str], priority: int, topk: Union[int, None] = None) -> "DetectorErrorModelHypergraph":
        """获取优先级子超图，选择连接指定节点数量≥priority的超边，并按权重排序。

        Args:
            nodes (List[str]): 要包含在子超图中的节点列表
            priority (int): 超边需要连接的最小节点数量
            topk (Union[int, None], optional): 取排序后的前k个超边. Defaults to None(取所有).

        Returns:
            DetectorErrorModelHypergraph: 包含符合条件超边的新超图实例
        """
        # 确保输入节点都是当前超图中的节点
        invalid_nodes = set(nodes) - set(self.nodes)
        if invalid_nodes:
            raise ValueError(f"Nodes {invalid_nodes} not found in the hypergraph.")
        
        # 计算每条超边的priority(连接nodes的数量)和权重
        edge_info = []
        for edge, weight in zip(self.hyperedges, self.weights):
            connected_count = sum(node in nodes for node in edge)
            if connected_count >= priority:
                edge_info.append((edge, weight, connected_count))
        
        # 按priority降序，权重降序排序
        edge_info.sort(key=lambda x: (-x[2], -x[1]))
        
        # 如果指定了topk，则取前k个
        if topk is not None:
            edge_info = edge_info[:topk]
        
        # 提取排序后的超边和权重
        sub_hyperedges = [info[0] for info in edge_info]
        sub_weights = [info[1] for info in edge_info]
        
        # 获取子超图中实际存在的节点(超边中出现的节点)
        actual_nodes = set()
        for edge in sub_hyperedges:
            actual_nodes.update(edge)
        
        # 创建新的超图实例
        new_hypergraph = type(self)(self.detector_error_model)
        new_hypergraph.nodes = sorted(actual_nodes)
        new_hypergraph.hyperedges = sub_hyperedges
        new_hypergraph.weights = sub_weights
        new_hypergraph.have_logical_observable = self.have_logical_observable
        
        return new_hypergraph

    def get_new_priority_sub_hypergraph(self, nodes: List[str], priority: int, topk: Union[int, None] = None) -> "DetectorErrorModelHypergraph":
        """获取优先级子超图，选择连接指定节点数量≥priority的超边，并按权重排序。

        Args:
            nodes (List[str]): 要包含在子超图中的节点列表
            priority (int): 超边需要连接的最小节点数量
            topk (Union[int, None], optional): 取排序后的前k个超边. Defaults to None(取所有).

        Returns:
            DetectorErrorModelHypergraph: 包含符合条件超边的新超图实例
        """
        # 确保输入节点都是当前超图中的节点
        invalid_nodes = set(nodes) - set(self.nodes)
        if invalid_nodes:
            raise ValueError(f"Nodes {invalid_nodes} not found in the hypergraph.")
        
        # 计算每条超边的priority(连接nodes的数量)和权重
        edge_info = []
        for edge, weight in zip(self.hyperedges, self.weights):
            edge_priority = 0
            for node in edge:
                if node.startswith("D"):
                    if node in nodes:
                        edge_priority += 1
                    else:
                        edge_priority -= 1

            if edge_priority >= priority:
                edge_info.append((edge, weight, edge_priority))
        
        # 按priority降序，权重降序排序
        edge_info.sort(key=lambda x: (-x[2], -x[1]))
        
        # 如果指定了topk，则取前k个
        if topk is not None:
            edge_info = edge_info[:topk]
        
        # 提取排序后的超边和权重
        sub_hyperedges = [info[0] for info in edge_info]
        sub_weights = [info[1] for info in edge_info]
        
        # 获取子超图中实际存在的节点(超边中出现的节点)
        actual_nodes = set()
        for edge in sub_hyperedges:
            actual_nodes.update(edge)
        
        # 创建新的超图实例
        new_hypergraph = type(self)(self.detector_error_model)
        new_hypergraph.nodes = sorted(actual_nodes)
        new_hypergraph.hyperedges = sub_hyperedges
        new_hypergraph.weights = sub_weights
        new_hypergraph.have_logical_observable = self.have_logical_observable
        
        return new_hypergraph

    def partition_hypergraph(self, r, m: int, n: int,
                             have_data_stabilizer = False, have_connect_hyperedges = False,
                             partition_strategy = "connected num") -> Tuple[List["DetectorErrorModelHypergraph"], List[Dict[str, str]]]:
        """将r轮的检测器错误模型对应的超图划分为m层，并返回划分后的超图和detector映射关系。

        Args:
            r (int): 轮数
            m (int): 划分的层数
            n (int): 每轮的stabilizer数量（不等于第一轮的detector数）
            have_data_stabilizer (bool, optional): 是否存在data stabilizer. Defaults to False.
            have_connect_hyperedges (bool, optional): 是否存在连接超边. Defaults to False.
            partition_strategy (str, optional): 划分策略. Defaults to "connected num", "before layer".

        Returns:
            Tuple[List[DetectorErrorModelHypergraph], List[Dict[str, str]]]: 
                1. 划分后的m个超图
                2. 每层detector的映射关系
        """
        # 计算每层的轮数
        # all_detector_number = self.detector_number
        if have_data_stabilizer:
            # 如果存在data stabilizer, 则本身r轮的，会存在r+1轮的detector。
            r = r + 1
            
        base_rounds = r // m
        last_rounds = base_rounds + r % m
        
        # 初始化结果
        partitioned_hypergraphs = []
        detector_maps = []
        
        # 初始化detector起始索引
        start_idx = 0
        
        for layer in range(m):
            # 计算当前层的轮数
            rounds = base_rounds if layer < m - 1 else last_rounds
            
            # 计算当前层的detector范围
            if layer == 0:
                end_idx = start_idx + n // 2 + (rounds-1) * n
            else:
                end_idx = start_idx + rounds * n
            
            # 获取当前层的detector节点
            current_detectors = [f"D{i}" for i in range(start_idx, end_idx)]
            
            # 创建detector映射
            detector_map = {f"D{i}": f"D{i-start_idx}" for i in range(start_idx, end_idx)}
            detector_num = end_idx - start_idx
            detector_maps.append(detector_map)
            
            # 获取当前层的超边
            current_hyperedges = []
            current_weights = []
            logical_hyperedges = []
            
            connect_node_index = 0
            for edge, weight in zip(self.hyperedges, self.weights):
                # 如果超边连接的节点，所有检测器都属于当前层，或者连接的是logical_observable，则该超边属于该层。
                have_detector = any(node.startswith('D') for node in edge )
                
                if have_detector:
                    if all((node.startswith('D') and node in current_detectors) or (not node.startswith('D')) for node in edge):
                        # 映射detector名称
                        mapped_edge = tuple(detector_map.get(node, node) for node in edge)
                        current_hyperedges.append(mapped_edge)
                        current_weights.append(weight)
                    else:
                        # 是否考虑连接不同层的超边，将其归结到某个超图中。
                        if have_connect_hyperedges:
                            # 计算超边在当前层、上一层和下一层的连接节点数
                            current_layer_count = sum(node in current_detectors for node in edge)
                            prev_layer_count = sum(node.startswith('D') and int(node[1:]) < start_idx for node in edge)
                            next_layer_count = sum(node.startswith('D') and int(node[1:]) >= end_idx for node in edge)
                            
                            # 添加节点到当前层，本身是虚拟节点。
                            if current_layer_count != 0 and (prev_layer_count != 0 or next_layer_count != 0):
                                if partition_strategy == "connected num":
                                    if current_layer_count > max(prev_layer_count, next_layer_count) or current_layer_count == next_layer_count:
                                        # 当前层最多，则属于当前层。
                                        # 当前层等于下一轮，则属于当前层
                                        
                                        # 存在该带连接的超边连接的节点。
                                        for node in edge:
                                            # 不属于当前层的detector，我们添加到detector_map中。
                                            if node.startswith('D') and (int(node[1:]) < start_idx or int(node[1:]) >= end_idx):
                                                if node not in detector_map:
                                                    detector_map[node] = f"D{detector_num+connect_node_index}"
                                                    connect_node_index += 1
                                        
                                        mapped_edge = tuple(detector_map.get(node, node) for node in edge)
                                        current_hyperedges.append(mapped_edge)
                                        current_weights.append(weight)
                                        
                                elif partition_strategy == "before layer":
                                    # 存在当前层和下一层，默认存储在当前层
                                    if prev_layer_count == 0 and next_layer_count != 0:
                                        # 存在该带连接的超边连接的节点。
                                        for node in edge:
                                            # 不属于当前层的detector，我们添加到detector_map中。
                                            if node.startswith('D') and (int(node[1:]) < start_idx or int(node[1:]) >= end_idx):
                                                if node not in detector_map:
                                                    detector_map[node] = f"D{detector_num+connect_node_index}"
                                                    connect_node_index += 1
                                                    
                                        mapped_edge = tuple(detector_map.get(node, node) for node in edge)
                                        current_hyperedges.append(mapped_edge)
                                        current_weights.append(weight)
                else:
                    # 不包含detector的超边，添加到logical_hyperedges中。
                    logical_hyperedges.append(edge)
            
            # 创建新的超图实例
            new_hypergraph = type(self)(self.detector_error_model)
            new_hypergraph.nodes = sorted(detector_map.values())
            new_hypergraph.hyperedges = current_hyperedges
            new_hypergraph.weights = current_weights
            new_hypergraph.have_logical_observable = self.have_logical_observable
            # 该层内部的detector 数量
            new_hypergraph.detector_number = detector_num
            # 该层内部的logical_observable 数量，默认等于整体的逻辑数量。
            new_hypergraph.logical_observable_number = self.logical_observable_number
            
            partitioned_hypergraphs.append(new_hypergraph)
            
            # 更新起始索引
            start_idx = end_idx
        
        return partitioned_hypergraphs, detector_maps, logical_hyperedges


    def remove_nodes(self, nodes: List[str]) -> "DetectorErrorModelHypergraph":
        """从超图中移除指定节点及其相连的所有超边。

        Args:
            nodes (List[str]): 要从超图中删除的节点列表

        Returns:
            DetectorErrorModelHypergraph: 删除指定节点及其相连超边后的新超图实例
        """
        # 确保输入节点都是当前超图中的节点
        invalid_nodes = set(nodes) - set(self.nodes)
        if invalid_nodes:
            raise ValueError(f"Nodes {invalid_nodes} not found in the hypergraph.")
        
        # 获取所有不包含任何输入节点的超边
        sub_hyperedges = []
        sub_weights = []
        for edge, weight in zip(self.hyperedges, self.weights):
            if not any(node in nodes for node in edge):
                sub_hyperedges.append(edge)
                sub_weights.append(weight)
        
        # 获取子超图中实际存在的节点(超边中出现的节点)
        actual_nodes = set()
        for edge in sub_hyperedges:
            actual_nodes.update(edge)
        
        # 创建新的超图实例
        new_hypergraph = type(self)(self.detector_error_model)
        new_hypergraph.nodes = sorted(actual_nodes)
        new_hypergraph.hyperedges = sub_hyperedges
        new_hypergraph.weights = sub_weights
        new_hypergraph.have_logical_observable = self.have_logical_observable
        
        return new_hypergraph


    def detector_error_model_to_hypergraph(self, detector_error_model: stim.DetectorErrorModel)-> Tuple[List[str], List[Tuple[str]], List[Decimal]]:
        """将错误检测模型转换为超图。

        Args:
            detector_error_model (stim.DetectorErrorModel): 错误检测模型

        Returns:
            Tuple[List[str], List[List[str]], List[Decimal]]: 节点集合和超边集合
        """
        self.detector_number = detector_error_model.num_detectors
        self.logical_observable_number = detector_error_model.num_observables
        
        nodes = self.detector_and_logical_observable_number_to_hypernodes(self.detector_number, self.logical_observable_number)
        hyperedges, weights = self.detector_error_model_to_hyperedge(detector_error_model)
        return nodes, hyperedges, weights

    def detector_and_logical_observable_number_to_hypernodes(self, detector_number: int, logical_observable_number: int) -> List[str]:
        """获取超图节点集合。

        Args:
            detector_number (int): 错误检测模型中检测器的数量。
            logical_observable_number (int): 错误检测模型中逻辑可观测量的数量。
            
        Returns:
            List[str]: 超图节点集合。例如["D0","D1","D2","L0","L1"]
        """
        detector_nodes = [f"D{i}" for i in range(detector_number)]
        logical_observable_nodes = [f"L{i}" for i in range(logical_observable_number)]
        
        return detector_nodes + logical_observable_nodes if self.have_logical_observable else detector_nodes

    def detector_error_model_to_hyperedge(self, detector_error_model: stim.DetectorErrorModel) -> Tuple[List[str], List[Tuple[str]], List[Decimal]]:
        """将错误检测模型转换为超图。

        Args:
            detector_error_model (stim.DetectorErrorModel): 错误检测模型

        Returns:
            Tuple[List[str], List[List[str]], List[Decimal]]: 节点集合和超边集合
        """
        hyperedges = []
        weights = []
        # 通过遍历DEM得到的是dem_instruction，其中默认对于error参数进行处理，转换为double/float的精度，但是这个精度不够。
        
        dem_lines = str(detector_error_model).split('\n')
        
        for i in range(len(dem_lines)):
            dem_instruction = detector_error_model[i]
            if dem_instruction.type == "error":
                dem_line = dem_lines[i]
                error_even = dem_instruction.targets_copy()
                # weight = error.args_copy()[0]
                if dem_line.startswith('error'):
                    # 提取括号内的数值
                    start_index = dem_line.find('(') + 1
                    end_index = dem_line.find(')')
                    value_str = dem_line[start_index:end_index]
                    try:
                        # 将数值转换为 Decimal 类型
                        weight = Decimal(value_str)
                    except ValueError:
                        print(f"无法将 '{value_str}' 转换为 Decimal 类型。")
                hyperedge = self.error_even_to_hyperedge(error_even)
                hyperedges.append(hyperedge)
                weights.append(weight)

        return hyperedges, weights

    def error_even_to_hyperedge(self, error_even: List[stim.DemTarget]) -> Tuple[str]:
        """从一个error事件中,转换为一条超图边hyperedge。即错误事件翻转的detector和logical_observable的index。

        Args:
            error_even (List[stim.DemTarget]): 错误事件中的detector或logical_observable对象
            
        Returns:
            Tuple[str]: 超图对应的边. 例如("D0","D1","D2")
        """
        hyperedge = []
        for flip_object in error_even:
            if flip_object.is_relative_detector_id():
                hyperedge.append(f"D{flip_object.val}")
            elif self.have_logical_observable and flip_object.is_logical_observable_id():
                hyperedge.append(f"L{flip_object.val}")
        return tuple(hyperedge)
    
    # def to_hypernetx_hypergraph(self) -> hnx.Hypergraph:
    #     """将超图转换为hypernetx.Hypergraph对象。

    #     Returns:
    #         hypernetx.Hypergraph: 超图对象
        
    #     use H.get_properties(0, level=0, prop_name='weight') to get the weight of the edge 0.
    #     """
    #     return hnx.Hypergraph(self.hyperedges, edge_properties={i:{'weight':self.weights[i]} for i in range(len(self.hyperedges))})
    
    def draw_bipartite_graph(self, nodes: Union[int, None] = None, hyperedges: Union[int, None] = None) -> None:
        """绘制二部图。

        Args:
            nodes (Union[int, None], optional): 超图的节点. Defaults to None.
            hyperedges (Union[int, None], optional): 超图的边. Defaults to None.
        """
        if nodes is None or hyperedges is None:
            nodes = self.nodes
            hyperedges = self.hyperedges
            
        # 创建一个空的二部图
        B = nx.Graph()

        # 添加节点
        for node in nodes:
            B.add_node(node, bipartite=0)  # 将节点分类为一类

        # 添加超边作为新的节点类型
        for idx, hyperedge in enumerate(hyperedges):
            hyperedge_name = f'edge_{idx}'
            B.add_node(hyperedge_name, bipartite=1)  # 超边作为另一类节点
            for node in hyperedge:
                B.add_edge(hyperedge_name, node)  # 连接超边与它包含的节点

        # 调整布局参数，使节点更加分散
        pos = nx.spring_layout(B, k=0.6, scale=2, iterations=100)

        # 进行绘制
        nx.draw(B, pos, with_labels=True, node_color=['lightblue' if B.nodes[node]['bipartite'] == 0 else 'lightgreen' for node in B], 
                font_size=8, node_size=600, edge_color='gray')

        # 显示图像
        plt.show()
        
    def copy(self) -> "DetectorErrorModelHypergraph":
        """
        创建当前超图的副本，拷贝必要的属性。

        Returns:
            DetectorErrorModelHypergraph: 当前超图的副本。
        """
        # 创建一个新的 DetectorErrorModelHypergraph 实例
        new_hypergraph = type(self)(self.detector_error_model)
        # 浅拷贝超边列表和节点列表
        new_hypergraph.hyperedges = self.hyperedges[:]
        new_hypergraph.nodes = self.nodes[:]
        # 拷贝是否存在逻辑可观察量的标志
        new_hypergraph.have_logical_observable = self.have_logical_observable
        return new_hypergraph
    
    def slice_hyperedges(self, sliced_hyperedges: List[Tuple[str]]) -> "DetectorErrorModelHypergraph":
        """
        从超图中移除指定的超边。

        Args:
            sliced_hyperedges (List[Tuple[str]]): 需要移除的超边列表。

        Raises:
            ValueError: 如果要移除的超边在超图中找不到，则抛出异常。
        
        Returns:
            DetectorErrorModelHypergraph: 移除指定超边后的新超图实例。
        """
        # 确保 sliced_hyperedges 为嵌套列表的形式
        if isinstance(sliced_hyperedges, list) and not isinstance(sliced_hyperedges[0], tuple):
            sliced_hyperedges = [tuple(edge) for edge in sliced_hyperedges]

        # 创建当前超图的副本，以确保不修改原始超图
        new_hypergraph = self.copy()
        
        # 移除指定的超边
        for sliced_hyperedge in sliced_hyperedges:
            if sliced_hyperedge in new_hypergraph.hyperedges:
                new_hypergraph.hyperedges.remove(sliced_hyperedge)
            else:
                # 如果超边不存在，抛出异常
                raise ValueError(f"Hyperedge {sliced_hyperedge} not found in the hypergraph.")
        
        # 返回移除超边后的新超图实例，支持链式调用
        return new_hypergraph
