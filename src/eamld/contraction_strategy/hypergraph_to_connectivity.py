import stim
import networkx as nx
import logging
from typing import List, Dict
from eamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
import matplotlib.pyplot as plt
import re
from eamld.logging_config import setup_logger

# 设置 logging 配置，放在模块级别
logger = setup_logger("./contraction_strategy/hypergraph_to_connectivity_graph", log_level=logging.INFO)

class ConnectivityGraph(nx.Graph):
    def __init__(self):
        """初始化一个连通图对象, 继承自nx.Graph"""
        super().__init__()
        self.detector_error_model = None
        self.hypergraph = None
        logger.debug("Connectivity graph initialized.")

    def hypergraph_to_connectivity_graph(self, hypergraph: DetectorErrorModelHypergraph, have_logical_observable: bool = False) -> None:
        """将超图转换为连通图

        Args:
            hypergraph (DetectorErrorModelHypergraph): 需要转换的超图对象
        """
        self.hypergraph = hypergraph
        nodes = self.hypergraph.get_nodes().copy()
        hyperedges = self.hypergraph.get_hyperedges().copy()
        
        logical_observable_number = self.hypergraph.logical_observable_number
        # logical_qubits = [f"L{i}" for i in range(logical_observable_number)]
        if have_logical_observable:
            # 如果可以存在逻辑观测节点，则保留节点
            pass
        elif have_logical_observable == False:
            if hypergraph.have_logical_observable:
                # 如果存在 logical observable，我们考虑将其从nodes和hyperedges中删除。
                for i in range(logical_observable_number):
                    if f"L{i}" in nodes:
                        # 异常处理，针对某些子图转化为连通图。
                        nodes.remove(f"L{i}")
                
                # 使用列表推导式去除元组中的 "L"
                hyperedges = [tuple(e for e in edge if "D" in e) for edge in hyperedges]

            
        # 将节点添加到图中
        self.add_nodes_from(nodes)

        # 遍历每个超边，拆解为普通边
        for hyperedge in hyperedges:
            self.add_edges_from_hyperedge(hyperedge)

    def dem_to_connectivity_graph(self, detector_error_model: stim.DetectorErrorModel, have_logical_observable: bool = False) -> None:
        """DEMs直接转化为连通图, 本质上也是先找到所有超边，再将超图转换为连通图

        Args:
            detector_error_model (stim.DetectorErrorModel): 检测器错误模型对象
        """
        self.detector_error_model = detector_error_model
        detector_number = self.detector_error_model.num_detectors
        detector_nodes = [f"D{i}" for i in range(detector_number)]
        self.add_nodes_from(detector_nodes)

        for error in self.detector_error_model:
            if error.type == "error":
                error_even = error.targets_copy()
                hyperedge = []
                for flip_object in error_even:
                    if have_logical_observable:
                        if flip_object.is_relative_detector_id():
                            hyperedge.append(f"D{flip_object.val}")
                        else:
                            hyperedge.append(f"L{flip_object.val}")
                    else:
                        if flip_object.is_relative_detector_id():
                            hyperedge.append(f"D{flip_object.val}")
                self.add_edges_from_hyperedge(hyperedge)

    def add_edges_from_hyperedge(self, hyperedge: List[str]) -> None:
        """根据超边添加图的边

        Args:
            hyperedge (List[str]): 超边节点列表
        """
        if len(hyperedge) > 1:
            # 更高效的生成器表达式来添加边
            edges = [(hyperedge[i], hyperedge[j]) for i in range(len(hyperedge)) for j in range(i + 1, len(hyperedge))]
            self.add_edges_from(edges)

    def add_node_pos_from_dem(self, detector_error_model: stim.DetectorErrorModel) -> None:
        """根据检测器坐标为图中的每个节点添加位置属性。
        
        注意：暂时不支持QLDPC code。

        Args:
            detector_error_model (stim.DetectorErrorModel): 检测器错误模型对象
        """
        detector_coordinates = detector_error_model.get_detector_coordinates()
        def extract_node_number(node_name: str) -> int:
            """提取节点名称中的数字部分并转换为整数。"""
            return int(re.sub(r"\D", "", node_name))  # 匹配所有非数字字符并替换为空字符串

        for node in self.nodes:
            # 提取节点编号并获取对应坐标
            node_number = extract_node_number(node)
            if node_number in detector_coordinates:
                self.nodes[node]['pos'] = detector_coordinates[node_number]
            else:
                logger.warning(f"Detector coordinates for node {node} not found!")

    def find_min_degree_node(self) -> str:
        """高效地找到度最小的节点。

        返回:
            str: 度最小的节点名称
        """
        # 使用 degree() 方法一次性计算所有节点的度
        return min(self.nodes, key=lambda node: self.degree(node))

    def draw(self, layout: str = 'spring', save_path: str = None, title = None) -> None:
        """绘制图形，支持不同的布局方式，并可选择保存为文件。

        Args:
            layout (str): 布局方式，默认为 'spring'，可选 'circular', 'spectral' 等。
            save_path (str): 如果提供，则将图像保存为该路径的文件（如 PDF），不显示。
        """

        plt.figure(figsize=(12, 8))
        if layout == 'spring':
            pos = nx.spring_layout(self)
        elif layout == 'circular':
            pos = nx.circular_layout(self)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self)
        else:
            logger.warning(f"Unknown layout type '{layout}', defaulting to 'spring'.")
            pos = nx.spring_layout(self)

        nx.draw(
            self,
            pos,
            with_labels=True,
            node_size=1600,
            node_color='skyblue',
            font_size=20,
            font_weight='bold'
        )
        if title:
            plt.title(title, fontsize=20)
        else:
            # plt.title('Connectivity Graph Representation')
            pass

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # def draw_3D(self): # 如果需要添加3D绘图功能，可以在这里扩展
    def get_nodes_pos(self) -> Dict[str, List[int]]:
        """
        获取图中所有节点的坐标信息。

        返回:
            Dict[str, List[int]]: 节点名与其对应坐标的字典。
                如果节点没有设置坐标，返回默认值 None。
        """
        
        return nx.get_node_attributes(self, 'pos', None)

    def get_nodes_pos(self) -> Dict[str, List[int]]:
        """获取图中所有节点的坐标信息"""
        return nx.get_node_attributes(self, 'pos', {})

    def draw_with_pos(self):
        """绘制图形，如果节点有 'pos' 属性则按位置绘制"""
        try:
            # 获取图中所有节点的 'pos' 属性
            pos = nx.get_node_attributes(self, 'pos')

            if not pos:
                raise AttributeError("Graph does not have 'pos' attribute for nodes.")
            
            logger.info("Graph has 'pos' attribute, proceeding with drawing.")

        except AttributeError as e:
            # 捕获异常并处理，显示错误信息
            logger.error(f"Error: {e}")

            # 如果没有 pos 属性，则使用 draw 方法绘制2D图形
            logger.info("No 'pos' attribute, calling 2D draw method.")
            self.draw()
            return  # 直接返回，结束绘制流程

        # 3D 绘图
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制节点
        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, label=node, s=50, c='skyblue', marker='o')
            ax.text(x, y, z, node, fontsize=10, ha='right', color='black')

        # 绘制边
        for edge in self.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, color='black', alpha=0.5)

        # 图例和展示
        ax.legend()
        plt.title('3D Graph Representation')
        plt.show()
