from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import logging
import time
from hamld.logging_config import setup_logger
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor  # 使用多进程
# from heapq import nlargest, nsmallest
import heapq

from enum import Enum

from hamld.contraction_strategy.mld_order_finder import GreedyMLDOrderFinder
from hamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph

# 定义四种策略的枚举类型
class ApproximateStrategy(Enum):
    NODE_TOPK = 'node_topk'
    HYPEREDGE_TOPK = 'hyperedge_topk'
    NODE_THRESHOLD = 'node_threshold'
    HYPEREDGE_THRESHOLD = 'hyperedge_threshold'
    NO_NO = 'no_no'

# 存储每种策略对应的参数
strategy_params = {
    ApproximateStrategy.NODE_TOPK: 10**3,
    ApproximateStrategy.HYPEREDGE_TOPK: 10**4,
    ApproximateStrategy.NODE_THRESHOLD: 10**-8,
    ApproximateStrategy.HYPEREDGE_THRESHOLD: 10**-9,
    ApproximateStrategy.NO_NO: None,
}

def build_hyperedge_contraction_caches(contractable_hyperedges_weights_dict: Dict[str, Union[np.float32, np.float64, np.float128]], order: List[str]):
    """构建用于收缩的缓存构建方法"""
    hyperedge_cache = {}  # hyperedge -> list(parts)
    relevant_hyperedge_cache = {}  # detector -> set(hyperedges)
    
    # 使用集合提高查找效率
    remaining_hyperedges = set(contractable_hyperedges_weights_dict.keys())
    
    for detector in order:
        relevant_set = set()
        hyperedges_to_remove = set()
        
        for hyperedge in remaining_hyperedges:
            # 缓存分割结果
            if hyperedge not in hyperedge_cache:
                hyperedge_cache[hyperedge] = hyperedge.split(',')
            
            # 检查是否包含当前detector
            if detector in hyperedge_cache[hyperedge]:
                relevant_set.add(hyperedge)
                hyperedges_to_remove.add(hyperedge)
        
        relevant_hyperedge_cache[detector] = relevant_set
        remaining_hyperedges -= hyperedges_to_remove  # 批量移除
    return (hyperedge_cache, relevant_hyperedge_cache)

# 设置 logging 配置，放在模块级别
logger = setup_logger("contraction_executor/priority_approx_contraction_executor_qldpc", log_level=logging.WARNING)

def str_to_binary_bitwise(s: str) -> Tuple[bool]:
    """
    将二进制字符串转换为布尔数组
    :param s: 二进制字符串，例如 "1010"
    :return: 布尔元组，例如 (True, False, True, False)
    """
    return tuple(c == '1' for c in s)

def binary_to_str(binary: np.ndarray | Tuple[bool]) -> str:
    """
    将布尔数组或np.ndarray[bool]转换为二进制字符串
    :param binary: 布尔数组或np.ndarray[bool]
    :return: 二进制字符串，例如 "1010"
    """
    if isinstance(binary, np.ndarray):
        return ''.join('1' if bit else '0' for bit in binary)
    elif isinstance(binary, tuple):
        return ''.join('1' if bit else '0' for bit in binary)
    else:
        raise ValueError("Unsupported type for binary_to_str. Expected np.ndarray or Tuple[bool].")

def check_str_value(str_syndrome: str, position: int, expected_value: str) -> bool:
    """
    此函数用于检测字符串 str_syndrome 的指定位是否等于预期值。
    :param str_syndrome: 要检测的二进制字符串
    :param position: 要检测的位的位置，从 0 开始计数（从左到右）
    :param bit_length: 字符串的总长度
    :param expected_value: 待检测的值，应为 0 或 1
    :return: 如果指定位等于预期值，返回 True；否则返回 False
    """
    # 确保预期值是 0 或 1
    assert expected_value in ("0", "1"), "预期值必须是 0 或 1。"
    # 直接比较字符
    return str_syndrome[position] == expected_value

def hamming_distance_numpy(s1, s2):
    # 默认对其，其中某个为测量的syndrome，某个为(syndrome+logical observable)。其中默认s1为syndrome，s2为(syndrome+logical observable)。
    len1 = len(s1)
    len2 = len(s2)

    # 对齐 s2 的长度：截断或补齐
    if len2 < len1:
        s2 = s2 + '0' * (len1 - len2)
    elif len2 > len1:
        s2 = s2[:len1]

    return np.sum(
        np.frombuffer(s1.encode(), dtype=np.uint8) != 
        np.frombuffer(s2.encode(), dtype=np.uint8)
    )

def build_hyperedge_caches(contractable_hyperedges_weights_dict, order):
    """优化后的缓存构建方法"""
    hyperedge_cache = {}  # hyperedge -> list(parts)
    relevant_hyperedge_cache = {}  # detector -> set(hyperedges)
    
    # 使用集合提高查找效率
    remaining_hyperedges = set(contractable_hyperedges_weights_dict.keys())
    
    for detector in order:
        relevant_set = set()
        hyperedges_to_remove = set()
        
        for hyperedge in remaining_hyperedges:
            # 缓存分割结果
            if hyperedge not in hyperedge_cache:
                hyperedge_cache[hyperedge] = hyperedge.split(',')
            
            # 检查是否包含当前detector
            if detector in hyperedge_cache[hyperedge]:
                relevant_set.add(hyperedge)
                hyperedges_to_remove.add(hyperedge)
        
        relevant_hyperedge_cache[detector] = relevant_set
        remaining_hyperedges -= hyperedges_to_remove  # 批量移除
    return hyperedge_cache, relevant_hyperedge_cache

# 基于new_priority_approx_contraction_executor_qldpc.py函数进行修改
class SyndromePriorityApproximateContractionExecutorLog:
    def __init__(self, detector_number: int, logical_number: int, order: List[str],
                 sliced_hyperedges: List[str],
                 contractable_hyperedges_weights_dict: Dict[str, Union[np.float32, np.float64, np.float128]],
                 accuracy: str = "float64", approximatestrategy: str = "hyperedge_threshold",
                 approximate_param: Optional[int | float] = None,
                 hyperedge_cache: Optional[Dict] = None, relevant_hyperedge_cache: Optional[Dict] = None, hypergraph = None,
                 priority: int = 1, priority_topk: int = 10,
                 gamma = 23):
        """
        初始化 ContractionExecutor 对象。

        Args:
            detector_number (int): 检测器比特的数量。
            logical_number (int): 逻辑量子比特的数量。
            order (List[str]): 检测器的收缩顺序。
            sliced_hyperedges (List[str]): 用于并行任务执行的切片超边。
            contractable_hyperedges_weights_dict (Dict[str, Union[np.float32, np.float64, np.float128]]): 每个可收缩超边的权重。
            accuracy (str): 计算使用的数值精度，可选 'float32', 'float64', 或 'float128'。
            approximatestrategy (str): 近似策略，默认为 "hyperedge_threshold"。
            approximate_param (Optional[int | float]): 近似策略的参数，可以是整数（如 top-k 中的 k）或浮点数（如阈值），默认为 None。
        """
        self.detector_number = detector_number
        self.logical_number = logical_number
        self.total_length = detector_number + logical_number
        self.order = order
        self.sliced_hyperedges = sliced_hyperedges
        self.accuracy = accuracy
        self.priority = priority
        self.priority_topk = priority_topk
        self.min_hamming_weight = None
        self.gamma = gamma
        
        # sub
        self.flipped_detector_set: set = None
        self.hypergraph = hypergraph
        
        self._execution_contraction_time: float = 0
        self._execution_max_distribution_size: int = 0

        self.contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict
        self.const_1 = self._to_numeric_type(0)
        
        # 近似策略以及近似参数
        valid_strategies = ["node_topk", "hyperedge_topk", "node_threshold", "hyperedge_threshold", "no_no"]
        if approximatestrategy not in valid_strategies:
            raise ValueError(f"approximatestrategy 必须是以下之一: {valid_strategies}")
        self.approximatestrategy: str = approximatestrategy
        self._approximatestrategy_method = self.approximatestrategy.split("_")[1]
        self._is_topk = self._approximatestrategy_method == "topk"
        self._is_threshold = self._approximatestrategy_method == "threshold"
        self._approximate_position: str = approximatestrategy.split("_")[0]
        self.approximate_param: int | float = approximate_param
        
        # 验证 approximate_param 的有效性
        if approximate_param is not None:
            if approximatestrategy.endswith("_topk") and not isinstance(approximate_param, int):
                raise ValueError(f"当 approximatestrategy 为 {approximatestrategy} 时，approximate_param 必须是整数。")
            if approximatestrategy.endswith("_threshold") and not isinstance(approximate_param, (int, float)):
                raise ValueError(f"当 approximatestrategy 为 {approximatestrategy} 时，approximate_param 必须是数字。")
        else:
            self.approximate_param = strategy_params[ApproximateStrategy(approximatestrategy)]

        if self._is_threshold:
            self.approximate_param = self._to_numeric_type(self.approximate_param)
        if hyperedge_cache == None:
            self.hyperedge_cache = {}  # hyperedge -> list(parts)
        else:
            self.hyperedge_cache = hyperedge_cache
        if relevant_hyperedge_cache == None:
            self.relevant_hyperedge_cache = {}  # detector -> set(hyperedges)
        else:
            self.relevant_hyperedge_cache = relevant_hyperedge_cache
            
        # if hyperedge_cache == None or relevant_hyperedge_cache == None:
            # self.hyperedge_cache, self.relevant_hyperedge_cache = self.build_hyperedge_caches(self.contractable_hyperedges_weights_dict, self.order)
            
    def _priority_sub_hypergraph_contraction_strategy(self):
        # 获取包含所有翻转检测器的子超图
        # sub_hypergraph = self.hypergraph.get_sub_hypergraph(nodes = self.flipped_detector_set)
        sub_priorty_hypergraph = self.hypergraph.get_new_priority_sub_hypergraph(nodes = self.flipped_detector_set, priority = self.priority, topk = self.priority_topk)
        
        # 从子超图中获取原始超边权重字典
        original_dict = sub_priorty_hypergraph.get_hyperedges_weights_dict()
        
        # 转换超边权重字典格式：将元组键转换为逗号分隔的字符串
        sub_contractable_hyperedges_weights_dict: dict[str, float] = {
            ",".join(key): -np.log(float(value)/(1-float(value))) for key, value in original_dict.items()
        }

        # 创建连通图并构建子超图的连通性结构
        sub_connectivity_graph = ConnectivityGraph()
        sub_connectivity_graph.hypergraph_to_connectivity_graph(
            sub_priorty_hypergraph, 
            have_logical_observable = False  # 不包含逻辑可观测节点
        )
        
        # 使用贪心算法寻找最优收缩顺序
        order_finder = GreedyMLDOrderFinder(sub_connectivity_graph)
        order = order_finder.find_order()
        
        return sub_contractable_hyperedges_weights_dict, order

    def approximate_distribution(self, updated_prob_dist: Dict[str, Union[np.float32, np.float64, np.float128]], str_syndrome: str) -> Dict[str, Union[np.float32, np.float64, np.float128]]:
        """
        Approximate the probability distribution based on the specified strategy and parameter.
        """
        approximate_param = self.approximate_param
        if self._is_topk:
            if len(updated_prob_dist) <= approximate_param:
                logger.info(f"没有执行更新操作，概率分布为{len(updated_prob_dist)} 小于 {approximate_param}")
                return updated_prob_dist
            
            logger.info(f"执行了更新操作，更新前的概率分布大小为{len(updated_prob_dist)}")
            # Python 要求最小排序；我们构造最大堆来维护最小 approximate_param 个元素
            max_heap = []  # 存储 (-score, key)
            
            # 更新value值，为log(v)/log(1-v) * |str_syndrome \oplus k|的汉明权重
            for k, v in updated_prob_dist.items():
                hamming_weight = hamming_distance_numpy(str_syndrome, k)

                # # 更新当前最小汉明权重
                # if self.min_hamming_weight is None or self.min_hamming_weight > hamming_weight:
                #     self.min_hamming_weight = hamming_weight

                # # 跳过距离过远的解
                # if hamming_weight >= self.min_hamming_weight + self.gamma:
                #     continue

                score = -v * (hamming_weight)

                # 构造最大堆元素，按 -score 排序（score越小越优先），
                # 相同score时概率越大越优先（用 -v 保证）
                heap_item = (score, k)
                
                if len(max_heap) < approximate_param:
                    heapq.heappush(max_heap, heap_item)
                else:
                    # 仅当当前元素优先级高于堆顶时替换堆顶元素
                    if heap_item > max_heap[0]:
                        heapq.heappushpop(max_heap, heap_item)

            top_keys = [k for _, k in max_heap]

            return {k: updated_prob_dist[k] for k in top_keys}
        elif self._is_threshold:
            # 使用字典推导式直接返回结果
            return {k: v for k, v in updated_prob_dist.items() if v > approximate_param}
        else:
            raise ValueError(f"Invalid approximate strategy: {self.approximatestrategy}")
        
    def flip_bits(self, binary_str: str, hyperedge: str) -> str:
        """
        Flip the specified bits in a binary string based on the hyperedge.

        Args:
            binary_str (str): The binary string. D0对应最高位， D1对应次高位，依此类推，直到L0，同时L_n为最低位。
            hyperedge (str): The hyperedge specifying which bits to flip (e.g., 'D0','L1'.
            其中self.logical_number为逻辑量子比特的数量，即L的数量，self.detector_number为探测器的数量，即D的数量。
        Returns:
            str: The binary string after flipping the specified bits.

        Raises:
            ValueError: If an invalid bit label format is encountered in the hyperedge.
        """
        # 使用bytearray进行高效位操作
        binary_bytes = bytearray(binary_str, 'ascii')
        
        if hyperedge not in self.hyperedge_cache:
            hyperedge_list = hyperedge.split(',')
            self.hyperedge_cache[hyperedge] = hyperedge_list
        else:
            hyperedge_list = self.hyperedge_cache[hyperedge]
            
        for bit_label in hyperedge_list:
            if bit_label[0] == 'D':
                index = int(bit_label[1:])
            elif bit_label[0] == 'L':
                index = self.detector_number + int(bit_label[1:])
            else:
                raise ValueError(f"Invalid bit label '{bit_label}': must start with 'D' or 'L'.")
            
            # 直接修改bytearray中的字节
            binary_bytes[index] = 49 if binary_bytes[index] == 48 else 48  # ASCII码中48='0', 49='1'
        
        # 直接返回bytearray转换的字符串
        return binary_bytes.decode('ascii')

    def contract_hyperedge(
        self,
        prob_dist: Dict[str, Union[np.float32, np.float64, np.float128]],
        contractable_hyperedges_weights_dict: Dict[str, Union[np.float32, np.float64, np.float128]],
        contracted_hyperedge: str,
        str_syndrome: str
    ) -> Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
        """
        Contract a hyperedge and update the probability distribution and hyperedge weights dictionary.

        Args:
            prob_dist (Dict[str, Union[np.float32, np.float64, np.float128]]): Current probability distribution.
            contractable_hyperedges_weights_dict (Dict[str, Union[np.float32, np.float64, np.float128]]): Hyperedge weights dictionary.
            contracted_hyperedge (str): The hyperedge to contract.
            str_syndrome (str): The syndrome of the contracted hyperedge to approximation contraction.

        Returns:
            Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
                - Updated probability distribution.
                - Updated hyperedge weights dictionary.
        """
        # Get the probability of the contracted hyperedge
        contracted_hyperedge_prob = contractable_hyperedges_weights_dict.pop(contracted_hyperedge)
        # non_flip_contracted_hyperedge_prob = (self.const_1 - contracted_hyperedge_prob)
        
        # Create a defaultdict to store the updated probability distribution (default to 0.0)
        updated_prob_dist = {}

        # Iterate over the current probability distribution
        for binary_str, prob in prob_dist.items():
            # Flip the bits for the current hyperedge
            flipped_int = self.flip_bits(binary_str, contracted_hyperedge)
            
            # Calculate the updated probabilities based on the hyperedge contraction
            flipped_prob = prob + contracted_hyperedge_prob
            non_flipped_prob = prob

            # Add probabilities to the new distribution (manually handle key existence)
            if flipped_int in updated_prob_dist:
                if updated_prob_dist[flipped_int] > flipped_prob:
                    updated_prob_dist[flipped_int] = flipped_prob
            else:
                updated_prob_dist[flipped_int] = flipped_prob

            if binary_str in updated_prob_dist:
                if updated_prob_dist[binary_str] > non_flipped_prob:
                    updated_prob_dist[binary_str] = non_flipped_prob
            else:
                updated_prob_dist[binary_str] = non_flipped_prob              

        if self._approximate_position == "hyperedge":
            updated_prob_dist = self.approximate_distribution(updated_prob_dist, str_syndrome)
            
        return updated_prob_dist, contractable_hyperedges_weights_dict

    def _to_numeric_type(self, value):
        # float128，对应C++中的long double
        if self.accuracy == "float128":
            return np.float128(value)
        # float64，对应C++中的double
        elif self.accuracy == "float64":
            return np.float64(value)
        elif self.accuracy == "float32":
            return np.float32(value)
        else:
            raise ValueError(f"Unsupported accuracy type: {self.accuracy}")

    def accuracy_type(self):
        if self.accuracy == "float128":
            return np.float128
        elif self.accuracy == "float64":
            return np.float64
        elif self.accuracy == "float32":
            return np.float32
        else:
            raise ValueError(f"Unsupported accuracy type: {self.accuracy}")
    
    def get_task_initial_input(self) -> Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
        """
        Get the initial input for the task, including the probability distribution
        and the hyperedge weights dictionary.

        This method initializes the probability distribution with a single deterministic entry
        (all False for both detectors and logical qubits). It also prepares the hyperedge weights dictionary.

        Returns:
            Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
                - Initial probability distribution.
                - Dictionary of contractable hyperedge weights.
        """
        init_prob_dist: Dict[str, Union[np.float32, np.float64, np.float128]] = {}
        contractable_hyperedges_weights_dict: Dict[str, Union[np.float32, np.float64, np.float128]] = {}
        
        # Initialize the key with all False values
        # init_key = tuple([False] * self.detector_number + [False] * self.logical_number)
        init_key: str = "0" * self.total_length
        init_prob_dist[init_key] = self.const_1
        
        # Generate hyperedge weights dictionary, 这是因为在更新过程中，我们需要修改contractable_hyperedges_weights_dict，所以在每次解码的情况下，我们copy一份。
        contractable_hyperedges_weights_dict = self.contractable_hyperedges_weights_dict.copy()
        
        return init_prob_dist, contractable_hyperedges_weights_dict
        

    def mld_contraction_no_slicing(self, syndrome: np.ndarray[bool]) -> Dict[str, Union[np.float32, np.float64, np.float128]]:
        """
        Perform MLD contraction without slicing.

        This method computes the MLD contraction based on the provided order, without applying slicing.
        The contraction is performed serially according to the specified order.

        Args:
            syndrome (np.ndarray[bool]): The syndrome to be processed.
            order (List[str], optional): The order in which the contraction should be performed.
                                        If not provided, the default order (self.order) will be used.

        Returns:
            Dict[str, Union[np.float32, np.float64, np.float128]]: The resulting probability distribution after the contraction.
        """
        # Validate input syndrome type
        if not isinstance(syndrome, np.ndarray) or syndrome.ndim != 1 or not np.issubdtype(syndrome.dtype, np.bool_):
            raise TypeError("syndrome must be a 1D np.ndarray of boolean values.")

        logger.debug(f"Calling {self.mld_contraction_no_slicing.__name__}")

        # Convert the syndrome to a set of detector indices that are True
        true_indices = np.where(syndrome)[0]
        self.flipped_detector_set = set([f"D{idx}" for idx in true_indices])
        
        if len(self.flipped_detector_set) == 0:
            init_prob_dist = {}
            init_key: str = "0" * self.total_length
            init_prob_dist[init_key] = self.const_1
            return init_prob_dist
        else:
            # Get the initial probability distribution and contractable hyperedges weights
            init_prob_dist, init_contractable_hyperedges_weights_dict = self.get_task_initial_input()
            
            sub_contractable_hyperedges_weights_dict, self.order = self._priority_sub_hypergraph_contraction_strategy()

            # 构建缓存
            self.hyperedge_cache, self.relevant_hyperedge_cache = build_hyperedge_contraction_caches(sub_contractable_hyperedges_weights_dict, self.order)
            # self.hyperedge_cache = hyperedge_cache
            
            start_time = time.time()
            # Perform the MLD contraction on the syndrome
            prob_dist, _ = self.single_node_online_mld_contraction(
                syndrome = syndrome,
                init_prob_dist = init_prob_dist,
                init_contractable_hyperedges_weights = sub_contractable_hyperedges_weights_dict
            )
            
            self._execution_contraction_time = time.time() - start_time
            if len(prob_dist) == 0:
                prob_dist = {}
                init_key: str = "0" * self.total_length
                prob_dist[init_key] = self.const_1
            return prob_dist
    
    def single_node_online_mld_contraction(
            self,
            syndrome: np.ndarray[bool],
            init_prob_dist: Dict[str, Union[np.float32, np.float64, np.float128]],
            init_contractable_hyperedges_weights: Dict[str, Union[np.float32, np.float64, np.float128]]
    ) -> Dict[str, Union[np.float32, np.float64, np.float128]]:
        """
        Perform single-node online MLD contraction.

        This method computes the probability distribution for a given syndrome and updates it
        based on the initial probability distribution and hyperedge weights.

        Args:
            syndrome (np.ndarray[bool]): The measurement values of the syndrome bits.
                It should be a binary numpy array representing the syndrome.
            init_prob_dist (Dict[str, Union[np.float32, np.float64, np.float128]]): Initial probability distribution.
                If there is no slicing, it defaults to {'0...0': 1}. Otherwise, it may be transformed.
            init_contractable_hyperedges_weights (Dict[str, Union[np.float32, np.float64, np.float128]]): The weights associated with each hyperedge to contract.

        Returns:
            Dict[str, Union[np.float32, np.float64, np.float128]]: The updated probability distribution for the given syndrome.
                The size of the distribution depends on the number of logical qubits.
        """
        logger.debug("Starting contraction for syndrome: %s", syndrome)
        
        # Initialize the probability distribution and hyperedge weights
        prob_dist = init_prob_dist.copy()  # Avoid modifying the input directly
        contractable_hyperedges_weights = init_contractable_hyperedges_weights.copy()
        
        str_syndrome = binary_to_str(syndrome)
        
        contraction_step_num = len(self.order)
        # Iterate over each detector in the contraction order
        for contraction_step in range(contraction_step_num):
            # print("contraction_step: ", contraction_step)
            # Get the current contract detector and its index
            contract_detector = self.order[contraction_step]
            contract_detector_index = int(contract_detector[1:])  # Extract the index from the detector name
            observed_detector_syndrome = str(int(syndrome[contract_detector_index])) # Convert boolean to str

            # logger.debug(f"Processing detector {contract_detector} (index {contract_detector_index}) with observed syndrome {observed_detector_syndrome}")
            # Contract all hyperedges connected to the current detector
            relevant_hyperedges = self.relevant_hyperedge_cache.get(contract_detector, set())
            
            for hyperedge in relevant_hyperedges:
                # Perform contraction and update the probability distribution
                prob_dist, contractable_hyperedges_weights = self.contract_hyperedge(
                    prob_dist, contractable_hyperedges_weights, hyperedge, str_syndrome
                )
            # Filter out candidates where the syndrome bit does not match the observed syndrome
            prob_dist = {
                candidate_syndrome: prob for candidate_syndrome, prob in prob_dist.items()
                if check_str_value(candidate_syndrome, contract_detector_index, observed_detector_syndrome)
            }
            
            if self._approximate_position == "node":
                # logger.info(f"Before approximate_distribution, prob_dist's shape: {len(prob_dist)}")
                prob_dist = self.approximate_distribution(prob_dist, str_syndrome)
                
            # Log current state after processing the detector
            # logger.debug(f"Contraction step {contraction_step}, contract_detector: {contract_detector}")
            # logger.debug(f"Updated prob_dist: {prob_dist}")
            # logger.debug(f"Remaining hyperedges: {list(contractable_hyperedges_weights.keys())}")
        return prob_dist, contractable_hyperedges_weights

    # @classmethod
    def validate_logical_operator(self, prob_dist: Dict[str, Union[np.float32, np.float64, np.float128]]) -> Tuple[np.ndarray[bool], Union[np.float32, np.float64, np.float128]]:
        """
        Validate if many logical error (logical operator) occurs by comparing the values of probability distributions.

        This method checks if a logical error has occurred by comparing the resulting probability distribution
        with the logical operators provided ('X' or 'Z'). It returns a logical error indicator and the probability
        of correct error correction.

        Args:
            prob_dist (Dict[str, Union[np.float32, np.float64, np.float128]]): The resulting probability distribution after contraction.
            logical_operators (str): A string indicating the logical operator to check against the distribution.
                Can be 'X' or 'Z' (default is 'X').

        Returns:
            Tuple[np.ndarray[bool], Union[np.float32, np.float64, np.float128]]:
                - A numpy array indicating whether a logical error is detected (True if logical error occurs).
                - The probability of correct error correction for the current syndrome.
        """
        keys = list(prob_dist.keys())
        
        # Ensure the dictionary contains exactly two keys, with the last element being True and False
        if len(keys) == 1:
            last_logical_bits = keys[0][-self.logical_number:]
            logical_error_detected = np.array([bit == '1' for bit in last_logical_bits], dtype=bool)
            return np.array([logical_error_detected], dtype=bool), 1.0
        elif len(keys) == 0:
            # 如果没有keys，证明，经过近似之后，该syndrome发生的概率为0，因此默认返回不出现逻辑错误率，纠错正确概率为0.5
            return np.array([[False]*self.logical_number], dtype=bool), 0.5
        elif len(keys) >= 2:
            # 寻找最可能的逻辑错误情况
            # 找到最大value对应的key
            max_key = max(prob_dist, key=prob_dist.get)
            # 获取后logical_number个字符
            last_logical_bits = max_key[-self.logical_number:]
            
            # 将字符串转换为布尔数组
            # 0为False，1为True
            logical_error_detected = np.array([bit == '1' for bit in last_logical_bits], dtype=bool)
            
            # 计算正确纠错的概率（不太准确，因为进行了近似操作）
            # max_prob = prob_dist[max_key]
            # total_prob = sum(prob_dist.values())
            # prob_correct_correction = max_prob / total_prob
        
            # Return the logical error indicator and the probability of correct error correction
            return np.array([logical_error_detected], dtype=bool), None