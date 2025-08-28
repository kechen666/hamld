from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import logging
import time
# from decimal import Decimal
from hamld.logging_config import setup_logger
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor  # 使用多进程

# 设置 logging 配置，放在模块级别
logger = setup_logger("contraction_executor/contraction_executor_cpp_py", log_level=logging.WARNING)

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

class PyContractionExecutorCpp:
    def __init__(self, detector_number: int, logical_number: int, order: List[str], sliced_hyperedges: List[str], contractable_hyperedges_weights_dict: Dict[str, Union[np.float32, np.float64, np.float128]],
                 accuracy: str = "float64", contract_logical_hyperedges: bool = True):
        """
        Initialize the ContractionExecutor with the given detector error model and contraction strategy.

        Args:
            detector_number (int): The number of detector bits in the error model.
            logical_number (int): The number of logical qubits in the error model.
            order (List[str]): The contraction order for the detectors.
            sliced_hyperedges (List[str]): The hyperedges to be sliced for
                parallel task execution.
            contractable_hyperedges_weights_dict (Dict[str, Union[np.float32, np.float64, np.float128]]): The weights associated with each hyperedge to contract.
            accuracy (str): The numerical precision to use for calculations. Can be 'float32', 'float64', or 'float128'.
        """
        self.detector_number = detector_number
        self.logical_number = logical_number
        self.total_length = detector_number + logical_number
        self.order = order
        self.sliced_hyperedges = sliced_hyperedges
        self.accuracy = accuracy
        self.contract_logical_hyperedges = contract_logical_hyperedges
        
        self._execution_contraction_time: float = 0
        self._execution_max_distribution_size: int = 0

        self.contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict
        
        self.const_1 = self._to_numeric_type(1)
        
        self.hyperedge_cache = {}  # hyperedge -> list(parts)
        self.relevant_hyperedge_cache = {}  # detector -> set(hyperedges)
        
        self._build_hyperedge_caches()
        # print()
        
    def _build_hyperedge_caches(self):
        """优化后的缓存构建方法"""
        self.hyperedge_cache = {}  # hyperedge -> list(parts)
        self.relevant_hyperedge_cache = {}  # detector -> set(hyperedges)
        
        # 使用集合提高查找效率
        remaining_hyperedges = set(self.contractable_hyperedges_weights_dict.keys())
        
        for detector in self.order:
            relevant_set = set()
            hyperedges_to_remove = set()
            
            for hyperedge in remaining_hyperedges:
                # 缓存分割结果
                if hyperedge not in self.hyperedge_cache:
                    self.hyperedge_cache[hyperedge] = hyperedge.split(',')
                
                # 检查是否包含当前detector
                if detector in self.hyperedge_cache[hyperedge]:
                    relevant_set.add(hyperedge)
                    hyperedges_to_remove.add(hyperedge)
            
            self.relevant_hyperedge_cache[detector] = relevant_set
            remaining_hyperedges -= hyperedges_to_remove  # 批量移除

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
        contracted_hyperedge: str
    ) -> Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
        """
        Contract a hyperedge and update the probability distribution and hyperedge weights dictionary.

        Args:
            prob_dist (Dict[str, Union[np.float32, np.float64, np.float128]]): Current probability distribution.
            contractable_hyperedges_weights_dict (Dict[str, Union[np.float32, np.float64, np.float128]]): Hyperedge weights dictionary.
            contracted_hyperedge (str): The hyperedge to contract.

        Returns:
            Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
                - Updated probability distribution.
                - Updated hyperedge weights dictionary.
        """
        # Get the probability of the contracted hyperedge
        contracted_hyperedge_prob = contractable_hyperedges_weights_dict.pop(contracted_hyperedge)
        non_flip_contracted_hyperedge_prob = (self.const_1 - contracted_hyperedge_prob)
        
        # Create a defaultdict to store the updated probability distribution (default to 0.0)
        updated_prob_dist = {}

        # Iterate over the current probability distribution
        for binary_str, prob in prob_dist.items():
            # Flip the bits for the current hyperedge
            flipped_int = self.flip_bits(binary_str, contracted_hyperedge)
            
            # Calculate the updated probabilities based on the hyperedge contraction
            flipped_prob = prob * contracted_hyperedge_prob
            non_flipped_prob = prob * non_flip_contracted_hyperedge_prob

            # Add probabilities to the new distribution (manually handle key existence)
            if flipped_int in updated_prob_dist:
                updated_prob_dist[flipped_int] += flipped_prob
            else:
                updated_prob_dist[flipped_int] = flipped_prob

            if binary_str in updated_prob_dist:
                updated_prob_dist[binary_str] += non_flipped_prob
            else:
                updated_prob_dist[binary_str] = non_flipped_prob

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

    def get_parallel_task_initial_input(self) -> Tuple[Dict[str, Union[np.float32, np.float64, np.float128]],
                                                       Dict[str, Union[np.float32, np.float64, np.float128]]]:
        """
        Generate the initial input for parallel task execution.

        This method computes the initial probability distribution and the hyperedge weights dictionary
        after slicing, preparing them for parallel task execution.

        Returns:
            Tuple[Dict[str, Union[np.float32, np.float64, np.float128]], Dict[str, Union[np.float32, np.float64, np.float128]]]:
                - The parallelizable initial probability distribution.
                - The dictionary of contractable hyperedge weights.
        """
        sliced_hyperedges: List[str] = self.sliced_hyperedges
        parallelizable_init_prob_dist, contractable_hyperedges_weights_dict = self.get_task_initial_input()

        # Process each sliced hyperedge to update the probability distribution and hyperedge weights
        for hyperedge in sliced_hyperedges:
            parallelizable_init_prob_dist, contractable_hyperedges_weights_dict = self.contract_hyperedge(
                parallelizable_init_prob_dist, contractable_hyperedges_weights_dict, hyperedge
            )

        return parallelizable_init_prob_dist, contractable_hyperedges_weights_dict

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

        # Get the initial probability distribution and contractable hyperedges weights
        init_prob_dist, init_contractable_hyperedges_weights_dict = self.get_task_initial_input()
        
        start_time = time.time()
        # Perform the MLD contraction on the syndrome
        prob_dist, _ = self.single_node_online_mld_contraction(
            syndrome=syndrome,
            init_prob_dist=init_prob_dist,
            init_contractable_hyperedges_weights=init_contractable_hyperedges_weights_dict
        )
        self._execution_contraction_time = time.time() - start_time
        return prob_dist

    def mld_contraction_serial(self, syndrome: np.ndarray[bool]) -> Dict[str, Union[np.float32, np.float64, np.float128]]:
        """
        Perform serial MLD (Maximum Likelihood Decoding) contraction for a single syndrome.

        This method applies MLD contraction serially to a single syndrome, using the initial probability distribution
        obtained after slicing. No parallelization is involved in this process.

        Args:
            syndrome (np.ndarray[bool]): A 1D NumPy array representing the syndrome, where each element is a boolean value.

        Returns:
            Dict[str, Union[np.float32, np.float64, np.float128]]: A dictionary representing the probability distribution after contraction.
                                        Keys are tuples of boolean values, and values are the corresponding probabilities.

        Raises:
            TypeError: If the `syndrome` is not a 1D NumPy array of boolean values.
        """
        # Validate input syndrome type
        if not isinstance(syndrome, np.ndarray) or syndrome.ndim != 1 or not np.issubdtype(syndrome.dtype, np.bool_):
            raise TypeError("syndrome must be a 1D np.ndarray of boolean values.")

        logger.debug(f"Calling {self.mld_contraction_serial.__name__}")
        logger.debug(f"Contraction strategy: No parallelization, strategy: {self.contraction_strategy}")

        # Get the initial probability distribution and contractable hyperedges weights for serial computation
        parallelizable_init_prob_dist, init_contractable_hyperedges_weights_dict = self.get_parallel_task_initial_input()
        
        start_time = time.time()
        # Perform the MLD contraction on the syndrome
        prob_dist, _ = self.single_node_online_mld_contraction(
            syndrome=syndrome,
            init_prob_dist=parallelizable_init_prob_dist,
            init_contractable_hyperedges_weights=init_contractable_hyperedges_weights_dict
        )
        self._execution_contraction_time = time.time() - start_time
        return prob_dist
    
    def mld_contraction_parallel_concurrent(
        self,
        syndrome: np.ndarray[bool],
        max_thread: int = 4
    ) -> Dict[str, Union[np.float32, np.float64, np.float128]]:
        """
        Perform data-parallel MLD contraction.

        This method splits the parallelizable initial probability distribution into
        separate tasks and runs them in parallel using threading.

        Args:
            syndrome (np.ndarray[bool]): A 1D NumPy array representing the syndrome.
            max_thread (int): The maximum number of threads to use for parallelization. Default is 4.

        Returns:
            Dict[str, Union[np.float32, np.float64, np.float128]]: The merged probability distribution after contraction.
        #TODO: 目前采用的可能是多线程的实现, 可能需要进行多进程的实现。同时似乎并行的优化效果不大。
        """
        # Validate input syndrome type
        if not isinstance(syndrome, np.ndarray) or syndrome.ndim != 1 or not np.issubdtype(syndrome.dtype, np.bool_):
            raise TypeError("syndrome must be a 1D np.ndarray of boolean values.")
        
        logger.debug(f"Calling {self.mld_contraction_parallel_concurrent.__name__} with max_thread={max_thread}")
        
        # Get the initial probability distribution and hyperedges weights
        parallelizable_init_prob_dist, init_contractable_hyperedges_weights_dict = self.get_parallel_task_initial_input()

        # Split the tasks based on the parallelizable probability distribution
        tasks = [
            (key, value) for key, value in parallelizable_init_prob_dist.items()
        ]
        
        start_time = time.time()
        # 多进程并行执行
        with ProcessPoolExecutor(max_workers=max_thread) as executor:
            futures = [
                executor.submit(
                    self.single_node_online_mld_contraction,
                    syndrome=syndrome,
                    init_prob_dist={key: value},
                    init_contractable_hyperedges_weights=init_contractable_hyperedges_weights_dict
                )
                for key, value in tasks
            ]
            results = [future.result() for future in futures]

        # Merge the results from all parallel tasks
        # TODO: 是否需要使用原始的dict更加高效？
        merged_prob_dist = defaultdict(self.accuracy_type())

        for result in results:
            for key, value in result[0].items():
                merged_prob_dist[key] += value
        self._execution_contraction_time = time.time() - start_time
        
        return dict(merged_prob_dist)
    
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

        # Iterate over each detector in the contraction order
        for contraction_step in range(self.detector_number):
            # print("contraction_step: ", contraction_step)
            # Get the current contract detector and its index
            contract_detector = self.order[contraction_step]
            contract_detector_index = int(contract_detector[1:])  # Extract the index from the detector name
            observed_detector_syndrome = str(int(syndrome[contract_detector_index])) # Convert boolean to str

            # Contract all hyperedges connected to the current detector
            relevant_hyperedges = self.relevant_hyperedge_cache.get(contract_detector, set())
            
            for hyperedge in relevant_hyperedges:
                # Perform contraction and update the probability distribution
                prob_dist, contractable_hyperedges_weights = self.contract_hyperedge(
                    prob_dist, contractable_hyperedges_weights, hyperedge
                )
            # Filter out candidates where the syndrome bit does not match the observed syndrome
            prob_dist = {
                candidate_syndrome: prob for candidate_syndrome, prob in prob_dist.items()
                if check_str_value(candidate_syndrome, contract_detector_index, observed_detector_syndrome)
            }

            # Log current state after processing the detector
            logger.debug(f"Contraction step {contraction_step}, contract_detector: {contract_detector}")
            logger.debug(f"Updated prob_dist: {prob_dist}")
            logger.debug(f"Remaining hyperedges: {list(contractable_hyperedges_weights.keys())}")

        # 如果contract_logical_hyperedges且contractable_hyperedges_weights非空
        if self.contract_logical_hyperedges and contractable_hyperedges_weights:
            relevant_logical_hyperedges = [hyperedge for hyperedge in contractable_hyperedges_weights.keys()]
            for hyperedge in relevant_logical_hyperedges:
                # Perform contraction and update the probability distribution
                prob_dist, contractable_hyperedges_weights = self.contract_hyperedge(
                    prob_dist, contractable_hyperedges_weights, hyperedge
                )
            logger.debug(f"Contraction only logical hyperedges")
            logger.debug(f"Updated prob_dist: {prob_dist}")
            logger.debug(f"Remaining hyperedges: {list(contractable_hyperedges_weights.keys())}")
        return prob_dist, contractable_hyperedges_weights
    
    @classmethod
    def validate_logical_operator(cls, prob_dist: Dict[str, Union[np.float32, np.float64, np.float128]]) -> Tuple[np.ndarray[bool], Union[np.float32, np.float64, np.float128]]:
        """
        Validate if a logical error (logical operator) occurs by comparing the values of probability distributions.

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
        if len(keys) != 2:
            raise ValueError("prob_dist must contain exactly two keys, with the last element being True and False.")
        
        # Extract p_1 (False) and p_2 (True)
        p_1 = None
        p_2 = None
        for key in keys:
            if key[-1] == '0':  # 检查字符串最后一位是否为'0'
                p_1 = prob_dist[key]
            elif key[-1] == '1':  # 检查字符串最后一位是否为'1'
                p_2 = prob_dist[key]
        
        # Ensure both p_1 and p_2 are successfully extracted
        if p_1 is None or p_2 is None:
            raise ValueError("prob_dist must contain keys with the last element being True and False.")
        
        # Compare p_1 and p_2 to determine if a logical error occurred
        logical_error_detected = p_2 > p_1
        
        # Calculate the probability of correct error correction
        # Here, we assume the probability of correct error correction is the higher of p_1 and p_2
        prob_correct_correction = max(p_1, p_2) / (p_1+p_2)
        
        # Return the logical error indicator and the probability of correct error correction
        return np.array([logical_error_detected], dtype=bool), prob_correct_correction