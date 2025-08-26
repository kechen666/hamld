from typing import List, Dict, Tuple, Set, Optional, Generator
import numpy as np
import stim
import logging
import time
import multiprocessing
from enum import Enum

from hamld.logging_config import setup_logger
from hamld.contraction_strategy.contraction_strategy import ContractionStrategy

from hamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
from hamld.contraction_executor import (
    ContractionExecutor,
    ContractionExecutorInt,
    PyContractionExecutorCpp,
    ApproximateContractionExecutor,
    ApproximatePyContractionExecutorCpp,
    # ContractionExecutorCpp,
    # ApproximateContractionExecutorCpp,
    binary_to_str,
    str_to_binary_bitwise,
    int_to_binary_bitwise,
    build_hyperedge_contraction_caches,
    ApproximateContractionExecutorQldpc,
    HierarchicalApproximateContractionExecutorQldpc,
    PriorityApproximateContractionExecutorQldpc,
    NewPriorityApproximateContractionExecutorQldpc,
    LogApproximateContractionExecutorQldpc,
    BiasedApproximateContractionExecutorQldpc,
    SyndromePriorityApproximateContractionExecutor,
    SyndromePriorityApproximateContractionExecutorLog)

from hamld.config import (
    DEFAULT_ORDER_METHOD,
    DEFAULT_SLICE_METHOD,
    DEFAULT_SLICE_COUNT,
    DEFAULT_SLICE_WIDTH,
    DEFAULT_ORDER_UPDATE,
    DEFAULT_EXECUTION_METHOD,
)

# import itertools
import time

from hamld.hamld_utility import (
    preprocess_parallel_rounds_syndromes,
    parallel_rounds_decode_task,
    compute_logical_prob_dist,
    decode_merge_task,
)

# from tqdm import tqdm
# from tqdm.notebook import tqdm

# Configure logging at the module level
logger = setup_logger("src/hamld", log_level=logging.WARNING)

class OrderMethod(Enum):
    MLD = "mld"
    GREEDY = "greedy"

class SliceMethod(Enum):
    PARALLELISM = "parallelism"
    MEMORY = "memory"
    NO_SLICE = "no_slice"

def decode_task(syndrome: np.ndarray[bool], decoder, output_prob=False, shared_dict=None, timing=False):
    """Decoding task for a single syndrome, used in parallel decoding."""
    import time
    start_time = time.perf_counter() if timing else None

    result = decoder.decode(syndrome)
    pred, prob_dist, _ = (result[0], result[1], result[2]) if output_prob else (result[0], None, None)

    decode_time = time.perf_counter() - start_time if timing else None
    return pred, prob_dist, decode_time

def approx_decode_task_cpp(syndrome: np.ndarray[bool], contractor_init_args, output_prob=False, shared_dict=None, timing=False):
    """Decoding task for a single syndrome, used in parallel decoding."""
    start_time = time.perf_counter() if timing else None

    contractor = ApproximateContractionExecutorCpp(*contractor_init_args)
    prob_dist = contractor.mld_contraction_no_slicing(syndrome)
    prediction, _ = contractor.validate_logical_operator(prob_dist)

    pred = prediction if not output_prob else prediction
    prob_out = prob_dist if output_prob else None

    decode_time = time.perf_counter() - start_time if timing else None
    return pred, prob_out, decode_time

def decode_task_cpp(syndrome: np.ndarray[bool], contractor_init_args, output_prob=False, shared_dict=None, timing=False):
    """Decoding task for a single syndrome, used in parallel decoding."""
    start_time = time.perf_counter() if timing else None

    contractor = ContractionExecutorCpp(*contractor_init_args)
    prob_dist = contractor.mld_contraction_no_slicing(syndrome)
    prediction, _ = contractor.validate_logical_operator(prob_dist)

    pred = prediction if not output_prob else prediction
    prob_out = prob_dist if output_prob else None

    decode_time = time.perf_counter() - start_time if timing else None
    return pred, prob_out, decode_time


def batch_generator(generator: Generator[np.ndarray[bool], None, None], batch_size: int) -> Generator[np.ndarray[bool], None, None]:
    """Yields batches of syndromes from the generator."""
    batch = []
    for syndrome in generator:
        batch.append(syndrome)
        if len(batch) == batch_size:
            yield batch
            batch = []  # Reset the batch
    if batch:
        yield batch  # Yield remaining syndromes if not a full batch

class HAMLD:
    def __init__(
        self,
        detector_error_model: stim.DetectorErrorModel,
        order_method: OrderMethod = OrderMethod.MLD,
        slice_method: SliceMethod = SliceMethod.PARALLELISM,
        slice_count: int = DEFAULT_SLICE_COUNT,
        slice_width: int = DEFAULT_SLICE_WIDTH,
        order_update: bool = DEFAULT_ORDER_UPDATE,
        execution_method: str = DEFAULT_EXECUTION_METHOD,
        is_look_up_table: bool = False,
        use_decimal: bool = False,
        use_approx: bool = False,
        approximatestrategy: str = "node_topk",
        approximate_param: Optional[int|float] = None,
        contraction_code: str = "normal",
        accuracy: str = "float64",
        priority: int = 1,
        priority_topk: int = 10,
        p_00: float = 1.0,
        p_11: float = 1.0,
        r: int = 6,
        m: int = 1,
        n: int = 72,
        gamma : int =23,
    ):
        """
        Initializes the HAMLD decoder with the given configuration.
        """
        # self.detector_error_model = detector_error_model
        self.num_detectors = detector_error_model.num_detectors
        self.num_observables = detector_error_model.num_observables
        self.total_length = self.num_detectors + self.num_observables
        # Convert order_method and slice_method from string to Enum if necessary
        if isinstance(order_method, str):
            self.order_method = OrderMethod[order_method.upper()]
        else:
            self.order_method = order_method

        if isinstance(slice_method, str):
            self.slice_method = SliceMethod[slice_method.upper()]
        else:
            self.slice_method = slice_method

        self.slice_count = slice_count
        self.slice_width = slice_width
        self.order_update = order_update
        self.execution_method = execution_method
        self.is_look_up_table = is_look_up_table
        self.use_decimal = use_decimal
        # TODO: 将use_approx和contraction_code合并为一个参数，可能部分示例调用会出错。
        self.use_approx = use_approx
        self.contraction_code = contraction_code
        self.accuracy = accuracy
        
        if self.contraction_code == "parallel-rounds-qlpdc":
            # 并行轮次的参数
            self.r = r
            self.m = m
            self.n = n
        else:
            self.r = None
            self.m = None
            self.n = None

        # Validate methods and set slicing
        self._validate_methods()

        # Initialize contraction strategy
        self.contraction_strategy = ContractionStrategy()
        self.contraction_strategy.get_contraction_strategy_from_error_model(
            detector_error_model=detector_error_model,
            order_strategy=self.order_method.value,
            perform_slicing=self.perform_slicing,
            slice_strategy=None,
            sliced_hyperedge_count=self.slice_count,
            update_order=self.order_update
        )

        # Lookup table for caching decoding results
        # Lookup table，将key使用int表示，从而加速查找
        self.lookup_table = {}
        self.order = self.contraction_strategy.order
        self.approximatestrategy = approximatestrategy
        self.approximate_param = approximate_param
        
        # self.total_decode_time
        self.total_decode_time = 0
        
        # Execute contraction
        if self.use_approx:
            if self.contraction_code == "normal":
                self.contractor = ApproximateContractionExecutor(
                    detector_error_model=detector_error_model,
                    contraction_strategy=self.contraction_strategy,
                    use_decimal = self.use_decimal,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param
                )
            # If you want to use C++ version, please uncomment the following code
            # elif self.contraction_code == "cpp":
            #     # TODO: 待测试 - 测试 C++ 版本的近似收缩执行器
            #     # 需要验证以下内容：
            #     # 1. 收缩结果的正确性，与 Python 版本对比
            #     # 2. 性能测试，包括执行时间和内存使用
            #     # 3. 不同精度（float32/float64）下的数值稳定性
            #     # 4. 大规模数据集的扩展性
            #     hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
            #     original_dict = hypergraph.get_hyperedges_weights_dict()
            #     contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                
                
            #     num_detectors : int = detector_error_model.num_detectors
            #     num_observables : int = detector_error_model.num_observables

            #     order: list[str] = self.contraction_strategy.order
            #     sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
            #     accuracy: str = "float64"

            #     self.contractor_init_args = (
            #         num_detectors,
            #         num_observables,
            #         order,
            #         sliced_hyperedges,
            #         contractable_hyperedges_weights_dict,
            #         accuracy,
            #         approximatestrategy,
            #         approximate_param
            #     )

            #     # from pympler import asizeof
            #     # memory_usage = asizeof.asizeof(contractable_hyperedges_weights_dict) 
            #     # print(f"memory_usage: {memory_usage}")
                
            #     self.contractor = ApproximateContractionExecutorCpp(
            #         num_detectors,
            #         num_observables,
            #         order,
            #         sliced_hyperedges,
            #         contractable_hyperedges_weights_dict,
            #         accuracy,
            #         approximatestrategy,
            #         approximate_param
            #     )
            elif self.contraction_code == "cpp-py":
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables

                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = "float64"

                # from pympler import asizeof
                # memory_usage = asizeof.asizeof(contractable_hyperedges_weights_dict) 
                # print(f"memory_usage: {memory_usage}")
                
                self.contractor = ApproximatePyContractionExecutorCpp(
                    num_detectors,
                    num_observables,
                    order,
                    sliced_hyperedges,
                    contractable_hyperedges_weights_dict,
                    accuracy,
                    approximatestrategy,
                    approximate_param
                )
            elif self.contraction_code == "qldpc":
                # TODO: 待测试 - 测试 QL-DPC 版本的近似收缩执行器
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}

                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables

                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy

                hyperedge_cache, relevant_hyperedge_cache = build_hyperedge_contraction_caches(contractable_hyperedges_weights_dict, order)
                self.contractor = ApproximateContractionExecutorQldpc(
                    num_detectors,
                    num_observables,
                    order,
                    sliced_hyperedges,
                    contractable_hyperedges_weights_dict,
                    accuracy,
                    approximatestrategy,
                    approximate_param,
                    hyperedge_cache,
                    relevant_hyperedge_cache,
                )
            elif self.contraction_code == "qldpc-hierarchical-first":
                # TODO: 待测试 - 测试 QL-DPC 分层版本的近似收缩执行器
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                # hyperedge_cache, relevant_hyperedge_cache = build_hyperedge_contraction_caches(contractable_hyperedges_weights_dict, order)
                self.contractor = HierarchicalApproximateContractionExecutorQldpc(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    contract_no_flipped_detector=False,
                    change_approximate_param=False
                )
            elif self.contraction_code == "qldpc-priority":
                # 依旧目前执行是非近似，但是本质上是对超图进行了近似，在超图收缩中没有近似，依旧放置在近似部分中。
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                
                # 默认不使用近似策略，因为我们只精准的收缩topk的边。
                # approximatestrategy = "no_no"
                
                self.contractor = PriorityApproximateContractionExecutorQldpc(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    priority=priority,
                    priority_topk=priority_topk
                )
            elif self.contraction_code == "qldpc-new-priority":
                # 依旧目前执行是非近似，但是本质上是对超图进行了近似，在超图收缩中没有近似，依旧放置在近似部分中。
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                
                # 默认不使用近似策略，因为我们只精准的收缩topk的边。
                # approximatestrategy = "no_no"
                
                self.contractor = NewPriorityApproximateContractionExecutorQldpc(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    priority=priority,
                    priority_topk=priority_topk
                )
            elif self.contraction_code == "syndrome-priority":
                # 依旧目前执行是非近似，但是本质上是对超图进行了近似，在超图收缩中没有近似，依旧放置在近似部分中。
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                
                self.contractor = SyndromePriorityApproximateContractionExecutor(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    priority=priority,
                    priority_topk=priority_topk,
                    gamma = gamma
                )
            elif self.contraction_code == "syndrome-priority-log":
                # 依旧目前执行是非近似，但是本质上是对超图进行了近似，在超图收缩中没有近似，依旧放置在近似部分中。
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                
                self.contractor = SyndromePriorityApproximateContractionExecutorLog(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    priority=priority,
                    priority_topk=priority_topk,
                    gamma = gamma
                )
            elif self.contraction_code == "qldpc-log":
                 # 依旧目前执行是非近似，但是本质上是对超图进行了近似，在超图收缩中没有近似，依旧放置在近似部分中。
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                
                # 默认不使用近似策略，因为我们只精准的收缩topk的边。
                # approximatestrategy = "no_no"。
                
                # 进行了log处理，增加了区分度。
                self.contractor = LogApproximateContractionExecutorQldpc(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    priority=priority,
                    priority_topk=priority_topk
                )
            elif self.contraction_code == "qldpc-biased":
                 # 依旧目前执行是非近似，但是本质上是对超图进行了近似，在超图收缩中没有近似，依旧放置在近似部分中。
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                num_detectors : int = detector_error_model.num_detectors
                num_observables : int = detector_error_model.num_observables
                order: list[str] = self.contraction_strategy.order
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                accuracy: str = self.accuracy
                
                # 默认不使用近似策略，因为我们只精准的收缩topk的边。
                # approximatestrategy = "no_no"。
                
                # 进行了log处理，增加了区分度。
                self.contractor = BiasedApproximateContractionExecutorQldpc(
                    detector_number = num_detectors,
                    logical_number = num_observables,
                    order = order,
                    sliced_hyperedges = sliced_hyperedges,
                    contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                    accuracy = accuracy,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param,
                    hyperedge_cache = None,
                    relevant_hyperedge_cache = None,
                    hypergraph = hypergraph,
                    priority=priority,
                    priority_topk=priority_topk,
                    p_00=p_00,
                    p_11=p_11
                )
            elif self.contraction_code == "parallel-rounds-qlpdc":
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                self.partitioned_hypergraphs, self.detector_maps, logical_hyperedges = hypergraph.partition_hypergraph(r=self.r, m=self.m, n = self.n, have_connect_hyperedges = True)
                # In here, self.contractor is a List[Contractor].
                self.logical_prob_dist = compute_logical_prob_dist(hypergraph, logical_hyperedges, hypergraph.logical_observable_number)

                self.contractor  = []
                for layer_i in range(self.m):
                    par_hypergraph = self.partitioned_hypergraphs[layer_i]
                    original_dict = par_hypergraph.get_hyperedges_weights_dict()
                    contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                    num_detectors : int = par_hypergraph.get_nodes_number(have_logical_observable = False)
                    num_observables : int = par_hypergraph.logical_observable_number
                    # The order is find In-contractor.
                    # For different syndrome, have different detectors, so have different order.
                    order: list[str] = None
                    sliced_hyperedges: list[str] = []
                    accuracy: str = "float64"
                    approximatestrategy = "hyperedge_topk"

                    contractor = NewPriorityApproximateContractionExecutorQldpc(
                        detector_number = num_detectors,
                        logical_number = num_observables,
                        order = order,
                        sliced_hyperedges = sliced_hyperedges,
                        contractable_hyperedges_weights_dict = contractable_hyperedges_weights_dict,
                        accuracy = accuracy,
                        approximatestrategy = approximatestrategy,
                        approximate_param = approximate_param,
                        hyperedge_cache = None,
                        relevant_hyperedge_cache = None,
                        hypergraph = par_hypergraph,
                        priority=priority,
                        priority_topk=priority_topk
                    )
                    self.contractor.append(contractor)
            else:
                raise ValueError(f"Invalid approx_contraction_code: {self.contraction_code}. Must be implement.")
        else:
            if self.contraction_code == "normal":
                self.contractor = ContractionExecutor(
                    detector_error_model=detector_error_model,
                    contraction_strategy=self.contraction_strategy,
                    use_decimal = self.use_decimal
                )
            elif self.contraction_code == "int":
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict = {",".join(key): float(value) for key, value in original_dict.items()}

                # from pympler import asizeof
                # memory_usage = asizeof.asizeof(contractable_hyperedges_weights_dict) 
                # print(f"memory_usage: {memory_usage}")

                self.contractor = ContractionExecutorInt(
                    detector_number=detector_error_model.num_detectors,
                    logical_number= detector_error_model.num_observables,
                    order = self.contraction_strategy.order,
                    sliced_hyperedges = self.contraction_strategy.sliced_hyperedges,
                    contractable_hyperedges_weights_dict= contractable_hyperedges_weights_dict,
                    accuracy = "float64"
                )
            elif self.contraction_code == "cpp-py":
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                original_dict = hypergraph.get_hyperedges_weights_dict()
                contractable_hyperedges_weights_dict = {",".join(key): float(value) for key, value in original_dict.items()}
                
                # from pympler import asizeof
                # memory_usage = asizeof.asizeof(contractable_hyperedges_weights_dict) 
                # print(f"memory_usage: {memory_usage}")
                
                self.contractor = PyContractionExecutorCpp(
                    detector_number=detector_error_model.num_detectors,
                    logical_number= detector_error_model.num_observables,
                    order = self.contraction_strategy.order,
                    sliced_hyperedges = self.contraction_strategy.sliced_hyperedges,
                    contractable_hyperedges_weights_dict= contractable_hyperedges_weights_dict,
                    accuracy = "float64"
                )
            # if you want to use C++ version, please uncomment the following code
            # elif self.contraction_code == "cpp":
            #     hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
            #     original_dict = hypergraph.get_hyperedges_weights_dict()
            #     contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                
                
            #     num_detectors : int = detector_error_model.num_detectors
            #     num_observables : int = detector_error_model.num_observables

            #     order: list[str] = self.contraction_strategy.order
            #     sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
            #     accuracy: str = "float64"

            #     self.contractor_init_args = (
            #         num_detectors,
            #         num_observables,
            #         order,
            #         sliced_hyperedges,
            #         contractable_hyperedges_weights_dict,
            #         accuracy,
            #     )
                
            #     self.contractor = ContractionExecutorCpp(
            #         num_detectors,
            #         num_observables,
            #         order,
            #         sliced_hyperedges,
            #         contractable_hyperedges_weights_dict,
            #         accuracy
            #     )
            else:
                raise ValueError(f"Invalid contraction_code: {self.contraction_code}. Must be one of ['normal', 'opt', 'cpp', 'cpp-py', 'cpp', 'cupy'].")
        
    def _validate_methods(self):
        """Validates the order and slice methods."""
        if self.order_method not in OrderMethod:
            raise ValueError(f"Invalid order_method: '{self.order_method}'. Must be one of {list(OrderMethod)}.")
        
        if self.slice_method not in SliceMethod:
            raise ValueError(f"Invalid slice_method: '{self.slice_method}'. Must be one of {list(SliceMethod)}.")

        # Set the slicing flag based on the slice method
        self.perform_slicing = self.slice_method in {SliceMethod.PARALLELISM, SliceMethod.MEMORY}

    def _get_from_lookup(self, syndrome: np.ndarray[bool]) -> Optional[Tuple[np.ndarray[bool], float]]:
        """Checks if the result for the given syndrome is already cached in the lookup table."""
        # syndrome_key = tuple(syndrome)
        syndrome_key_int = binary_to_str(syndrome)
        return self.lookup_table.get(syndrome_key_int)

    def _add_to_lookup(self, syndrome: np.ndarray[bool], prediction: np.ndarray[bool], prob_dist: Dict[Tuple[bool], float], prob_correct_correction: float):
        """Adds a decoded result to the lookup table."""
        # syndrome_key = tuple(syndrome)
        syndrome_key_int = binary_to_str(syndrome)
        self.lookup_table[syndrome_key_int] = (prediction, prob_dist, prob_correct_correction)

    def set_look_up_table(self, is_look_up_table: bool = True, look_up_table=None):
        """Enables or disables lookup table caching."""
        self.is_look_up_table = is_look_up_table

        if is_look_up_table:
            # 如果启用了查找表，则检查是否传入了 look_up_table
            if look_up_table is not None:
                # 如果传入了 look_up_table，则赋值给 self.look_up_table
                self.look_up_table = look_up_table
            # 如果没有传入 look_up_table，保持 self.look_up_table 不变

    def decode(self, syndrome: np.ndarray[bool]) -> np.ndarray[bool]:
        """Decodes a single syndrome."""
        if self.is_look_up_table:
            cached_result = self._get_from_lookup(syndrome)
            if cached_result:
                logger.debug(f"Cache hit for syndrome {syndrome}")
                return cached_result[0], cached_result[1], cached_result[2]
        # start = time.time()
        # print("start contractor")
        # print(f"self.perform_slicing:{self.perform_slicing}")
        # self.contractor.mld_contraction_parallel_concurrent(syndrome) if self.perform_slicing else 
        prob_dist = self.contractor.mld_contraction_no_slicing(syndrome)
        # print(f"end contraction: {time.time() - start}")
        
        # Validate and return the prediction
        # print(f"output prob_dist:", prob_dist)
        prediction, prob_correct_correction = self.contractor.validate_logical_operator(prob_dist)
        
        if self.is_look_up_table:
            self._add_to_lookup(syndrome, prediction, prob_dist, prob_correct_correction)
            
        return prediction, prob_dist, prob_correct_correction

    def decode_batch(self, syndromes: np.ndarray[bool], output_prob=False) -> np.ndarray[bool]:
        """Decodes a batch of syndromes."""
        # TODO:
        num_observables = self.num_observables
        num_possible_outcomes = 2 ** num_observables
        predictions = np.empty((syndromes.shape[0], num_observables), dtype=bool)

        if self.use_decimal:
            prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float128) if output_prob else None
        else:
            prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float64) if output_prob else None

        if syndromes.shape[0]>=10:
            per_print_counts = syndromes.shape[0] // 10
        else:
            per_print_counts = 1
        # per_print_counts = syndromes.shape[0] // 10
        for i, syndrome in enumerate(syndromes):
            result = self.decode(syndrome)
            predictions[i] = result[0]
            # TODO: output_prob，是否需要删除？
            if output_prob:
                if "cpp" not in self.contraction_code and "int" not in self.contraction_code:
                    prob_dist = result[1]
                    prob = np.full(num_possible_outcomes, None)
                    for key, prob_value in prob_dist.items():
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                    prob_dists[i] = prob
                elif "cpp" in self.contraction_code:
                    prob_dist = result[1]
                    prob = np.full(num_possible_outcomes, None)
                    for key, prob_value in prob_dist.items():
                        key = str_to_binary_bitwise(key)
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                    prob_dists[i] = prob
                elif "int" == self.contraction_code:
                    prob_dist = result[1]
                    prob = np.full(num_possible_outcomes, None)
                    for key, prob_value in prob_dist.items():
                        key = int_to_binary_bitwise(key, self.total_length)
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                    prob_dists[i] = prob
                else:
                    raise ValueError(f"Invalid contraction_code: {self.contraction_code}. Must be one of ['normal', 'opt', 'cpp', 'cpp-py', 'cpp', 'cupy'].")
                    
            if (i + 1) % per_print_counts == 0:
                logger.debug(f"Processed {i + 1} syndromes out of {syndromes.shape[0]}")
            
        return (predictions, prob_dists) if output_prob else predictions

    def parallel_decode_batch(self, syndromes: np.ndarray[bool], output_prob=False, timing=True) -> np.ndarray[bool]:
        """Performs parallel batch decoding."""
        num_observables = self.num_observables
        num_possible_outcomes = 2 ** num_observables

        predictions = np.empty((syndromes.shape[0], num_observables), dtype=bool)
        if self.use_decimal:
            prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float128) if output_prob else None
        else:
            prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float64) if output_prob else None

        num_workers = multiprocessing.cpu_count() - 60
        per_print_counts = syndromes.shape[0] // 10
        
        # parallel decode batch 暂时不支持look up table，直接利用look up table效果不好。
        self.set_look_up_table(False)
        logger.debug(f"start decode.")
        start_time = time.time()
        
        if self.contraction_code != "cpp":
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.starmap(decode_task, [(syndrome, self, output_prob, None, timing) for syndrome in syndromes])
        elif self.use_approx and self.contraction_code == "cpp":
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.starmap(approx_decode_task_cpp, [(syndrome, self.contractor_init_args, output_prob, None, timing) for syndrome in syndromes])
        elif not self.use_approx and self.contraction_code == "cpp":
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.starmap(decode_task_cpp, [(syndrome, self.contractor_init_args, output_prob, None, timing) for syndrome in syndromes])
        
        end_time = time.time()
        logger.debug(f"finish decode, spend time: {end_time - start_time}")
        
        # 处理解码结果
        decode_times = []  # 用于统计时间
        for i, (pred, prob_dist, decode_time) in enumerate(results):
            predictions[i] = pred
            decode_times.append(decode_time)

            if output_prob:
                prob = np.full(num_possible_outcomes, None)
                if self.contraction_code == "normal":
                    for key, prob_value in prob_dist.items():
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                elif "cpp" in self.contraction_code:
                    for key, prob_value in prob_dist.items():
                        key = str_to_binary_bitwise(key)
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                elif self.contraction_code == "int":
                    for key, prob_value in prob_dist.items():
                        key = int_to_binary_bitwise(key, self.total_length)
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                else:
                    output_prob = False
                    prob_dists = None

                prob_dists[i] = prob

            if (i + 1) % per_print_counts == 0:
                logger.debug(f"Processed {i + 1} syndromes out of {syndromes.shape[0]}")

        # 统计总解码时间
        self.total_decode_time = np.sum(decode_times)
        
        # print("predictions", predictions)
        return (predictions, prob_dists) if output_prob else predictions

    def parallel_rounds_decode_batch(self, syndromes: np.ndarray[bool], m = None) -> np.ndarray[bool]:
        """Performs parallel rounds decoding."""
        num_observables = self.num_observables
        # num_possible_outcomes = 2 ** num_observables

        predictions = np.empty((syndromes.shape[0], num_observables), dtype=bool)

        all_num_workers = multiprocessing.cpu_count() -60

        # 分配rounds和shots的进程数，m表示rounds的分层数
        # rounds_num_workers = min(m, all_num_workers // 4)
        # shots_num_workers = max(1, all_num_workers // rounds_num_workers)
        num_shots = syndromes.shape[0]
        per_print_counts = syndromes.shape[0] // 10

        if m == None:
            m = self.m
        
        # 预处理syndrome：
        parallel_syndromes = preprocess_parallel_rounds_syndromes(syndromes=syndromes, partitioned_hypergraphs=self.partitioned_hypergraphs, m = self.m)
        logger.debug(f"parallel_syndromes len: {len(parallel_syndromes)}, one shape is:{parallel_syndromes[0].shape}")

        logger.debug(f"start decode.")
        start_time = time.time()
        # 初始化进程池                    
        with multiprocessing.Pool(processes=all_num_workers) as pool:
            # 传入一组syndrome，一组收缩器。
            # 修改成，传入一个syndrome和一个对应的收缩器。
            results = pool.starmap(parallel_rounds_decode_task, 
                                   [(parallel_syndromes[round_i][shots_i,:],
                                     self.contractor[round_i]) 
                                    for shots_i in range(num_shots) for round_i in range(m)])
        
        end_time = time.time()
        
        logger.debug(f"finish decode, spend time: {end_time - start_time}")

        logger.debug(f"start merge.")
        start_time = time.time()
        # 并行后处理概率分布
        
        grouped_results = [results[i:i+m] for i in range(0, len(results), m)]

        # 使用进程池并行合并
        with multiprocessing.Pool(processes=all_num_workers) as pool:
            merged_results = pool.starmap(decode_merge_task, 
                [(group, self.logical_prob_dist, self.num_observables) for group in grouped_results]
            )
        end_time = time.time()
        logger.debug(f"finish merge, spend time: {end_time - start_time}")
        
        start_time = time.time()
        # 填写内容
        for i, pred in enumerate(merged_results):
            predictions[i] = pred
        end_time = time.time()
        logger.debug(f"finish reset, spend time: {end_time - start_time}")
        
        # print("predictions", predictions)
        return predictions