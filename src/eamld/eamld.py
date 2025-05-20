from typing import List, Dict, Tuple, Set, Optional, Generator
import numpy as np
import stim
import logging
import time
import multiprocessing
from enum import Enum

from eamld.logging_config import setup_logger
from eamld.contraction_strategy.contraction_strategy import ContractionStrategy

from eamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
from eamld.contraction_executor import (
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
    BiasedApproximateContractionExecutorQldpc)

from eamld.config import (
    DEFAULT_ORDER_METHOD,
    DEFAULT_SLICE_METHOD,
    DEFAULT_SLICE_COUNT,
    DEFAULT_SLICE_WIDTH,
    DEFAULT_ORDER_UPDATE,
    DEFAULT_EXECUTION_METHOD,
)
# import itertools

from eamld.eamld_utility import (
    preprocess_parallel_rounds_syndromes,
    parallel_rounds_decode_task,
    compute_logical_prob_dist,
    decode_merge_task,
)

# from tqdm import tqdm
# from tqdm.notebook import tqdm

# Configure logging at the module level
logger = setup_logger("src/eamld", log_level=logging.WARNING)

class OrderMethod(Enum):
    MLD = "mld"
    GREEDY = "greedy"

class SliceMethod(Enum):
    PARALLELISM = "parallelism"
    MEMORY = "memory"
    NO_SLICE = "no_slice"

def decode_task(syndrome: np.ndarray[bool], decoder, output_prob=False, shared_dict=None):
    """Decoding task for a single syndrome, used in parallel decoding."""
    result = decoder.decode(syndrome)
    pred, prob_dist, prob_correct_correction = (result[0], result[1], result[2]) if output_prob else (result[0], None, None)

    return pred, prob_dist

# def approx_decode_task_cpp(syndrome: np.ndarray[bool], contractor_init_args, output_prob=False, shared_dict=None):
#     """Decoding task for a single syndrome, used in parallel decoding."""
#     contractor = ApproximateContractionExecutorCpp(*contractor_init_args)
#     prob_dist = contractor.mld_contraction_no_slicing(syndrome)
#     # Validate and return the prediction
#     prediction, prob_correct_correction = contractor.validate_logical_operator(prob_dist)
#     if output_prob:
#         pred, prob_dist, prob_correct_correction = (prediction, prob_dist, prob_correct_correction)  
#     else:
#         pred, prob_dist, prob_correct_correction = (prediction, None, None)

#     return pred, prob_dist

# def decode_task_cpp(syndrome: np.ndarray[bool], contractor_init_args, output_prob=False, shared_dict=None):
#     """Decoding task for a single syndrome, used in parallel decoding."""
#     contractor = ContractionExecutorCpp(*contractor_init_args)
#     prob_dist = contractor.mld_contraction_no_slicing(syndrome)
#     # Validate and return the prediction
#     prediction, prob_correct_correction = contractor.validate_logical_operator(prob_dist)
#     if output_prob:
#         pred, prob_dist, prob_correct_correction = (prediction, prob_dist, prob_correct_correction)  
#     else:
#         pred, prob_dist, prob_correct_correction = (prediction, None, None)

#     return pred, prob_dist

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

class EAMLD:
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
        approximatestrategy: str = "hyperedge_topk",
        approximate_param: Optional[int|float] = None,
        contraction_code: str = "emld",
        accuracy: str = "float64",
        priority: int = -2,
        priority_topk: int = 150,
        p_00: float = 1.0,
        p_11: float = 1.0,
        r: int = 6,
        m: int = 1,
        n: int = 72,
    ):
        """
        Initializes the EAMLD decoder with the given configuration.
        
        Parameters:
        - detector_error_model: The detector error model to use for decoding.
        - order_method: The method to use for ordering hyperedges. Can be 'mld' or 'greedy'.
        - slice_method: [DEPRECATED] The method to use for slicing hyperedges. Can be 'parallelism', 'memory', or 'no_slice'.
        - slice_count: [DEPRECATED] The number of slices to use.
        - slice_width: [DEPRECATED] The width of each slice.
        - order_update: [DEPRECATED] Whether to update the order after each slice.
        - execution_method: [DEPRECATED] The method to use for executing contractions. Can be 'sequential', 'parallel', or 'parallel_rounds'.
        - is_look_up_table: [DEPRECATED] Whether to use a look-up table for caching decoding results.
        - use_decimal: [DEPRECATED] Whether to use decimal numbers for calculations.
        - use_approx: Whether to use approximate calculations, eamld need to be set to True.
        - approximatestrategy: The strategy to use for approximation. Defaults to "hyperedge_topk".
        - approximate_param: Optional parameter for the approximation strategy.
        - contraction_code: The code specifying the contraction method, including [emld, eamld]. Defaults to "emld".
        - accuracy: The accuracy level for calculations. Defaults to "float64".
        - priority: Priority value for certain contraction strategies. Defaults to -2.
        - priority_topk: Top-k value for priority-based contraction strategies. Defaults to 150.
        - p_00: Probability value. Defaults to 1.0.
        - p_11: Probability value. Defaults to 1.0.
        - r: [DEPRECATED] Parameter for parallel rounds contraction. Defaults to 6.
        - m: [DEPRECATED] Parameter for parallel rounds contraction. Defaults to 1.
        - n: [DEPRECATED] Parameter for parallel rounds contraction. Defaults to 72.
        """
        # self.detector_error_model = detector_error_model
        # Get the number of detectors from the detector error model
        self.num_detectors = detector_error_model.num_detectors
        # Get the number of observables from the detector error model
        self.num_observables = detector_error_model.num_observables
        # Calculate the total length as the sum of detectors and observables
        self.total_length = self.num_detectors + self.num_observables
        # Convert order_method and slice_method from string to Enum if necessary
        if isinstance(order_method, str):
            # Convert string to OrderMethod enum
            self.order_method = OrderMethod[order_method.upper()]
        else:
            self.order_method = order_method

        if isinstance(slice_method, str):
            # Convert string to SliceMethod enum
            self.slice_method = SliceMethod[slice_method.upper()]
        else:
            self.slice_method = slice_method

        # Store the slice count
        self.slice_count = slice_count
        # Store the slice width
        self.slice_width = slice_width
        # Store the flag for order update
        self.order_update = order_update
        # Store the execution method
        self.execution_method = execution_method
        # Store the flag for using a look-up table
        self.is_look_up_table = is_look_up_table
        # Store the flag for using decimal numbers
        self.use_decimal = use_decimal
        # TODO: Merge use_approx and contraction_code into one parameter, may cause errors in some example calls.
        # Store the flag for using approximate calculations
        self.use_approx = use_approx
        # Store the contraction code
        self.contraction_code = contraction_code
        # Store the accuracy level
        self.accuracy = accuracy
        
        if self.contraction_code == "parallel-rounds-qlpdc":
            # 并行轮次的参数
            # Store the parameter r for parallel rounds contraction
            self.r = r
            # Store the parameter m for parallel rounds contraction
            self.m = m
            # Store the parameter n for parallel rounds contraction
            self.n = n
        else:
            # Set parameters to None if not using parallel-rounds-qlpdc
            self.r = None
            self.m = None
            self.n = None

        # Validate methods and set slicing
        self._validate_methods()

        # Initialize contraction strategy
        self.contraction_strategy = ContractionStrategy()
        # Get the contraction strategy from the error model
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
        # Initialize an empty lookup table
        self.lookup_table = {}
        # Store the order from the contraction strategy
        self.order = self.contraction_strategy.order
        # Store the approximation strategy
        self.approximatestrategy = approximatestrategy
        # Store the approximation parameter
        self.approximate_param = approximate_param
        
        # Execute contraction
        if self.use_approx:
            if self.contraction_code == "normal":
                # [DEPRECATED]
                logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'normal' is deprecated. Please set `contraction_code` to 'eamld' instead.")
                
                # Initialize the approximate contraction executor for normal code
                self.contractor = ApproximateContractionExecutor(
                    detector_error_model=detector_error_model,
                    contraction_strategy=self.contraction_strategy,
                    use_decimal = self.use_decimal,
                    approximatestrategy = approximatestrategy,
                    approximate_param = approximate_param
                )
            # elif self.contraction_code == "cpp":
            #     # TODO: C++ with contraction approximation.
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
                # [DEPRECATED] Python with contraction approximation.
                logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'cpp-py' is deprecated. Please set `contraction_code` to 'eamld' instead.")
                
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                
                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables

                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = "float64"
                
                # Initialize the approximate C++-Python contraction executor
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
                # [DEPRECATED]
                logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'cpp-py' is deprecated. Please set `contraction_code` to 'eamld' instead.")
                                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}

                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables

                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = self.accuracy

                # Build hyperedge contraction caches
                hyperedge_cache, relevant_hyperedge_cache = build_hyperedge_contraction_caches(contractable_hyperedges_weights_dict, order)
                # Initialize the QL-DPC approximate contraction executor
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
                # [DEPRECATED]
                # logger.warning(logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'qldpc-hierarchical-first' is deprecated. Please set `contraction_code` to 'eamld' instead."))
                
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables
                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = self.accuracy
                # hyperedge_cache, relevant_hyperedge_cache = build_hyperedge_contraction_caches(contractable_hyperedges_weights_dict, order)
                # Initialize the hierarchical QL-DPC approximate contraction executor
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
                # [DEPRECATED]
                # logger.warning(logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'qldpc-priority' is deprecated. Please set `contraction_code` to 'eamld' instead."))
                
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables
                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = self.accuracy
                
                # Initialize the priority-based QL-DPC approximate contraction executor
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
            elif self.contraction_code == "eamld":
                # The EAMLD method.
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables
                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = self.accuracy
                
                # Initialize the new priority-based QL-DPC approximate contraction executor
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
            elif self.contraction_code == "qldpc-log":
                # [DEPRECATED]
                # logger.warning(logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'qldpc-log' is deprecated. Please set `contraction_code` to 'eamld' instead."))
                
                # Use log process the Probability distribution.
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables
                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = self.accuracy
                
                
                # 进行了log处理，增加了区分度。
                # Initialize the log-based QL-DPC approximate contraction executor
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
                 # Consider the biased noise with EAMLD method.
                 # 
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                # Get the number of detectors
                num_detectors : int = detector_error_model.num_detectors
                # Get the number of observables
                num_observables : int = detector_error_model.num_observables
                # Get the order from the contraction strategy
                order: list[str] = self.contraction_strategy.order
                # Convert sliced hyperedges to comma-separated strings
                sliced_hyperedges: list[str] = [",".join(hyperedge) for hyperedge in self.contraction_strategy.sliced_hyperedges]
                # Set the accuracy level
                accuracy: str = self.accuracy
                
                # 默认不使用近似策略，因为我们只精准的收缩topk的边。
                # approximatestrategy = "no_no"。
                
                # 进行了log处理，增加了区分度。
                # Initialize the biased QL-DPC approximate contraction executor
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
                # [DEPRECATED]
                logger.warning("When `use_approx` is set to `True`, the `contraction_code` value of 'parallel-rounds-qlpdc' is deprecated. Please set `contraction_code` to 'eamld' instead.")
                
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Partition the hypergraph based on parallel rounds parameters
                self.partitioned_hypergraphs, self.detector_maps, logical_hyperedges = hypergraph.partition_hypergraph(r=self.r, m=self.m, n = self.n, have_connect_hyperedges = True)
                # In here, self.contractor is a List[Contractor].
                # Compute the logical probability distribution
                self.logical_prob_dist = compute_logical_prob_dist(hypergraph, logical_hyperedges, hypergraph.logical_observable_number)

                # Initialize an empty list for contractors
                self.contractor  = []
                for layer_i in range(self.m):
                    # Get the partitioned hypergraph for the current layer
                    par_hypergraph = self.partitioned_hypergraphs[layer_i]
                    # Get the dictionary of hyperedge weights
                    original_dict = par_hypergraph.get_hyperedges_weights_dict()
                    # Convert the dictionary keys to comma-separated strings
                    contractable_hyperedges_weights_dict: dict[str, float] = {",".join(key): float(value) for key, value in original_dict.items()}
                    # Get the number of detectors in the partitioned hypergraph
                    num_detectors : int = par_hypergraph.get_nodes_number(have_logical_observable = False)
                    # Get the number of observables in the partitioned hypergraph
                    num_observables : int = par_hypergraph.logical_observable_number
                    # The order is find In-contractor.
                    # For different syndrome, have different detectors, so have different order.
                    # Set the order to None
                    order: list[str] = None
                    # Initialize an empty list for sliced hyperedges
                    sliced_hyperedges: list[str] = []
                    # Set the accuracy level
                    accuracy: str = "float64"
                    # Set the approximation strategy
                    approximatestrategy = "hyperedge_topk"

                    # Initialize the new priority-based QL-DPC approximate contraction executor for the current layer
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
                    # Add the contractor to the list
                    self.contractor.append(contractor)
            else:
                # Raise an error if the approximate contraction code is invalid
                raise ValueError(f"Invalid approx_contraction_code: {self.contraction_code}. Must be implement.")
        else:
            if self.contraction_code == "normal":
                # [DEPRECATED]
                logger.warning("When `use_approx` is set to `False`, the `contraction_code` value of 'normal' is deprecated. Please set `contraction_code` to 'emld' instead.")

                self.contractor = ContractionExecutor(
                    detector_error_model=detector_error_model,
                    contraction_strategy=self.contraction_strategy,
                    use_decimal = self.use_decimal
                )
            elif self.contraction_code == "int":
                # Use int to represent syndrome. 
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict = {",".join(key): float(value) for key, value in original_dict.items()}

                # from pympler import asizeof
                # memory_usage = asizeof.asizeof(contractable_hyperedges_weights_dict) 
                # print(f"memory_usage: {memory_usage}")

                # Initialize the integer-based contraction executor
                self.contractor = ContractionExecutorInt(
                    detector_number=detector_error_model.num_detectors,
                    logical_number= detector_error_model.num_observables,
                    order = self.contraction_strategy.order,
                    sliced_hyperedges = self.contraction_strategy.sliced_hyperedges,
                    contractable_hyperedges_weights_dict= contractable_hyperedges_weights_dict,
                    accuracy = "float64"
                )
            elif self.contraction_code == "emld":
                # EMLD or MLD method.
                # Initialize the normal contraction executor
                
                # Create a hypergraph from the detector error model
                hypergraph = DetectorErrorModelHypergraph(detector_error_model = detector_error_model, have_logical_observable=True)
                # Get the dictionary of hyperedge weights
                original_dict = hypergraph.get_hyperedges_weights_dict()
                # Convert the dictionary keys to comma-separated strings
                contractable_hyperedges_weights_dict = {",".join(key): float(value) for key, value in original_dict.items()}
                
                # Initialize the C++-Python contraction executor
                self.contractor = PyContractionExecutorCpp(
                    detector_number=detector_error_model.num_detectors,
                    logical_number= detector_error_model.num_observables,
                    order = self.contraction_strategy.order,
                    sliced_hyperedges = self.contraction_strategy.sliced_hyperedges,
                    contractable_hyperedges_weights_dict= contractable_hyperedges_weights_dict,
                    accuracy = "float64"
                )
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
                # Raise an error if the contraction code is invalid
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
        """Decodes a single syndrome.
        
        Args:
            syndrome: A numpy array of booleans representing the syndrome to decode
            
        Returns:
            Tuple containing:
            - prediction: The decoded logical operator (numpy array of booleans)
            - prob_dist: Probability distribution over possible outcomes
            - prob_correct_correction: Probability that the correction is correct
        """
        # Check if lookup table is enabled and if syndrome exists in cache
        if self.is_look_up_table:
            cached_result = self._get_from_lookup(syndrome)
            if cached_result:
                logger.debug(f"Cache hit for syndrome {syndrome}")
                return cached_result[0], cached_result[1], cached_result[2]

        # Perform MLD contraction without slicing to get probability distribution
        prob_dist = self.contractor.mld_contraction_no_slicing(syndrome)

        # Validate the logical operator and get prediction with correctness probability
        prediction, prob_correct_correction = self.contractor.validate_logical_operator(prob_dist)
        
        # If lookup table is enabled, cache the new result
        if self.is_look_up_table:
            self._add_to_lookup(syndrome, prediction, prob_dist, prob_correct_correction)
            
        return prediction, prob_dist, prob_correct_correction

    def decode_batch(self, syndromes: np.ndarray[bool], output_prob=False) -> np.ndarray[bool]:
        """Decodes a batch of syndromes."""
        # TODO:
        num_observables = self.num_observables
        num_possible_outcomes = 2 ** num_observables
        predictions = np.empty((syndromes.shape[0], num_observables), dtype=bool)

        # use_decimal is not used in this method.
        # if self.use_decimal:
        #     prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float128) if output_prob else None
        # else:
        #     prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float64) if output_prob else None
        
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

    def parallel_decode_batch(self, syndromes: np.ndarray[bool], output_prob=False, num_workers:int = None) -> np.ndarray[bool]:
        """Performs parallel batch decoding.
        
        Args:
            syndromes: 2D numpy array of boolean syndromes to decode
            output_prob: Whether to return probability distributions along with predictions
            num_workers: Number of worker processes to use (defaults to half of available CPUs)
            
        Returns:
            If output_prob is True:
                Tuple of (predictions, probability_distributions)
            Else:
                Only predictions array
            Where:
            - predictions: 2D numpy array of decoded logical operators (bool)
            - probability_distributions: 2D numpy array of probability distributions
        """
        # Calculate number of possible outcomes based on observables
        num_observables = self.num_observables
        num_possible_outcomes = 2 ** num_observables

        # Initialize predictions array
        predictions = np.empty((syndromes.shape[0], num_observables), dtype=bool)

        # Initialize probability distributions array if requested
        prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float64) if output_prob else None
        # use_decimal is not used in this method.
        # if self.use_decimal:
        #     prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float128) if output_prob else None
        # else:
        #     prob_dists = np.zeros((syndromes.shape[0], num_possible_outcomes), dtype=np.float64) if output_prob else None

        # Set number of workers if not specified
        if num_workers == None:
            # Calculate number of workers (leaving half CPUs free)
            num_workers = int(multiprocessing.cpu_count() // 2)
        
        # Set progress reporting interval (10% of total)
        per_print_counts = syndromes.shape[0] // 10
        
        # Disable lookup table for parallel decoding as it's not effective
        self.set_look_up_table(False)
        logger.debug(f"start decode.")
        start_time = time.time()
        
        # Perform parallel decoding if not using C++ implementation
        if self.contraction_code != "cpp":
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.starmap(decode_task, [(syndrome, self, output_prob, None) for syndrome in syndromes])
        # elif self.use_approx and self.contraction_code == "cpp":
        #     with multiprocessing.Pool(processes=num_workers) as pool:
        #         results = pool.starmap(approx_decode_task_cpp, [(syndrome, self.contractor_init_args, output_prob, None) for syndrome in syndromes])
        # elif not self.use_approx and self.contraction_code == "cpp":
        #     with multiprocessing.Pool(processes=num_workers) as pool:
        #         results = pool.starmap(decode_task_cpp, [(syndrome, self.contractor_init_args, output_prob, None) for syndrome in syndromes])
        
        end_time = time.time()
        logger.debug(f"finish decode, spend time: {end_time - start_time}")
        
        # Process results from parallel workers
        for i, (pred, prob_dist) in enumerate(results):
            predictions[i] = pred
            
            # Handle probability distribution output if requested
            if output_prob:
                if self.contraction_code == "normal":
                    # Standard case - convert probability distribution to array
                    prob = np.full(num_possible_outcomes, None)
                    for key, prob_value in prob_dist.items():
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                    prob_dists[i] = prob
                elif self.contraction_code == "int":
                    # Integer case - convert integer keys to binary first
                    prob = np.full(num_possible_outcomes, None)
                    for key, prob_value in prob_dist.items():
                        key = int_to_binary_bitwise(key, self.total_length)
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                    prob_dists[i] = prob
                elif self.contraction_code == "emld" or self.contraction_code == "cpp-py":
                    # Integer case - convert integer keys to binary first
                    prob = np.full(num_possible_outcomes, None)
                    for key, prob_value in prob_dist.items():
                        key = str_to_binary_bitwise(key)
                        index = int("".join(map(str, map(int, key[-num_observables:]))), 2)
                        prob[index] = prob_value
                    prob_dists[i] = prob
                else:
                    # Disable probability output for other contraction codes
                    output_prob = False
                    prob_dists = None
            
            # Log progress at intervals
            if (i + 1) % per_print_counts == 0:
                logger.debug(f"Processed {i + 1} syndromes out of {syndromes.shape[0]}")
        
        if predictions is not None and prob_dists is not None:
            logger.debug(f"predictions:{predictions.shape}, prob_dists:{prob_dists.shape}.")
        
        # Return results based on output_prob flag
        return (predictions, prob_dists) if output_prob else predictions

    def parallel_rounds_decode_batch(self, syndromes: np.ndarray[bool], m: int= None, all_num_workers: int = None) -> np.ndarray[bool]:
        """Performs parallel rounds decoding.
        
        Args:
            syndromes: 2D numpy array of boolean syndromes to decode
            m: Number of rounds (layers) to use (defaults to self.m if None)
            
        Returns:
            2D numpy array of decoded logical operators (bool)
        """
        # Get number of observables from instance
        num_observables = self.num_observables
        
        # Initialize predictions array
        predictions = np.empty((syndromes.shape[0], num_observables), dtype=bool)

        if all_num_workers == None:
            # Calculate number of workers (leaving half CPUs free)
            all_num_workers = int(multiprocessing.cpu_count() // 2)
        
        # Get number of shots (syndromes) and set progress interval
        num_shots = syndromes.shape[0]

        # Use default m if not specified
        if m == None:
            m = self.m
        
        # Preprocess syndromes for parallel rounds
        parallel_syndromes = preprocess_parallel_rounds_syndromes(
            syndromes=syndromes, 
            partitioned_hypergraphs=self.partitioned_hypergraphs, 
            m=self.m
        )
        logger.debug(f"parallel_syndromes len: {len(parallel_syndromes)}, one shape is:{parallel_syndromes[0].shape}")

        # Phase 1: Parallel decoding of each round
        logger.debug(f"start decode.")
        start_time = time.time()
        
        with multiprocessing.Pool(processes=all_num_workers) as pool:
            # Distribute work across workers (all rounds for each shot)
            results = pool.starmap(
                parallel_rounds_decode_task,
                [(parallel_syndromes[round_i][shots_i,:], self.contractor[round_i]) 
                 for shots_i in range(num_shots) for round_i in range(m)]
            )
        
        end_time = time.time()
        logger.debug(f"finish decode, spend time: {end_time - start_time}")

        # Phase 2: Merge results from all rounds
        logger.debug(f"start merge.")
        start_time = time.time()
        
        # Group results by shot (m results per shot)
        grouped_results = [results[i:i+m] for i in range(0, len(results), m)]

        # Parallel merge of results
        with multiprocessing.Pool(processes=all_num_workers) as pool:
            merged_results = pool.starmap(
                decode_merge_task,
                [(group, self.logical_prob_dist, self.num_observables) 
                 for group in grouped_results]
            )
        
        end_time = time.time()
        logger.debug(f"finish merge, spend time: {end_time - start_time}")
        
        # Phase 3: Store final predictions
        start_time = time.time()
        for i, pred in enumerate(merged_results):
            predictions[i] = pred
        end_time = time.time()
        logger.debug(f"finish reset, spend time: {end_time - start_time}")
        
        return predictions