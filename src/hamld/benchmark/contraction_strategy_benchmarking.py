import csv
import os
import stim
import logging
from typing import List, Tuple

from hamld.logging_config import setup_logger
from hamld.contraction_strategy.dem_to_hypergraph import DetectorErrorModelHypergraph
from hamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph
from hamld.contraction_strategy.mld_order_finder import GreedyMLDOrderFinder
from hamld.contraction_strategy.contraction_tree import ContractionTree
from hamld.contraction_strategy.slice_finder import SliceFinder

# 设置日志
logger = setup_logger("contraction_strategy_benchmarking", log_level=logging.DEBUG)

def contraction_strategy_benchmarking(
    d_list: List[int], 
    r_list: List[int], 
    sliced_hyperedge_count_list: List[int], 
    updated_order_list: List[bool], 
    exp_name: str, 
    date_from: str
):
    """
    Perform benchmarking for the contraction strategy over a set of parameters.
    
    Parameters:
        d_list (List[int]): List of code distances.
        r_list (List[int]): List of rounds.
        sliced_hyperedge_count_list (List[int]): List of sliced hyperedge counts.
        updated_order_list (List[bool]): List of updated orders (True/False).
        exp_name (str): The name of the experiment.
        date_from (str): Source of the data, either "google23" or "stim".

    Saves the benchmarking results as a CSV file.
    """
    # TODO: Add a data file, if data_from is 'google23'.
    
    # 检查输入列表是否为空
    if not all([d_list, r_list, sliced_hyperedge_count_list, updated_order_list]):
        logger.error("One or more input lists are empty!")
        return

    # 存储所有结果的列表
    results = []

    # 遍历所有参数组合进行实验
    for d in d_list:
        for r in r_list:
            for sliced_hyperedge_count in sliced_hyperedge_count_list:
                for updated_order in updated_order_list:
                    try:
                        # 生成对应的检测器误差模型
                        if date_from == "google23":
                            detector_error_model = get_data_by_google23(r=r, d=d)
                        elif date_from == "stim":
                            detector_error_model = get_data_by_stim(r=r, d=d)
                        else:
                            raise ValueError(f"Invalid date_from parameter '{date_from}'!")

                        # 进行基准测试
                        baseline_contraction_cost, baseline_contraction_width, contraction_cost, contraction_width, sliced_contraction_cost, sliced_contraction_width = contraction_strategy_single_benchmarking(
                            exp_name=exp_name, 
                            detector_error_model=detector_error_model, 
                            d=d, r=r,
                            sliced_hyperedge_count=sliced_hyperedge_count, 
                            updated_order=updated_order
                        )
                        
                        # 将结果存储到列表中
                        results.append({
                            "d": d,
                            "r": r,
                            "sliced_hyperedge_count": sliced_hyperedge_count,
                            "updated_order": updated_order,
                            "baseline_contraction_cost": baseline_contraction_cost,
                            "baseline_contraction_width": baseline_contraction_width,
                            "contraction_cost": contraction_cost,
                            "contraction_width": contraction_width,
                            "sliced_contraction_cost": sliced_contraction_cost,
                            "sliced_contraction_width": sliced_contraction_width
                        })
                    except Exception as e:
                        logger.error(f"Error during benchmarking with d={d}, r={r}, sliced_hyperedge_count={sliced_hyperedge_count}, updated_order={updated_order}: {e}")
                        continue

    # 如果没有结果，打印错误信息并返回
    if not results:
        logger.error("No results generated. Please check the input parameters or experiment setup.")
        return
    
    # 保存结果到 CSV 文件
    output_file = f"../data/results/{exp_name}_benchmarking_results.csv"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 将结果写入 CSV 文件
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()  # 写入表头
            writer.writerows(results)  # 写入数据行
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write results to {output_file}: {e}")

def get_data_by_google23(d: int = 3, r: int = 1, type_basic: str = 'Z') -> stim.DetectorErrorModel:
    """
    Load data from the google23 dataset for a given code distance and round.

    Parameters:
        d (int): Code distance (default: 3).
        r (int): Round number (default: 1).
        type_basic (str): The type of basic operation (default: 'Z').

    Returns:
        DetectorErrorModel: The loaded detector error model.
    """
    try:
        # 根据距离和轮次选择对应的文件
        if d in [3, 5] and r in range(1, 26, 2):
            if d==3:
                filename = f"../data/external/google23/surface_code_b{type_basic}_d{d}_r{r:02d}_center_3_5/circuit_noisy.stim"
            elif d==5:
                filename = f"../data/external/google23/surface_code_b{type_basic}_d{d}_r{r:02d}_center_5_5/circuit_noisy.stim"
            circuit_noisy = stim.Circuit.from_file(filename)
        else:
            raise ValueError(f"Invalid d={d} or r={r} for google23 dataset.")

        # 获取检测器误差模型
        return circuit_noisy.detector_error_model(flatten_loops=True)
    except Exception as e:
        logger.error(f"Failed to load data for google23 with d={d}, r={r}: {e}")
        raise

def get_data_by_stim(
    r: int = 1, 
    d: int = 3, 
    acdp: float = 0.001, 
    arfp: float = 0.0001, 
    bmfp: float = 0.005, 
    brddp: float = 0.001, 
    code_task: str = "surface_code:rotated_memory_z"
) -> stim.DetectorErrorModel:
    """
    Generate a surface code circuit and return the corresponding DetectorErrorModel.
    
    Parameters:
        r (int): Number of rounds (default: 1)
        d (int): Distance of the code (default: 3)
        acdp (float): After Clifford depolarization probability (default: 0.001)
        arfp (float): After reset flip probability (default: 0.0001)
        bmfp (float): Before measure flip probability (default: 0.005)
        brddp (float): Before round data depolarization probability (default: 0.001)
        code_task (str): The code task to generate (default: surface_code:rotated_memory_z)

    Returns:
        DetectorErrorModel: The generated detector error model based on the circuit.
    """
    try:
        # 生成 surface_code 电路
        circuit_noisy = stim.Circuit.generated(
            code_task,
            rounds=r,
            distance=d,
            after_clifford_depolarization=acdp,
            after_reset_flip_probability=arfp,
            before_measure_flip_probability=bmfp,
            before_round_data_depolarization=brddp
        )
        return circuit_noisy.detector_error_model(flatten_loops=True)
    except Exception as e:
        logger.error(f"Failed to generate detector error model for stim with r={r}, d={d}: {e}")
        raise

def get_circuit_by_stim(
    r: int = 1, 
    d: int = 3, 
    acdp: float = 0.001, 
    arfp: float = 0.0001, 
    bmfp: float = 0.005, 
    brddp: float = 0.001, 
    code_task: str = "surface_code:rotated_memory_z"
) -> stim.Circuit:
    """
    Generate a surface code circuit and return it.
    
    Parameters:
        r (int): Number of rounds (default: 1)
        d (int): Distance of the code (default: 3)
        acdp (float): After Clifford depolarization probability (default: 0.001)
        arfp (float): After reset flip probability (default: 0.0001)
        bmfp (float): Before measure flip probability (default: 0.005)
        brddp (float): Before round data depolarization probability (default: 0.001)
        code_task (str): The code task to generate (default: surface_code:rotated_memory_z)

    Returns:
        Circuit: the circuit.
    """
    try:
        # 生成 surface_code 电路
        circuit_noisy = stim.Circuit.generated(
            code_task,
            rounds=r,
            distance=d,
            after_clifford_depolarization=acdp,
            after_reset_flip_probability=arfp,
            before_measure_flip_probability=bmfp,
            before_round_data_depolarization=brddp
        )
        return circuit_noisy
    except Exception as e:
        logger.error(f"Failed to generate detector error model for stim with r={r}, d={d}: {e}")
        raise

def contraction_strategy_single_benchmarking(
    exp_name: str,
    detector_error_model: stim.DetectorErrorModel,
    d: int,
    r: int,
    sliced_hyperedge_count: int,
    updated_order: bool
) -> Tuple[int, int, int, int, int, int]:
    """
    Perform a single benchmarking for contraction strategy.

    Parameters:
        exp_name (str): Experiment name.
        detector_error_model (DetectorErrorModel): The detector error model.
        d (int): Code distance.
        r (int): Round number.
        sliced_hyperedge_count (int): Number of sliced hyperedges (parallelism).
        updated_order (bool): Whether to update the contraction order after slicing.

    Returns:
        Tuple[int, int, int, int, int, int]: 
            - baseline_contraction_cost: Baseline contraction cost.
            - baseline_contraction_width: Baseline contraction width.
            - contraction_cost: Contraction cost after using the greedy order.
            - contraction_width: Contraction width after using the greedy order.
            - sliced_contraction_cost: Sliced contraction cost.
            - sliced_contraction_width: Sliced contraction width.
    """
    try:
        # Initialize contraction metrics
        baseline_contraction_cost = baseline_contraction_width = 0
        contraction_cost = contraction_width = sliced_contraction_cost = sliced_contraction_width = 0

        logger.debug(f"Starting experiment: {exp_name}, d: {d}, r: {r}, sliced_hyperedge_count: {sliced_hyperedge_count}, updated_order: {updated_order}")

        # Create hypergraph and corresponding connectivity graph
        hypergraph = DetectorErrorModelHypergraph(detector_error_model=detector_error_model, have_logical_observable=True)
        connectivity_graph = ConnectivityGraph()
        connectivity_graph.hypergraph_to_connectivity_graph(hypergraph)

        # Baseline strategy: Use a sequential order (detector index order)
        order = [f"D{i}" for i in range(detector_error_model.num_detectors)]
        baseline_contraction_tree = ContractionTree(order, detector_error_model=detector_error_model)
        baseline_contraction_cost, baseline_contraction_width, _ = baseline_contraction_tree.get_contraction_cost_information()

        # Apply greedy algorithm to find optimized contraction order
        order_finder = GreedyMLDOrderFinder(connectivity_graph)
        optimized_order = order_finder.find_order()

        # Create contraction tree using the optimized order and get contraction metrics
        contraction_tree = ContractionTree(optimized_order, detector_error_model=detector_error_model)
        contraction_cost, contraction_width, _ = contraction_tree.get_contraction_cost_information()

        # Slice based on parallelism and update order if necessary
        slice_finder = SliceFinder(contraction_tree)
        slice_finder.slice_based_on_parallelism(sliced_hyperedge_count, updated_order=updated_order)
        sliced_contraction_cost, sliced_contraction_width = slice_finder.get_sliced_contraction_cost_information()

        # Log contraction metrics for debugging
        logger.debug(f"Baseline Contraction Cost: {baseline_contraction_cost}, Baseline Contraction Width: {baseline_contraction_width}")
        logger.debug(f"Contraction Cost (Optimized Order): {contraction_cost}, Contraction Width (Optimized Order): {contraction_width}")
        logger.debug(f"Sliced Contraction Cost: {sliced_contraction_cost}, Sliced Contraction Width: {sliced_contraction_width}")

        # Return the benchmarking results
        return baseline_contraction_cost, baseline_contraction_width, contraction_cost, contraction_width, sliced_contraction_cost, sliced_contraction_width

    except Exception as e:
        logger.error(f"Error during contraction strategy benchmarking: {e}")
        raise
