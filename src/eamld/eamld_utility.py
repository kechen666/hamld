import numpy as np
from typing import Dict, Union, Tuple
import heapq

def validate_logical_operator(
    prob_dist: Dict[str, Union[np.float32, np.float64]],
    logical_number: int
) -> Tuple[np.ndarray[bool], Union[np.float32, np.float64]]:
    """
    验证是否发生逻辑错误，并返回纠错正确概率
    
    参数:
        prob_dist: 收缩后的概率分布
        logical_number: 逻辑量子比特的数量
        
    返回:
        Tuple[np.ndarray[bool], Union[np.float32, np.float64]]:
            - 逻辑错误指示数组 (True表示发生逻辑错误)
            - 正确纠错的概率
    """
    keys = list(prob_dist.keys())
    
    if len(keys) == 1:
        last_logical_bits = keys[0][-logical_number:]
        logical_error_detected = np.array([bit == '1' for bit in last_logical_bits], dtype=bool)
        return np.array([logical_error_detected], dtype=bool), 1.0
    elif len(keys) == 0:
        return np.array([[False]*logical_number], dtype=bool), 0.5
    elif len(keys) >= 2:
        max_key = max(prob_dist, key=prob_dist.get)
        last_logical_bits = max_key[-logical_number:]
        
        logical_error_detected = np.array([bit == '1' for bit in last_logical_bits], dtype=bool)
        
        max_prob = prob_dist[max_key]
        total_prob = sum(prob_dist.values())
        prob_correct_correction = max_prob / total_prob
    
        return np.array([logical_error_detected], dtype=bool), prob_correct_correction

def preprocess_parallel_rounds_syndromes(
    syndromes: np.ndarray,
    partitioned_hypergraphs: list,
    m: int
) -> list[np.ndarray]:
    """
    预处理并行rounds的syndrome数据
    
    参数:
        syndromes: 原始syndrome数据，形状为(num_shots, total_detectors)
        partitioned_hypergraphs: 分层的超图列表
        m: 分层数
        
    返回:
        预处理后的并行syndrome列表，每个元素对应一层的syndrome数据
    """
    parallel_syndromes = []
    num_shots = syndromes.shape[0]

    for layer_i in range(m):
        par_hypergraph = partitioned_hypergraphs[layer_i]
        all_detector_num = par_hypergraph.get_nodes_number(have_logical_observable = False)
        in_layer_detector_num = par_hypergraph.detector_number
        
        if layer_i == 0:
            start_index = 0
            end_index = in_layer_detector_num
        else:
            start_index = end_index
            end_index = end_index + in_layer_detector_num
            
        one_layer_syndromes = np.zeros((num_shots, all_detector_num), dtype=bool)
        one_layer_syndromes[:, :in_layer_detector_num] = syndromes[:, start_index:end_index]
        parallel_syndromes.append(one_layer_syndromes)
        
    return parallel_syndromes

def merge_logical_prob_dists(
    dist1: Dict[str, float], 
    dist2: Dict[str, float], 
) -> Dict[str, float]:
    """
    合并两个逻辑概率分布，对每一位进行异或操作，并将概率相乘
    
    参数:
        dist1: 第一个概率分布
        dist2: 第二个概率分布
        logical_number: 逻辑比特数
        
    返回:
        合并后的概率分布
    """
    # 将字典转换为numpy数组
    keys1 = np.array([list(map(int, key)) for key in dist1.keys()])
    probs1 = np.array(list(dist1.values()))
    keys2 = np.array([list(map(int, key)) for key in dist2.keys()])
    probs2 = np.array(list(dist2.values()))

    # 利用广播机制进行向量化异或操作
    xor_results = keys1[:, None, :] ^ keys2[None, :, :]

    # 计算概率乘积
    prob_products = probs1[:, None] * probs2[None, :]
    
    # 将结果转换为字符串并累加概率
    merged_dist = {}
    for i in range(xor_results.shape[0]):
        for j in range(xor_results.shape[1]):
            xor_key = ''.join(map(str, xor_results[i, j]))
            if xor_key in merged_dist:
                merged_dist[xor_key] += prob_products[i, j]
            else:
                merged_dist[xor_key] = prob_products[i, j]
                
    return merged_dist

def compute_logical_prob_dist(
    hypergraph,
    logical_hyperedges: list[str],
    logical_number: int
)-> Dict[str, float]:
    """
    计算逻辑概率分布

    参数:
        hypergraph: 超图对象
        logical_hyperedges: 逻辑超边列表
    返回:
        逻辑概率分布字典
    """
    logical_prob_dist = {"0"* logical_number: 1.0}
    logical_contractable_hyperedges_weights_dict = hypergraph.get_hyperedges_weights_dict(logical_hyperedges)
    for hyperedge, prob in logical_contractable_hyperedges_weights_dict.items():
        new_prob_dist = {}
        prob = float(prob)
        for key, value in logical_prob_dist.items():
            # 将key转换为list以便修改
            key_list = list(key)
            # value = float(value)
            
            # 翻转超边对应的位
            flipped_key = key_list.copy()
            for bit in hyperedge:
                index = int(bit[1:])  # 提取L后面的数字作为索引
                flipped_key[index] = '1' if flipped_key[index] == '0' else '0'
            
            # 更新概率分布
            flipped_key_str = ''.join(flipped_key)
            new_prob_dist[flipped_key_str] = new_prob_dist.get(flipped_key_str, 0.0) + value * prob
            new_prob_dist[key] = new_prob_dist.get(key, 0.0) + value * (1 - prob)
        
        logical_prob_dist = new_prob_dist
    
    return logical_prob_dist

def merge_parallel_rounds_prob_dists(prob_dists, logical_prob_dist, logical_number:int, topk = 10):
    """
    合并多个并行轮次的概率分布

    参数:
        prob_dists: 一个shots对应的概率一组概率分布
        logical_prob_dist: 最后一层的逻辑概率分布
        logical_number: 逻辑比特数
    返回:
        合并后的概率分布
    """
    prob_dist = {"0"* logical_number: 1.0}

    for i, prob_dist_i in enumerate(prob_dists):
        if len(prob_dist_i) >=3:
            topk_items  = heapq.nlargest(topk, prob_dist_i.items(), key=lambda x: x[1])
            logical_prob_dist_i = {key[-logical_number:]: value for key, value in topk_items}
        else:
            logical_prob_dist_i = {key[-logical_number:]: value for key, value in prob_dist_i.items()}
        prob_dist = merge_logical_prob_dists(prob_dist, logical_prob_dist_i)

    prob_dist = merge_logical_prob_dists(prob_dist, logical_prob_dist)
    
    return prob_dist

def decode_merge_task(prob_dists, logical_prob_dist, logical_number:int, topk = 10):
    """
    合并多个并行轮次的概率分布, 并预测解码结果
    参数:
        prob_dists: 一个shots对应的概率一组概率分布
        logical_prob_dist: 最后一层的逻辑概率分布
        logical_number: 逻辑比特数
    返回:
        合并后的概率分布
    """
    prob_dist = merge_parallel_rounds_prob_dists(prob_dists, logical_prob_dist, logical_number, topk)
    pred, prob_correct_correction = validate_logical_operator(prob_dist, logical_number)

    return pred
   
def decode_layer_task(syndrome: np.ndarray[bool], contractor):
    """Decoding task for a layer syndrome, used in parallel decoding."""
    prob_dist = contractor.mld_contraction_no_slicing(syndrome)
    # Validate and return the prediction
    # prediction, prob_correct_correction = contractor.validate_logical_operator(prob_dist)
    # pred, prob_dist, prob_correct_correction = (prediction, prob_dist, prob_correct_correction)  

    return prob_dist

def parallel_rounds_decode_task(rounds_syndromes, contractors):
    """
    并行解码任务，用于并行轮次的解码。

    参数:
        rounds_syndromes: 待解码的syndrome数据，形状为(num_shots, total_detectors)
        contractor_init_args: 收缩器的初始化参数列表

    """
    # 构建收缩器
    
    # 进行并行shots的收缩
    # with multiprocessing.Pool(processes = rounds_num_workers) as pool:
        # prob_dists = pool.starmap(decode_layer_task, [(rounds_syndromes[i], contractors[i]) for i in range(m)])

    # 后处理合并多个prob
    # 将后处理合并放到最后。
    # prob_dist = merge_parallel_rounds_prob_dists(prob_dists, logical_prob_dist, logical_number)

    # pred, prob_correct_correction = validate_logical_operator(prob_dist, logical_number)
    prob_dist = decode_layer_task(rounds_syndromes, contractors)
    return prob_dist