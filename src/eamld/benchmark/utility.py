import pathlib
from typing import List, Tuple
import itertools
import numpy as np
import stim

def parse_b8(data: bytes, bits_per_shot: int) -> List[List[bool]]:
    """
    解析 b8 格式的字节数据为布尔值的采样结果列表。
    
    参数：
    - data: 以字节形式给出的原始数据。
    - bits_per_shot: 每个采样包含的比特数。
    
    返回：
    - 返回一个包含多个采样结果的列表，每个采样结果为布尔值。

    TODO: 在读取r比较大的情况下，速度比较慢。
    """
    shots = []
    bytes_per_shot = (bits_per_shot + 7) // 8  # 每个采样的字节数
    for offset in range(0, len(data), bytes_per_shot):
        shot = []
        for k in range(bits_per_shot):
            byte = data[offset + k // 8]
            bit = (byte >> (k % 8)) & 1  # 提取单个比特值
            shot.append(bool(bit))  # 保证是布尔值
        shots.append(shot)
    return shots

def save_b8(shots: List[List[bool]]) -> bytes:
    output = b""
    for shot in shots:
        bytes_per_shot = (len(shot) + 7) // 8
        v = 0
        for b in reversed(shot):
            v <<= 1
            v += int(b)
        output += v.to_bytes(bytes_per_shot, 'little')
    return output

def b8_to_array(shots: List[List[bool]], logical_num = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 b8 格式的布尔采样结果转换为 numpy 数组。
    
    前 n-logical_num 列为 syndrome，最后logical_num列为 actual_observables。
    
    参数：
    - shots: 解析后的布尔采样结果列表。
    
    返回：
    - syndrome: 量子系统的 syndrome（二维数组）
    - actual_observables: 真实的观测值（二维数组）
    """
    shots = np.array(shots)
    syndrome = shots[:, :-logical_num]
    actual_observables = shots[:, -logical_num:]
    return syndrome, actual_observables

def generate_syndrome_and_observables(d: int, r: int, p: float, noise_model: str, error_type: str, 
                                       related_path: str = "../data/external/eamld_experiment_data/experiment_1_surface_code_fault_tolerance_threshold", 
                                       have_stabilizer = True, is_bias = False, bias_param = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据给定的 d, r 和噪声模型生成 syndrome 和 actual_observables。
    
    参数：
    - d: 量子码的尺寸
    - r: 问题相关的参数
    - p: 噪声模型的概率参数
    - noise_model: 噪声模型（例如，某种类型的比特翻转）
    - error_type: 错误类型（如 "Z"）
    - related_path: 数据文件路径，注意最后要加/。
    
    返回：
    - syndrome: 量子系统的 syndrome（二维数组）
    - actual_observables: 真实的观测值（1D 数组）
    """


    # 构建数据文件路径
    if have_stabilizer:
        # 计算每个 shot 所需的比特数
        bits_per_shot = int(d**2 - 1 + (r - 1) * (d**2 - 1) + 1)
        if is_bias:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}_bias{bias_param}.dat')
        else:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}.dat')
    else:
        bits_per_shot = int(d**2 - 1 + (r - 1) * (d**2 - 1) + 1 - (d**2 - 1)/2)
        if is_bias:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer_bias{bias_param}.dat')
        else:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer.dat')
    
    
    # 读取文件中的字节数据
    with open(file_path, 'rb') as f:
        data = f.read()

    # 解析 b8 格式数据
    shots = parse_b8(data, bits_per_shot)

    # 将解析结果转换为 numpy 数组
    syndrome, actual_observables = b8_to_array(shots)
        
    return syndrome, actual_observables

def generate_bias_measure_syndrome_and_observables(d: int, r: int, p: float, noise_model: str, error_type: str, 
                                       related_path: str = "../data/external/eamld_experiment_data/experiment_1_surface_code_fault_tolerance_threshold",
                                       have_stabilizer = True, p_00 = None, p_11 = None) -> str:
    """
    根据给定的 d, r 和噪声模型生成 翻转之后的syndrome 和 actual_observables。
    
    参数：
    - d: 量子码的尺寸
    - r: 问题相关的参数
    - p: 噪声模型的概率参数
    - noise_model: 噪声模型（例如，某种类型的比特翻转）
    - error_type: 错误类型（如 "Z"）
    - related_path: 数据文件路径，注意最后要加/。
    - have_stabilizer: 是否有stabilizer。
    - p_00: 测量00的概率。
    - p_11: 测量11的概率。
    
    返回：
    - b8_output: b8格式的syndrom和observables。
    """
    # 构建数据文件路径
    if have_stabilizer:
        # 计算每个 shot 所需的比特数
        bits_per_shot = int(d**2 - 1 + (r - 1) * (d**2 - 1) + 1)
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}.dat')
    else:
        bits_per_shot = int(d**2 - 1 + (r - 1) * (d**2 - 1) + 1 - (d**2 - 1)/2)
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer.dat')
        
    # 读取文件中的字节数据
    with open(file_path, 'rb') as f:
        data = f.read()

    # 解析 b8 格式数据
    shots = parse_b8(data, bits_per_shot)

    # 将解析结果转换为 numpy 数组
    syndrome, actual_observables = b8_to_array(shots)
    
    bias_measurement_syndrome = postprocess_syndrome(syndrome, p_00, p_11)
    
    # 合并bias_measurement_syndrome和actual_observables
    combined_data = np.hstack((bias_measurement_syndrome, actual_observables))
    
    combined_bool_list = combined_data.tolist()
    
    b8_output = save_b8(combined_bool_list)
    
    return b8_output

def generate_qldpc_syndrome_and_observables(nkd: List[int], r: int, p: float, noise_model: str, error_type: str, 
                                       related_path: str = "../data/external/eamld_experiment_data/paper_experiment_bb_codes", 
                                       have_stabilizer = True, is_bias = False, bias_param = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据给定的 nkd, r 和噪声模型生成 syndrome 和 actual_observables。
    
    参数：
    - nkd: 量子码的相关参数
    - r: 问题相关的参数
    - p: 噪声模型的概率参数
    - noise_model: 噪声模型（例如，某种类型的比特翻转）
    - error_type: 错误类型（如 "Z"）
    - related_path: 数据文件路径，注意最后要加/。
    
    返回：
    - syndrome: 量子系统的 syndrome（二维数组）
    - actual_observables: 真实的观测值（二维数组）
    
    # TODO: 待加速
    """
    # 计算每个 shot 所需的检测器数目和逻辑比特数


    # 构建数据文件路径
    if have_stabilizer:
        # 在BB中，检测器数量就是n的数量。
        bits_per_shot = int(nkd[0] * r + nkd[1])
        if is_bias:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/circuit_sample_{noise_model}_p{p}_bias{bias_param}.dat')
        else:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/circuit_sample_{noise_model}_p{p}.dat')
    else:
        bits_per_shot = int(nkd[0] * r + nkd[1] - nkd[0]/2)
        if is_bias:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer_bias{bias_param}.dat')
        else:
            file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer.dat')

    # 读取文件中的字节数据
    with open(file_path, 'rb') as f:
        data = f.read()

    # 解析 b8 格式数据
    shots = parse_b8(data, bits_per_shot)

    # 将解析结果转换为 numpy 数组
    syndrome, actual_observables = b8_to_array(shots, logical_num= nkd[1])
    
    return syndrome, actual_observables

def generate_bias_measure_qldpc_syndrome_and_observables(nkd: List[int], r: int, p: float, noise_model: str, error_type: str, 
                                       related_path: str = "../data/external/eamld_experiment_data/paper_experiment_bb_codes", 
                                       have_stabilizer = True, p_00 = None, p_11 = None) -> str:
    """
    根据给定的 nkd, r 和噪声模型生成 syndrome 和 actual_observables。
    
    参数：
    - nkd: 量子码的相关参数
    - r: 问题相关的参数
    - p: 噪声模型的概率参数
    - noise_model: 噪声模型（例如，某种类型的比特翻转）
    - error_type: 错误类型（如 "Z"）
    - related_path: 数据文件路径，注意最后要加/。
    - have_stabilizer: 是否有stabilizer。
    - p_00: 测量00的概率。
    - p_11: 测量11的概率。
    
    返回：
    - b8_output: b8格式的syndrom和observables。
    """

    # 构建数据文件路径
    if have_stabilizer:
        # 在BB中，检测器数量就是n的数量。
        bits_per_shot = int(nkd[0] * r + nkd[1])
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/circuit_sample_{noise_model}_p{p}.dat')
    else:
        bits_per_shot = int(nkd[0] * r + nkd[1] - nkd[0]/2)
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer.dat')

    # 读取文件中的字节数据
    with open(file_path, 'rb') as f:
        data = f.read()

    # 解析 b8 格式数据
    shots = parse_b8(data, bits_per_shot)

    # 将解析结果转换为 numpy 数组
    syndrome, actual_observables = b8_to_array(shots, logical_num= nkd[1])
    
    bias_measurement_syndrome = postprocess_syndrome(syndrome, p_00, p_11)
    
    # 合并bias_measurement_syndrome和actual_observables
    combined_data = np.hstack((bias_measurement_syndrome, actual_observables))
    
    combined_bool_list = combined_data.tolist()
    
    b8_output = save_b8(combined_bool_list)
    
    return b8_output

def generate_detector_error_model(d: int, r: int, p: float, noise_model: str, error_type: str, 
                                  decomposed_error: bool = False, 
                                  related_path: str = "../data/external/eamld_experiment_data/experiment_1_surface_code_fault_tolerance_threshold", have_stabilizer = True) -> stim.DetectorErrorModel:
    """
    生成对应的探测器错误模型。
    
    参数：
    - d: 量子码的尺寸
    - r: 问题相关的参数
    - p: 噪声模型的概率参数
    - noise_model: 噪声模型（如比特翻转）
    - error_type: 错误类型（如 "Z"）
    - decomposed_error: 是否使用分解错误模型（默认为 False）
    - related_path: 数据文件路径，注意最后要加/。
    
    返回：
    - 返回 `stim.DetectorErrorModel` 对象，表示探测器错误模型。
    """
    # 构建文件路径
    if have_stabilizer:
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/detector_error_model_{noise_model}_p{p}.dem')
        decomposed_file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/decomposed_detector_error_model_{noise_model}_p{p}.dem')
    else:
        # 构建文件路径
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/detector_error_model_{noise_model}_p{p}_no_stabilizer.dem')
        decomposed_file_path = pathlib.Path(related_path + "/" + f'{error_type}/d{d}_r{r}/decomposed_detector_error_model_{noise_model}_p{p}_no_stabilizer.dem')

    # 加载错误模型
    if decomposed_error:
        with open(decomposed_file_path) as f:
            dem = stim.DetectorErrorModel.from_file(decomposed_file_path)
    else:
        with open(file_path) as f:
            dem = stim.DetectorErrorModel.from_file(file_path)
                
    return dem

def generate_qldpc_detector_error_model(nkd: List[int], r: int, p: float, noise_model: str, error_type: str,
                                  related_path: str = "../data/external/eamld_experiment_data/paper_experiment_bb_codes", have_stabilizer = True) -> stim.DetectorErrorModel:
    """
    生成对应的探测器错误模型。
    
    参数：
    - nkd: 量子码的相关参数
    - r: 问题相关的参数
    - p: 噪声模型的概率参数
    - noise_model: 噪声模型（如比特翻转）
    - error_type: 错误类型（如 "Z"）
    - related_path: 数据文件路径，可以是相对路径或者绝对路径。
    
    返回：
    - 返回 `stim.DetectorErrorModel` 对象，表示探测器错误模型。
    """
    # 构建文件路径
    if have_stabilizer:
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/detector_error_model_{noise_model}_p{p}.dem')
    else:
        file_path = pathlib.Path(related_path + "/" + f'{error_type}/nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}/detector_error_model_{noise_model}_p{p}_no_stabilizer.dem')

    # 加载错误模型
    with open(file_path) as f:
        dem = stim.DetectorErrorModel.from_file(file_path)
                
    return dem

def generate_all_possible_syndromes(d: int, r: int, have_stabilizer = True):
    # In our paper, 'have_stabilizer' indicates that both data qubits and measurement qubits contribute to forming stabilizer information (syndrome) for decoding purposes.
    if have_stabilizer ==  True:
        syndrome_length = (d**2 - 1) * r
    else:
        syndrome_length = int((d**2 - 1) * r - (d**2 - 1)/2)
    
    # 生成所有可能的布尔组合
    all_syndromes = np.array(list(itertools.product([False, True], repeat=syndrome_length)))
    
    return all_syndromes

def generate_syndromes_generator(d: int, r: int, have_stabilizer = True):
    # 生成器的形式，可以通过遍历逐步生成。
    if have_stabilizer ==  True:
        syndrome_length = (d**2 - 1) * r
    else:
        syndrome_length = int((d**2 - 1) * r - (d**2 - 1)/2)
        
    for syndrome in itertools.product([False, True], repeat=syndrome_length):
        yield np.array(syndrome)

def postprocess_syndrome(syndrome, p_00, p_11):
    # Compute flip probabilities for both cases
    flip_if_false = 1 - p_00  # Probability to flip False to True
    flip_if_true = 1 - p_11   # Probability to flip True to False
    
    # Create random matrix
    rand_matrix = np.random.random(syndrome.shape)
    
    # Apply the appropriate flip for each case
    return ((syndrome & (rand_matrix >= flip_if_true)) | ((~syndrome) & (rand_matrix < flip_if_false)))

def unique_syndrome_and_observables(syndrome: np.ndarray, actual_observables: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 syndrome 和 actual_observables 进行去重。
    参数：
    - syndrome: 量子系统的 syndrome（二维数组）
    - actual_observables: 真实的观测值（二维数组）
    返回：
    - unique_syndrome: 去重后的 syndrome（二维数组）
    - unique_actual_observables: 对应去重后的 syndrome 的实际观测值（二维数组）
    - counts: 对应去重后的 syndrome 的出现次数（一维数组）
    """
    # 将 syndrome 和 actual_observables 合并为一个数组，然后进行去重。
    num_shots = int(syndrome.shape[0])
    len_syndrome = syndrome.shape[1]

    sample = np.hstack((syndrome, actual_observables))
    try :
        import cudf
    except:
        import pandas as pd
        print("if you have many shots. cudf can process data efficiently. pleased install cudf.")
    try:
        df = cudf.DataFrame(sample)
        unique_counts = df.value_counts().reset_index()
    except:
        df = pd.DataFrame(sample)
        unique_counts = df.value_counts().reset_index()

    unique_sample = unique_counts.iloc[:, :-1].to_numpy()
    counts = unique_counts.iloc[:, -1].to_numpy()

    # 切割出syndrome和observables部分
    unique_syndrome = unique_sample[:, :len_syndrome]
    unique_actual_observables = unique_sample[:, len_syndrome:]
    return unique_syndrome, unique_actual_observables, counts