import time
from .utility import generate_syndrome_and_observables, generate_qldpc_syndrome_and_observables
import numpy as np

import logging
from eamld.logging_config import setup_logger

try :
    import cudf
except:
    import pandas as pd
    print("if you have many shots. cudf can process data efficiently. pleased install cudf.")

def process_syndrome(sample_syndrome, shots):
    """
    处理 采样的syndrome 数据, 将相同的syndrome作为。
    
    参数：
        - syndrome: 原始 syndrome 数据
    返回：
        - processed_syndrome: 处理后的 syndrome 数据
        - shots: 保留的采样次数
    """
    len_syndrome = sample_syndrome.shape[1]

    try:
        # 尝试使用 cudf 进行处理
        df = cudf.DataFrame(sample_syndrome)
        unique_counts = df.value_counts().reset_index()
    except:
        # 无法调用cudf，使用df进行处理
        df = pd.DataFrame(sample_syndrome)
        # 使用 pandas 的 value_counts 获取唯一行计数
        unique_counts = df.value_counts().reset_index()

    unique_sample_syndrome = unique_counts.iloc[:, :-1].to_numpy()
    # counts = unique_counts.iloc[:, -1].to_numpy()
    
    # 随机选择shots个样本
    if unique_sample_syndrome.shape[0] > shots:
        random_indices = np.random.choice(unique_sample_syndrome.shape[0], shots, replace=False)
        unique_syndrome = unique_sample_syndrome[random_indices, :len_syndrome]
    else:
        unique_syndrome = unique_sample_syndrome[:, :len_syndrome]

    return unique_syndrome

logger = setup_logger("eamld/benchmark/decoder_speed_benchmarking", log_level=logging.WARNING)

class DecoderSpeedBenchmark:
    def __init__(self, decoder_function, d = None , nkd =  None, r = 1, p = 10, noise_model = "si1000", error_type="Z", num_runs=10, 
                 detector_num:int = None, code_name = "surface code",
                 data_path = None, have_stabilizer = False):
        """
        初始化基准测试对象。

        参数：
        - decoder_function: 解码器函数
        - d: 量子码尺寸, 其中d并未参与到计算中, 后续将删除。
        - r: 相关参数
        - p: 噪声模型的概率参数
        - noise_model: 噪声模型
        - error_type: 错误类型（如 "Z"）
        - num_runs: 基准测试运行次数
        - data_path: 数据文件路径
        - detector_num: 检测器数量
        - code_name: 代码名称
        """
        self.decoder_function = decoder_function

        self.nkd = nkd
        self.d = d
        self.r = r
        self.p = p
        self.noise_model = noise_model
        self.error_type = error_type
        self.num_runs = num_runs
        self.detector_num = detector_num
        self.code_name = code_name

        self.data_path = data_path
        self.have_stabilizer = have_stabilizer

    def set_decoder_function(self, decoder_function):
        """
        设置新的解码器函数。

        参数：
        - decoder_function: 新的解码器函数
        """
        self.decoder_function = decoder_function
        print("Decoder function has been updated.")

    def decoder_per_time(self, syndrome: np.ndarray) -> float:
        """
        估计解码速度。

        参数：
        - syndrome: 量子错误模式数据

        返回：
        - 逻辑错误率
        """
        num_shots = int(syndrome.shape[0])

        start_time = time.time()
        self.decoder_function.decode_batch(syndrome)
        per_shots_time = (time.time() - start_time) / num_shots
        
        per_rounds_time = per_shots_time / self.r
        
        return per_rounds_time

    def generate_random_syndrome(self, shots):
        # 生成器的形式，可以通过遍历逐步生成。
        if self.have_stabilizer ==  True:
            syndrome_length = (self.d**2 - 1) * self.r
        else:
            syndrome_length = int((self.d**2 - 1) * self.r - (self.d**2 - 1)/2)
        syndrome = np.random.rand(shots, syndrome_length) < 0.5  # 直接生成布尔值
        return syndrome

    def generate_random_syndrome_qldpc(self, shots):
        syndrome_length = self.detector_num
        syndrome = np.random.rand(shots, syndrome_length) < 0.5  # 直接生成布尔值
        return syndrome

    def run(self, shots):
        """
        运行基准测试，计算逻辑错误率。

        返回：
        - average_error_rate: 平均错误率
        - error_rates: 每次测试的错误率
        """
        per_time_list = []
        
        # 基准测试多次运行
        logger.debug("Starting benchmarking...")

        # 使用 tqdm 显示进度条
        # for _ in tqdm(range(self.num_runs), desc="Benchmarking runs", ncols=100, position=0, leave=True):
        for i in range(self.num_runs):
            # 创建所需的 syndrome 和 actual_observables
            if self.data_path is None:
                if self.code_name == "surface code":
                    syndrome = self.generate_random_syndrome(shots)
                elif self.code_name == "qldpc code":
                    syndrome = self.generate_random_syndrome_qldpc(shots)
            else:
                if self.code_name == "surface code":
                    syndrome, _ = generate_syndrome_and_observables(self.d, self.r, self.p, self.noise_model, self.error_type, self.data_path, self.have_stabilizer)
                elif self.code_name == "qldpc code":
                    syndrome, _ = generate_qldpc_syndrome_and_observables(self.nkd, self.r, self.p, self.noise_model, self.error_type, self.data_path, self.have_stabilizer)
                    # logger.info(f"Decode Syndrome shape: {syndrome.shape}")
                else:
                    raise ValueError("Invalid code name. Please choose 'surface code' or 'qldpc code'.")
                
                syndrome = process_syndrome(syndrome, shots)
                
            if i==0:
                logger.info(f"Decode Syndrome shape: {syndrome.shape}")
            
            start_time = time.time()  # 记录开始时间
            
            # 计算逻辑错误率
            per_time = self.decoder_per_time(syndrome)
            per_time_list.append(per_time)
            
            end_time = time.time()  # 记录结束时间
            logger.debug(f"Run completed in {end_time - start_time:.4f} seconds - Decoder per time: {per_time:.6f}")

        # 计算平均逻辑错误率
        average_per_rounds_time = np.mean(per_time_list)
        
        logger.info(f"Average logical error rate (over {self.num_runs} runs), rounds is {self.r}: {average_per_rounds_time:.6f}")
        
        return average_per_rounds_time, per_time_list
