import time
from .utility import generate_syndrome_and_observables, generate_qldpc_syndrome_and_observables, postprocess_syndrome
import numpy as np
import logging
from hamld.logging_config import setup_logger
import pandas as pd
import multiprocessing

try :
    import cudf
except:
    print("if you have many shots. cudf can process data efficiently. pleased install cudf.")
    

# from hamld import extract_sub_dem_by_syndrome
import stim

logger = setup_logger("hamld/benchmark/logical_error_rate_benchmarking", log_level=logging.INFO)

class LogicalErrorRateBenchmark:
    def __init__(self, decoder_function, d = None , nkd =  None, r = 1, p = 10, noise_model = "si1000", error_type="Z", num_runs=1, 
                 data_path="../data/external/hamld_experiment_data/experiment_1_surface_code_fault_tolerance_threshold",
                 code_name = "surface code", have_stabilizer = False,
                 is_biased = False, biased_measure_param = 1,
                 measurement_param = None, code_language= "python"):
        """
        初始化基准测试对象。

        参数：
        - decoder_function: 解码器函数
        - d: 量子码尺寸,
        - r: 相关参数
        - p: 噪声模型的概率参数
        - noise_model: 噪声模型
        - error_type: 错误类型（如 "Z"）
        - num_runs: 基准测试运行次数
        - data_path: 数据文件路径
        - code_name: 纠错码名称
        """
        self.decoder_function = decoder_function
        self.d = d
        self.nkd = nkd
        self.r = r
        self.p = p
        self.noise_model = noise_model
        self.error_type = error_type
        self.num_runs = num_runs
        self.data_path = data_path
        self.code_name = code_name
        self.experiment_time = None
        self.have_stabilizer = have_stabilizer
        
        # biased
        self.is_biased = is_biased
        self.biased_measure_param = biased_measure_param

        # measurement
        self.measurement_param = measurement_param
        
        self.unique_syndrome = None
        self.unique_actual_observables = None
        self.counts = None
        self.unique_predicted_observables = None
        self.code_language = code_language
        
        self.total_time_seconds = None
        
    def set_decoder_function(self, decoder_function):
        """
        设置新的解码器函数。

        参数：
        - decoder_function: 新的解码器函数
        """
        self.decoder_function = decoder_function
        print("Decoder function has been updated.")

    def logical_error_rate(self, syndrome: np.ndarray, actual_observables: np.ndarray, is_unique_decode=True) -> float:
        """
        计算逻辑错误率。
+
        参数：
        - syndrome: 量子错误模式数据
        - actual_observables: 实际观测结果
        -is_unique_decode: 是否使用unique简化采样解码。

        返回：
        - 逻辑错误率
        """
        num_shots = int(syndrome.shape[0])
        if is_unique_decode:
            logger.info(f"start preprocessing, syndrome:{syndrome.shape}...")
            start_time = time.time()
            # 拼接syndrome和实际observables
            sample = np.hstack((syndrome, actual_observables))
            len_syndrome = syndrome.shape[1]
            
            try:
                # 尝试使用 cudf 进行处理
                df = cudf.DataFrame(sample)
                unique_counts = df.value_counts().reset_index()
            except:
                # 无法调用cudf，使用df进行处理
                df = pd.DataFrame(sample)
                # 使用 pandas 的 value_counts 获取唯一行计数
                unique_counts = df.value_counts().reset_index()
            unique_sample = unique_counts.iloc[:, :-1].to_numpy()
            counts = unique_counts.iloc[:, -1].to_numpy()
            
            # unique_sample, counts = np.unique(sample, axis=0, return_counts=True)
            num_unique_shots = unique_sample.shape[0]
            
            # 切割出syndrome和observables部分
            unique_syndrome = unique_sample[:, :len_syndrome]
            unique_actual_observables = unique_sample[:, len_syndrome:]
            logger.info(f"syndrome规模为{syndrome.shape} 预处理时间为：{time.time()-start_time}")
            
            if self.code_language == "python":
                try:
                    # 尝试使用并行的shots的解码，如果无法调用，则回退到普通的解码
                    logger.info("Use Parallel Decoder")
                    unique_predicted_observables = self.decoder_function.parallel_decode_batch(unique_syndrome)
                except:
                    # BP+OSD和pymatching都没有实现并行解码，但是sinter似乎可以实现差不多的效果。
                    # 转换为连续内存布局的数组
                    contiguous_syndrome = np.array(unique_syndrome)
                    unique_predicted_observables = self.decoder_function.decode_batch(contiguous_syndrome)
            elif self.code_language == "c++":
                thread_num = multiprocessing.cpu_count() -60
                unique_predicted_observables = self.decoder_function.parallel_decode_batch(unique_syndrome, False, thread_num)
            
            # logger.info(f"unique_predicted_observables.shape: {unique_predicted_observables.shape}")
            # logger.info(f"unique_predicted_observables: {unique_predicted_observables[:10,:]}")
            
            # 计算错误的样本数量
            mistakes_mask = np.any(unique_predicted_observables != unique_actual_observables, axis=1)
            num_mistakes = np.sum(mistakes_mask * counts)
            
            error_rate = num_mistakes / num_shots
            
            # 存储解码结果
            self.unique_syndrome = unique_syndrome
            self.unique_actual_observables = unique_actual_observables
            self.counts = counts
            self.unique_predicted_observables = unique_predicted_observables
            
            logger.info(f"{num_mistakes}/{num_shots} mistakes, num_unique_shots is {num_unique_shots}, error rate: {error_rate:.6f}")
        else:
            # 直接调用解码函数
            # 尝试调用 parallel_decode_batch，如果不存在则回退到 decode_batch
            if self.code_language == "python":
                # 如果 decoder_function 有并行接口和总时间属性，就并行
                if hasattr(self.decoder_function, "parallel_decode_batch") and hasattr(self.decoder_function, "total_decode_time"):
                    logger.info("Use Parallel Decoder")
                    predicted_observables = self.decoder_function.parallel_decode_batch(syndrome)
                    self.total_time_seconds = self.decoder_function.total_decode_time  # 从类中获取耗时
                    # else:
                    # raise AttributeError("Missing parallel_decode_batch or total_decode_time")
                else:
                    # 默认不使用并行的方法，用于统计解码时间。
                    logger.info("Parallel decoding failed or not available. Falling back to serial decoding.")
                    # logger.exception(e)  # 打印详细异常信息和 traceback
                    start_time = time.time()
                    predicted_observables = self.decoder_function.decode_batch(syndrome)
                    self.total_time_seconds = time.time() - start_time

                
            elif self.code_language == "c++":
                thread_num = multiprocessing.cpu_count() -60
                predicted_observables = self.decoder_function.parallel_decode_batch(syndrome, False, thread_num)
                self.total_time_seconds = self.decoder_function.total_running_time()
                
            # 计算错误次数
            mistakes_mask = np.any(predicted_observables != actual_observables, axis=1)
            num_mistakes = np.sum(mistakes_mask)
            
            error_rate = num_mistakes / num_shots

            logger.info(f"{num_mistakes}/{num_shots} mistakes, error rate: {error_rate:.6f}")
        
        return error_rate

    def run(self, is_unique_decode = True):
        """
        运行基准测试，计算逻辑错误率。

        返回：
        - average_error_rate: 平均错误率
        - error_rates: 每次测试的错误率
        """
        error_rates = []

        # 创建所需的 syndrome 和 actual_observables
        if self.code_name == "surface code":
            syndrome, actual_observables = generate_syndrome_and_observables(self.d, self.r, self.p, self.noise_model, 
                                                                             self.error_type, self.data_path, self.have_stabilizer,
                                                                             self.is_biased, self.biased_measure_param, measurement_param = self.measurement_param)
        elif self.code_name == "qldpc code":
            syndrome, actual_observables = generate_qldpc_syndrome_and_observables(self.nkd, self.r, self.p, self.noise_model,
                                                                                   self.error_type, self.data_path, self.have_stabilizer,
                                                                                   self.is_biased, self.biased_measure_param, measurement_param = self.measurement_param)
        else:
            raise ValueError("Invalid code name. Please choose 'surface code' or 'qldpc code'.")
        
        # 基准测试多次运行
        logger.info("Starting benchmarking...")
        
        for _ in range(self.num_runs):
            start_time = time.time()  # 记录开始时间
            
            # 计算逻辑错误率
            error_rate = self.logical_error_rate(syndrome, actual_observables, is_unique_decode)
            error_rates.append(error_rate)
            
            end_time = time.time()  # 记录结束时间
            # 保存实验时间
            self.experiment_time = end_time - start_time
            logger.info(f"Run completed in {self.experiment_time:.4f} seconds - Error rate: {error_rate:.6f}")

        # 计算平均逻辑错误率
        average_error_rate = np.mean(error_rates)
        
        logger.info(f"Average logical error rate (over {self.num_runs} runs): {average_error_rate:.6f}")
        
        return average_error_rate, error_rates
