import time
import numpy as np
from .utility import generate_all_possible_syndromes, generate_syndromes_generator
import logging
from hamld.logging_config import setup_logger
from typing import Dict, Optional, Tuple

logger = setup_logger("hamld/benchmark/logical_error_expectation_benchmarking", log_level=logging.WARNING)

class LogicalErrorExpectationBenchmark:
    def __init__(self, decoder_function, d, r, p, noise_model, error_type="Z"):
        """
        初始化期望值基准测试对象。
        
        该方法只支持MLD类的方法，比如MLD和HAMLD。

        参数：
        - decoder_function: 解码器函数
        - d: 量子码尺寸
        - r: 相关参数
        - p: 噪声模型的概率参数
        - noise_model: 噪声模型
        - error_type: 错误类型（如 "Z"）
        """
        self.decoder_function = decoder_function
        self.d = d
        self.r = r
        self.p = p
        self.noise_model = noise_model
        self.error_type = error_type

        # 初始化查找表
        self.all_syndrome_lookup_table = {}
        
        self.syndrome = None
        self.predicted_observables = None
        self.prob_dists = None
        
    def set_decoder_function(self, decoder_function):
        """
        设置新的解码器函数。

        参数：
        - decoder_function: 新的解码器函数
        """
        self.decoder_function = decoder_function
        print("Decoder function has been updated.")

    def logical_error_expectation(self, syndrome: np.ndarray) -> np.float128:
        """
        计算逻辑错误期望值。

        注意，这里默认只有一个观测量。

        参数:
        - syndrome: 量子错误模式数据，通常包含所有可能的 syndrome 情况。

        返回:
        - 逻辑错误期望值
        """
        # 获取样本数量
        num_shots = int(syndrome.shape[0])

        try:
            # 尝试使用并行解码方法，并输出概率分布
            predicted_observables, prob_dists = self.decoder_function.parallel_decode_batch(syndrome, output_prob=True)
        except Exception as e:
        ## 若并行解码方法不存在，则使用普通解码方法
            predicted_observables = self.decoder_function.decode_batch(syndrome)
            prob_dists = None
            logger.info("并行解码方法函数调用失败，错误信息: %s", e)
            logging.info("使用普通解码方法")
            
        # 记录解码完成信息
        logger.info("量子错误模式数据解码完成")
        
        # 存储预测的观测量和概率分布，用于生成查找表
        self.predicted_observables = predicted_observables
        self.prob_dists = prob_dists

        if prob_dists is not None:
            # 初始化概率累加器
            total_prob = np.float128(0)
            true_prob = np.float128(0)
            error_prob = np.float128(0)

            for i in range(num_shots):
                # 获取当前样本的预测值
                predicted_value = predicted_observables[i, 0]
                # print(type(predicted_value))
                if isinstance(np.bool_(True), (bool, np.bool_)):
                    if predicted_value:
                        # 预测值为 True 时，累加相应概率
                        true_prob += np.float128(prob_dists[i][1])
                        error_prob += np.float128(prob_dists[i][0])
                    else:
                        # 预测值为 False 时，累加相应概率
                        true_prob += np.float128(prob_dists[i][0])
                        error_prob += np.float128(prob_dists[i][1])
                else:
                    # 若预测值不是布尔类型，抛出异常
                    raise ValueError(f"预测观测量中索引 {i} 处出现意外值: {predicted_value}")

                # 累加当前样本的总概率
                total_prob += np.float128(prob_dists[i][0]) + np.float128(prob_dists[i][1])

            # 期望逻辑错误率
            expectation_error = error_prob

            # 记录日志，包含错误期望值、正确率和总概率
            logger.info(f"逻辑错误期望值计算完成，逻辑错误期望值为: {expectation_error}，正确预测概率为: {true_prob}，总概率为: {total_prob}")

            return expectation_error
        else:
            # 若解码函数不支持估计逻辑错误率，记录日志
            logger.info("解码函数不支持输出概率分布，无法计算逻辑错误期望值。")
            return np.float128(0)

    def run(self, have_stabilizer: bool = True):
        """
        运行基准测试，计算逻辑错误期望值。

        返回：
        - average_error_expectation: 平均逻辑错误期望值
        - error_expectations: 每次测试的逻辑错误期望值
        """
        error_expectations = []

        # 创建所需的 syndrome 和 actual_observables
        syndrome = generate_all_possible_syndromes(self.d, self.r, have_stabilizer)
        self.syndrome = syndrome
        # 基准测试运行一次
        logger.info("Starting benchmarking...")

        start_time = time.time()  # 记录开始时间
        
        # 计算逻辑错误期望值
        error_expectation = self.logical_error_expectation(syndrome)
        error_expectations.append(error_expectation)
        
        end_time = time.time()  # 记录结束时间
        logger.info(f"Run completed in {end_time - start_time:.4f} seconds - Error expectation: {error_expectation}")

        # 计算平均逻辑错误期望值
        average_error_expectation = np.mean(error_expectations)
        
        logger.info(f"Average logical error expectation (over d = {self.d} all syndrome runs): {average_error_expectation:.6f}")
        
        return average_error_expectation, error_expectations
    
    def get_all_syndrome_lookup_table(self):
        if self.syndrome is not None and self.prob_dists is not None and self.predicted_observables is not None:
            num_shots = int(self.syndrome.shape[0])  # 获取样本数量
            for i in range(num_shots):
                # 将 syndrome 对应的 predicted_observables 和 prob_dists 添加到 lookup table
                self.all_syndrome_lookup_table[tuple(self.syndrome[i,:])] = (
                    self.predicted_observables[i,:],  # 预测的观测值
                    self.prob_dists[i, :],  # 概率分布
                    None  # 这里可以根据需要计算 prob_correct_correction 或其他值
                )
            return self.all_syndrome_lookup_table
        else:
            raise ValueError("Syndrome, prob_dists, or predicted_observables is None.")

        