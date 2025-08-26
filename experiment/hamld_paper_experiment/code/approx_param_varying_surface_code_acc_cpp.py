import os
import logging
from datetime import datetime
import csv
import hamld
# import pymatching
# from stimbposd import BPOSD
from hamld.benchmark import generate_detector_error_model, LogicalErrorRateBenchmark, generate_detector_error_model_path
# 设置 logging 配置，放在模块级别
from hamld.logging_config import setup_logger

from hamld import HAMLDCpp_from_file

logger = setup_logger("epmld_paper_experiment/approx_param_varying_surface_code_acc_cpp", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    # related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/approx_param_varying/surface_code"
    related_path =  "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/surface_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/epmld/experiment/epmld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"approx_param_varying_surface_code_acc_cpp_results_{timestamp}.csv")

    # 设置Surface code的相关初始化参数
    # code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    code_tasks = ["surface_code:rotated_memory_x"]
    # distances = [3, 5, 7, 9]
    # distances = [13]
    # distances = [19, 17, 15]
    distances = [19]
    # 固定噪声参数和噪声模型：
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]

    # 设置待测试的解码方法：
    decoder_methods = ["HAMLD"]  # r=d的情况下，无损的HAMLD，也是一个非常大的指数，只有在d=3的时候才能跑，因为这里我们只考虑HAMLD，理论上是能够跑完的，10**5。

    # 近似的MLD方法的初始化参数：
    approximatestrategy = "hyperedge_topk"
    # approximate_params = [10, 50, 100, 200]
    # 超图截取的优先级，超图截取的最大数量
    # prioritys = [0, -1, -2]
    # priority_topks = [100, 150, 200]

    prioritys = [0, 0, 0, 0]
    # d=13,r=13, 2100
    # priority_topks = [210, 100, 150, 200]
    # approximate_params = [100, 500, 1000]
    # prioritys = [0, -1, -2]
    # priority_topks = [100, 500, 1000]

    # 创建存储结果的CSV文件并写入表头
    fieldnames = ['code_task', 'detector_number', 'd', 'round', 'approximate_param', 'priority', 'priority_topk', 'decoder_method', 'logical_error_rate', 'have_stabilizer', 'total_running_time']
    # 如果文件不存在，写入表头
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 使用 with 来避免每次打开文件
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 添加打印信息
        logger.info("Starting the benchmark process...")

        for code_task in code_tasks:
            error_type = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
            for have_stabilizer in have_stabilizers:
                for d in distances:
                    for p in probabilities:
                        # TODO: 根据所需轮次，修改该部分的代码。
                        rounds = [d]
                        for r in rounds:
                            for noise_model in noise_models:
                                for decoder_method in decoder_methods:
                                    detector_number = int((d**2-1)/2 + (d**2-1) * (r-1))
                                    
                                    prioritys = [0, 0, 0, 0]
                                    # d=13,r=13, 2100
                                    approximate_params = [500]
                                    # approximate_params = [100]
                                    # priority_topks = [int(detector_number/4), int(detector_number/3)]
                                    priority_topks = [int(detector_number/4)]
                                    # priority_topks = [1000]
                                    for priority_index in range(len(priority_topks)):
                                        priority = prioritys[priority_index]
                                        priority_topk = priority_topks[priority_index]
                                        for approximate_param in approximate_params:
                                            
                                            
                                            logger.info(f"Running for {code_task}, d={d}, r={r},probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")
                                            logger.info(f"approximate_param={approximate_param}, priority={priority}, priority_topk={priority_topk}")
                                            # 获取对应的检测器错误率模型，用于构建解码器
                                            dem_file = generate_detector_error_model_path(d = d, r=r, p=p, noise_model=noise_model,
                                                                                error_type=error_type, decomposed_error = False,
                                                                                related_path=related_path, have_stabilizer = have_stabilizer)
                                            decoder =  HAMLDCpp_from_file(dem_path=str(dem_file), 
                                                                            approx_strategy=str(approximatestrategy), 
                                                                            approx_param=float(approximate_param), 
                                                                            priority=int(priority), 
                                                                            priority_topk = int(priority_topk),
                                                                            use_heuristic = False, alpha = float(0.05))
                                            
                                            # 创建基准测试对象
                                            benchmark = LogicalErrorRateBenchmark(
                                                decoder_function=decoder,
                                                d=d,
                                                nkd=None,
                                                r=r,
                                                p=p,
                                                noise_model=noise_model,
                                                error_type=error_type,
                                                num_runs=1,
                                                data_path=related_path,
                                                code_name="surface code",
                                                have_stabilizer = have_stabilizer,
                                                code_language="c++"
                                            )
                                            
                                            # 运行基准测试并获取结果
                                            logical_error_rate = benchmark.run(is_unique_decode=False)[0]
                                            total_running_time = benchmark.decoder_function.total_running_time()
                                            logger.info(f"Logical Error Rate for {code_task}, d={d}, p={p}, noise_model={noise_model}, decoder={decoder_method}: {logical_error_rate}, total_running_time = {total_running_time}")

                                            # 将结果写入CSV文件
                                            writer.writerow({
                                                'code_task': code_task,
                                                'detector_number': detector_number,
                                                'd': d,
                                                'round': r,
                                                'approximate_param': approximate_param,
                                                'priority': priority,
                                                'priority_topk': priority_topk,
                                                'decoder_method': decoder_method,
                                                'logical_error_rate': logical_error_rate,
                                                'have_stabilizer': have_stabilizer,
                                                'total_running_time': total_running_time
                                            })

    logger.info(f"Finish the benchmark process.")

if __name__ == "__main__":
    main()

# 后台运行命令
# nohup python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/approx_param_varying_surface_code_acc_cpp.py > /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/log/approx_param_varying_surface_code_acc_cpp_output.log 2>&1 &
# 1599380