import os
import logging
from datetime import datetime
import csv
import hamld
# import pymatching
# from stimbposd import BPOSD
from hamld.benchmark import generate_qldpc_detector_error_model, LogicalErrorRateBenchmark, generate_qldpc_detector_error_model_path
# 设置 logging 配置，放在模块级别
from hamld.logging_config import setup_logger

from hamld import HAMLDCpp_from_file

logger = setup_logger("epmld_paper_experiment/approx_param_varying_qldpc_code_acc_cpp", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    # related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/approx_param_varying/qldpc_code"
    related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/qldpc_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/epmld/experiment/epmld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"approx_param_varying_qldpc_code_acc_cpp_results_{timestamp}.csv")

    # 设置QLDPC code的相关初始化参数
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    # code_tasks = ["bivariate_bicycle_code:rotated_memory_x"]
    # nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    # nkds = [[72, 12, 6], [144, 12, 12]]
    nkds = [[72, 12, 6]]
    
    # nkds = [[144, 12, 12]]
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]

    # 设置待测试的解码方法：
    decoder_methods = ["HAMLD"]  # 近似的MLD方法（HAMLD）的对比。

    # 近似的MLD方法的初始化参数：
    approximatestrategy = "hyperedge_topk"
    # approximatestrategy ="hyperedge_threshold"
    
    # approximate_params = [200, 500, 800, 1000]
    # # prioritys = [1, 0, -1]
    # prioritys = [0, 0, 0, 0]
    # priority_topks = [200, 500, 800, 1000]

    # approximate_params = [1e-6]
    # approximate_params = [1000]
    prioritys = [1,0,-1]
    # priority_topks = [1000]
    # BP+OSD 初始化参数
    # max_bp_iters = 100

    # 创建存储结果的CSV文件并写入表头
    fieldnames = ['code_task', 'nkd', 'round', 'approximate_param', 'priority', 'priority_topk', 'decoder_method', 'logical_error_rate', 'have_stabilizer', 'total_running_time']
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
                for nkd in nkds:
                    for p in probabilities:
                        d = nkd[2]
                        # rounds = [1]
                        rounds = [d]
                        for r in rounds:
                            for noise_model in noise_models:
                                for decoder_method in decoder_methods:
                                    detector_number = int(nkd[0]/2 + nkd[0] * (r-1))
                                    logger.info(f"detector_number={detector_number}")
                                    priority_topks = [detector_number/3, detector_number/3*2, detector_number/3*3]
                                    approximate_params = [detector_number/3, detector_number/3*2, detector_number/3*3]
                                    for priority_index in range(len(priority_topks)):
                                        # priority = prioritys[priority_index]
                                        priority = 0
                                        priority_topk = int(priority_topks[priority_index])
                                        approximate_param = int(approximate_params[priority_index])
                                        # for approximate_param in approximate_params:
                                        logger.info(f"Running for {code_task}, nkd={nkd}, r={r}, probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")
                                        logger.info(f"approximate_param={approximate_param}, priority={priority}, priority_topk={priority_topk}")
                                        # 获取对应的检测器错误率模型，用于构建解码器
                                        dem = generate_qldpc_detector_error_model(nkd=nkd, r=r, p=p, noise_model=noise_model, 
                                                                                error_type=error_type,related_path=related_path,
                                                                                have_stabilizer = have_stabilizer)
                                        dem_file = generate_qldpc_detector_error_model_path(nkd=nkd, r=r, p=p, noise_model=noise_model, 
                                                                            error_type=error_type,related_path=related_path,
                                                                            have_stabilizer = have_stabilizer)
                                        decoder =  HAMLDCpp_from_file(dem_path=str(dem_file), 
                                                                        approx_strategy=str(approximatestrategy), 
                                                                        approx_param=float(approximate_param), 
                                                                        priority=int(priority), 
                                                                        priority_topk = int(priority_topk),
                                                                        use_heuristic = False, alpha = float(0.05))
                                        
                                        # 创建基准测试对象
                                        benchmark = LogicalErrorRateBenchmark(
                                            decoder_function=decoder,
                                            d=nkd[2],
                                            nkd=nkd,
                                            r=r,
                                            p=p,
                                            noise_model=noise_model,
                                            error_type=error_type,
                                            num_runs=1,
                                            data_path=related_path,
                                            code_name="qldpc code",
                                            have_stabilizer = have_stabilizer,
                                            code_language="c++"
                                        )

                                        # 运行基准测试并获取结果
                                        logical_error_rate = benchmark.run(is_unique_decode = False)[0]
                                        total_running_time = benchmark.decoder_function.total_running_time()
                                        logger.info(f"Logical Error Rate for {code_task}, nkd={nkd}, p={p}, noise_model={noise_model}, decoder={decoder_method}: {logical_error_rate}, total_running_time: {total_running_time}")

                                        # 将结果写入CSV文件
                                        writer.writerow({
                                            'code_task': code_task,
                                            'nkd': nkd,
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
# nohup python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/approx_param_varying_qldpc_code_acc_cpp.py > /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/log/approx_param_varying_qldpc_code_acc_cpp_output.log 2>&1 &
# 607372

# ps aux | grep 'python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/approx_param_varying_qldpc_code_acc_cpp.py'
# pkill -f "python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/approx_param_varying_qldpc_code_acc_cpp.py"