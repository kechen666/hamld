import os
import logging
from datetime import datetime
import csv
import hamld
# import pymatching
from stimbposd import BPOSD
from hamld.benchmark import generate_qldpc_detector_error_model, DecoderSpeedBenchmark
# 设置 logging 配置，放在模块级别
from hamld.logging_config import setup_logger

logger = setup_logger("epmld_paper_experiment/approx_param_varying_qldpc_code_speed", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    # related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/approx_param_varying/qldpc_code"
    related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/qldpc_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/epmld/experiment/epmld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"approx_param_varying_qldpc_code_speed_results_{timestamp}.csv")
    # output_file = os.path.join(output_dir, f"noisy_varying_qldpc_code_acc_results.csv")

    # 设置QLDPC code的相关初始化参数
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    # nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    nkds = [[72, 12, 6], [144, 12, 12]]
    # 固定噪声参数和噪声模型：
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]

    # 设置待测试的解码方法：
    decoder_methods = ["HAMLD"] # "qldpc-new-priority"就是
    # 近似的MLD方法的初始化参数：
    approximatestrategy = "hyperedge_topk"
    # approximate_params = [10, 100, 500, 1000]
    # # 超图截取的优先级，超图截取的最大数量
    # prioritys = [-3,-2, -1, 0]
    # priority_topks = [10, 50, 100, 150]

    # approximate_params = [10, 100, 1000]
    # # 超图截取的优先级，超图截取的最大数量
    # prioritys = [0,-2,-3]
    # priority_topks = [10, 150, 200]
    # approximate_params = [10, 100, 500,1000]
    # # 超图截取的优先级，超图截取的最大数量
    # prioritys = [0,-1,-2,-3]
    # priority_topks = [10, 100, 150, 200]
    approximate_params = [100, 500, 1000]
    prioritys = [0, 0, 0, 0]
    priority_topks = [100, 500, 800, 1000]

    # 创建存储结果的CSV文件并写入表头
    fieldnames = ['code_task', 'nkd', 'round', 'approximate_param', 'priority', 'priority_topk', 'decoder_method', 'average_per_round_time', 'have_stabilizer']
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
                        rounds = [d]
                        for r in rounds:
                            for noise_model in noise_models:
                                for decoder_method in decoder_methods:
                                    for priority_index in range(len(priority_topks)):
                                        priority = prioritys[priority_index]
                                        priority_topk = priority_topks[priority_index]
                                    # for priority in prioritys:
                                        # for priority_topk in priority_topks:
                                        for approximate_param in approximate_params:
                                            logger.info(f"Running for {code_task}, d={d}, r={r},probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")
                                            logger.info(f"approximate_param={approximate_param}, priority={priority}, priority_topk={priority_topk}")
                                            # 获取对应的检测器错误率模型，用于构建解码器
                                            dem = generate_qldpc_detector_error_model(nkd=nkd, r=r, p=p, noise_model=noise_model, 
                                                                                    error_type=error_type,related_path=related_path,
                                                                                    have_stabilizer = have_stabilizer)

                                            decoder =  hamld.HAMLD(detector_error_model=dem,
                                                                        order_method='greedy',
                                                                        slice_method='no_slice',
                                                                        use_approx = True,
                                                                        approximatestrategy = approximatestrategy,
                                                                        approximate_param = approximate_param,
                                                                        contraction_code = "qldpc-new-priority",
                                                                        accuracy = "float64",
                                                                        priority = priority,
                                                                        priority_topk = priority_topk)
                                            
                                            # 创建基准测试对象
                                            benchmark = DecoderSpeedBenchmark(
                                                decoder_function=decoder,
                                                d=nkd[2],
                                                nkd=nkd,
                                                r=r,
                                                p=p,
                                                noise_model=noise_model,
                                                error_type=error_type,
                                                num_runs=10,
                                                data_path=related_path,
                                                code_name="qldpc code",
                                                have_stabilizer = have_stabilizer
                                            )

                                            # 运行基准测试并获取结果
                                            average_per_round_time = benchmark.run(shots=10)[0]
                                            logger.info(f"Logical Error Rate for {code_task}, nkd={nkd}, p={p}, noise_model={noise_model}, decoder={decoder_method}: {average_per_round_time}")

                                            # 将结果写入CSV文件
                                            writer.writerow({
                                                'code_task': code_task,
                                                'nkd': nkd,
                                                'round': r,
                                                'approximate_param': approximate_param,
                                                'priority': priority,
                                                'priority_topk': priority_topk,
                                                'decoder_method': decoder_method,
                                                'average_per_round_time': average_per_round_time,
                                                'have_stabilizer': have_stabilizer
                                            })
    
    logger.info(f"Finish the benchmark process.")

if __name__ == "__main__":
    main()

# 后台运行命令
# nohup python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/approx_param_varying_qldpc_code_speed.py > /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/log/approx_param_varying_qldpc_code_speed_output.log 2>&1 &
# 3297694