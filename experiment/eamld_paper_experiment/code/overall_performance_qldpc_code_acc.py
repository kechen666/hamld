import os
import logging
from datetime import datetime
import csv
import eamld
# import pymatching
from stimbposd import BPOSD
from eamld.benchmark import generate_qldpc_detector_error_model, LogicalErrorRateBenchmark
# 设置 logging 配置，放在模块级别
from eamld.logging_config import setup_logger

logger = setup_logger("eamld_paper_experiment/overall_performance_qldpc_code_acc", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    related_path = "/home/normaluser/ck/eamld/data/external/eamld_experiment_data/eamld_paper_experiment/overall_performance/qldpc_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/eamld/experiment/eamld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"overall_performance_qldpc_code_acc_results_{timestamp}.csv")

    # 设置QLDPC code的相关初始化参数
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    # nkds = [[72, 12, 6]]
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]

    # 设置待测试的解码方法：
    # decoder_methods = ["BP+OSD", "EAMLD", "EAPMLD"]  # 只有这三种方法，能够在QLDPC 中运行。
    # 移除有关并行的处理
    decoder_methods = ["BP+OSD", "EAMLD"]
    # decoder_methods = ["BP+OSD"]

    # 近似的MLD方法的初始化参数：
    approximatestrategy = "hyperedge_topk"
    approximate_params = [100]
    # 超图截取的优先级，超图截取的最大数量
    prioritys = [-2]
    priority_topks = [150]


    # BP+OSD 初始化参数
    max_bp_iters = 100

    # 创建存储结果的CSV文件并写入表头
    fieldnames = ['code_task', 'detector_number', 'nkd', 'round', 'approximate_param', 'priority', 'priority_topk', 'decoder_method', 'logical_error_rate', 'have_stabilizer']
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
                        rounds = [1, d]
                        # rounds = [d]
                        for r in rounds:
                            for noise_model in noise_models:
                                for decoder_method in decoder_methods:
                                    for priority in prioritys:
                                        for priority_topk in priority_topks:
                                            for approximate_param in approximate_params:
                                                if decoder_method == "BP+OSD":
                                                    # 对于r=d情况下的BP+OSD解码，只测量最小规模的情况。
                                                    if r == d and nkd[0] != 72:
                                                        logger.info("BP+OSD does only support [72,12,6] r=d.")
                                                        continue
                                                    
                                                detector_number = int(nkd[0]/2 + nkd[0] * (r-1))
                                                logger.info(f"Running for {code_task}, nkd={nkd}, r={r}, probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")
                                                logger.info(f"approximate_param={approximate_param}, priority={priority}, priority_topk={priority_topk}")
                                                # 获取对应的检测器错误率模型，用于构建解码器
                                                dem = generate_qldpc_detector_error_model(nkd=nkd, r=r, p=p, noise_model=noise_model, 
                                                                                        error_type=error_type,related_path=related_path,
                                                                                        have_stabilizer = have_stabilizer)
                                                if decoder_method == "BP+OSD":
                                                    decoder = BPOSD(dem, max_bp_iters=max_bp_iters)
                                                elif decoder_method == "EAMLD":
                                                    # 构建对应的解码器
                                                    decoder =  eamld.EAMLD(detector_error_model=dem,
                                                                                order_method='greedy',
                                                                                slice_method='no_slice',
                                                                                use_approx = True,
                                                                                approximatestrategy = approximatestrategy,
                                                                                approximate_param = approximate_param,
                                                                                contraction_code = "eamld",
                                                                                accuracy = "float64",
                                                                                priority = priority,
                                                                                priority_topk = priority_topk)
                                                # elif decoder_method == "EAPMLD":
                                                #     # 构建对应的解码器
                                                #     stabilizer_number = nkd[0]
                                                #     decoder =  eamld.EAMLD(detector_error_model=dem,
                                                #                                 order_method='greedy',
                                                #                                 slice_method='no_slice',
                                                #                                 use_approx = True,
                                                #                                 approximatestrategy = approximatestrategy,
                                                #                                 approximate_param = approximate_param,
                                                #                                 contraction_code = "parallel-rounds-qlpdc",
                                                #                                 accuracy = "float64",
                                                #                                 priority = priority,
                                                #                                 priority_topk = priority_topk,
                                                #                                 r = r,
                                                #                                 m = m,
                                                #                                 n = stabilizer_number)
                                                
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
                                                    have_stabilizer = have_stabilizer
                                                )

                                                # 运行基准测试并获取结果
                                                logical_error_rate = benchmark.run()[0]
                                                logger.info(f"Logical Error Rate for {code_task}, nkd={nkd}, p={p}, noise_model={noise_model}, decoder={decoder_method}: {logical_error_rate}")

                                                # 将结果写入CSV文件
                                                writer.writerow({
                                                    'code_task': code_task,
                                                    'detector_number': detector_number,
                                                    'nkd': nkd,
                                                    'round': r,
                                                    'approximate_param': approximate_param,
                                                    'priority': priority,
                                                    'priority_topk': priority_topk,
                                                    'decoder_method': decoder_method,
                                                    'logical_error_rate': logical_error_rate,
                                                    'have_stabilizer': have_stabilizer
                                                })
    
    logger.info(f"Finish the benchmark process.")

if __name__ == "__main__":
    main()

# 后台运行命令
# nohup python /home/normaluser/ck/eamld/experiment/eamld_paper_experiment/code/overall_performance_qldpc_code_acc.py > /home/normaluser/ck/eamld/experiment/eamld_paper_experiment/log/overall_performance_qldpc_code_acc_output.log 2>&1 &
# 3061821

# 跑一个r=d的bposd，大概为4个小时以上
# 1950827
