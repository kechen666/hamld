import os
import logging
from datetime import datetime
import csv
import hamld
# import pymatching
from stimbposd import BPOSD
from hamld.benchmark import generate_qldpc_detector_error_model, LogicalErrorRateBenchmark
# 设置 logging 配置，放在模块级别
from hamld.logging_config import setup_logger

logger = setup_logger("epmld_paper_experiment/qldpc_acc_with_data_qubits", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/noise_varying_threshold/qldpc_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/epmld/experiment/epmld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"qldpc_acc_with_data_qubits_results_{timestamp}.csv")
    # output_file = os.path.join(output_dir, f"qldpc_acc_with_data_qubits_results.csv")

    # 设置QLDPC code的相关初始化参数
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    # nkds = [[72, 12, 6]]
    # probabilities = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 范围值偏大
    # probabilities = [0.01, 0.1, 0.5, 1, 5, 10, 20, 30]
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [True]

    # 设置待测试的解码方法：
    # decoder_methods = ["BP+OSD", "HAMLD"]  # BP+OSD与近似的MLD方法（HAMLD）的对比。
    decoder_methods = ["BP+OSD", "HAMLD"]  # 只测试近似的MLD方法（HAMLD）。
    # 近似的MLD方法的初始化参数：
    # 注意，approximatestrategy 与 approximate_param 是HAMLD的参数，与BP+OSD无关。
    approximatestrategy = "hyperedge_topk"
    approximate_param = 100
    # 超图截取的优先级，超图截取的最大数量
    priority = -2
    priority_topk = 150

    # BP+OSD 初始化参数
    max_bp_iters = 100

    # 创建存储结果的CSV文件并写入表头
    fieldnames = ['code_task', 'nkd', 'round', 'probability', 'noise_model', 'decoder_method', 'logical_error_rate', 'have_stabilizer']
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
                        rounds = [1]
                        # rounds = [d]
                        # rounds = [d]
                        for r in rounds:
                            for noise_model in noise_models:
                                for decoder_method in decoder_methods:
                                    logger.info(f"Running for {code_task}, nkd={nkd}, r={r}, probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")
                                    if decoder_method == "BP+OSD" and r == d:
                                        if nkd !=[72, 12, 6]:
                                            logger.info(f"BP+OSD only support round = d, skip this round.")
                                            continue
                                    # 只跑HAMLD

                                    # 获取对应的检测器错误率模型，用于构建解码器
                                    dem = generate_qldpc_detector_error_model(nkd=nkd, r=r, p=p, noise_model=noise_model, 
                                                                              error_type=error_type,related_path=related_path,
                                                                              have_stabilizer = have_stabilizer)
                                    # 构建对应的解码器
                                    if decoder_method == "BP+OSD":
                                        # 使用BP+OSD的解码器
                                        decoder  = BPOSD(dem, max_bp_iters = max_bp_iters)
                                    elif decoder_method == "HAMLD":
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
                                        'nkd': nkd,
                                        'round': r,
                                        'probability': p,
                                        'noise_model': noise_model,
                                        'decoder_method': decoder_method,
                                        'logical_error_rate': logical_error_rate,
                                        'have_stabilizer': have_stabilizer
                                    })
    
    logger.info(f"Finish the benchmark process.")

if __name__ == "__main__":
    main()

# 后台运行命令
# nohup python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/qldpc_acc_with_data_qubits.py > /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/log/qldpc_acc_with_data_qubits_output.log 2>&1 &
# 3322681