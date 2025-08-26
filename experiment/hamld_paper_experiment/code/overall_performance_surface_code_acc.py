import os
import logging
from datetime import datetime
import csv
import hamld
import pymatching
# from stimbposd import BPOSD
from hamld.benchmark import generate_detector_error_model, LogicalErrorRateBenchmark
# 设置 logging 配置，放在模块级别
from hamld.logging_config import setup_logger

logger = setup_logger("epmld_paper_experiment/overall_performance_surface_code_acc", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/surface_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/epmld/experiment/epmld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"overall_performance_surface_code_acc_results_{timestamp}.csv")

    # 设置Surface code的相关初始化参数
    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    # distances = [3, 5, 7, 9]
    # distances = [3]
    distances = [19]
    # 固定噪声参数和噪声模型：
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]

    # 设置待测试的解码方法：
    # decoder_methods = ["MWPM", "MLD","EMLD", "HAMLD", "EAPMLD"]
    # 移除有关并行的处理
    # decoder_methods = ["MWPM", "MLD", "EMLD", "HAMLD"]
    # decoder_methods = ["MWPM", "MLD", "EMLD"]
    # 测试MWPM和EMLD的准确率，其中MLD
    # decoder_methods = ["MWPM", "EMLD", "MLD"]
    # decoder_methods = ["MLD"]
    decoder_methods = ["MWPM"]
    # 其中MLD只测试d=3，d=5，r=1的情况。
    # EMLD只测试r=1的情况。

    # 近似的MLD方法的初始化参数：
    approximatestrategy = "hyperedge_topk"
    approximate_params = [0]
    # 超图截取的优先级，超图截取的最大数量
    # prioritys = [-2]
    prioritys = [0]
    priority_topks = [0]

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
                        # rounds = [1]
                        # rounds = [1, d]
                        rounds = [d]
                        for r in rounds:
                            for noise_model in noise_models:
                                for decoder_method in decoder_methods:
                                    for priority in prioritys:
                                        for priority_topk in priority_topks:
                                            for approximate_param in approximate_params:
                                                # if decoder_method == "MLD" :
                                                #     if r != 1:
                                                #         logger.info("MLD方法只测试r=1的情况。")
                                                #         continue
                                                    
                                                # if decoder_method == "EMLD":
                                                #     # 对于d>5 并且r>1的情况下，进行跳过。
                                                #     if d >= 5 and r > 1:
                                                #         logger.info("EMLD方法只测试d=3，r=3，以及r=1的情况。")
                                                #         continue
                                                #     continue
                                                # if decoder_method == "MLD" and d not in [3, 5]:
                                                #     # MLD方法只测试d=3，d=5，r=1的情况。
                                                #     logger.info("MLD方法只测试d=3，d=5的情况。")
                                                #     continue
                                                # if decoder_method == "EMLD" and r==d:
                                                #     # EMLD方法只测试r=1的情况。
                                                #     logger.info("EMLD方法只测试r=1的情况。")
                                                #     continue

                                                detector_number = int((d**2-1)/2 + (d**2-1) * (r-1))
                                                logger.info(f"Running for {code_task}, d={d}, r={r},probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")
                                                logger.info(f"approximate_param={approximate_param}, priority={priority}, priority_topk={priority_topk}")

                                                # 获取对应的检测器错误率模型，用于构建解码器
                                                if decoder_method == "MWPM":
                                                    dem = generate_detector_error_model(d = d, r=r, p=p, noise_model=noise_model,
                                                                                        error_type=error_type, decomposed_error = True,
                                                                                        related_path=related_path, have_stabilizer = have_stabilizer)
                                                else:
                                                    dem = generate_detector_error_model(d = d, r=r, p=p, noise_model=noise_model,
                                                                                        error_type=error_type, decomposed_error = False,
                                                                                        related_path=related_path, have_stabilizer = have_stabilizer)
                                                    
                                                # 构建对应的解码器
                                                if decoder_method == "MWPM":
                                                    # 使用MWPM的解码器
                                                    decoder = pymatching.Matching.from_detector_error_model(dem)
                                                elif decoder_method == "MLD":
                                                    # 使用MLD解码器，这是是无损的HAMLD解码器。
                                                    # cpp
                                                    decoder = hamld.HAMLD(detector_error_model=dem,
                                                                        order_method='mld',
                                                                        slice_method='no_slice',
                                                                        contraction_code = 'cpp',
                                                                        accuracy = "float64")
                                                elif decoder_method == "EMLD":
                                                    # 使用HAMLD解码器，这是是无损的HAMLD解码器。
                                                    # cpp
                                                    decoder = hamld.HAMLD(detector_error_model=dem,
                                                                        order_method='greedy',
                                                                        slice_method='no_slice',
                                                                        contraction_code = 'cpp',
                                                                        accuracy = "float64")
                                                elif decoder_method == "HAMLD":
                                                    # 构建对应的解码器
                                                    # 使用HAMLD方法。
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
                                                # elif decoder_method == "EAPMLD":
                                                #     # 构建对应的解码器
                                                #     stabilizer_number = d**2-1
                                                #     decoder =  epmld.HAMLD(detector_error_model=dem,
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
                                                    d=d,
                                                    nkd=None,
                                                    r=r,
                                                    p=p,
                                                    noise_model=noise_model,
                                                    error_type=error_type,
                                                    num_runs=1,
                                                    data_path=related_path,
                                                    code_name="surface code",
                                                    have_stabilizer = have_stabilizer
                                                )

                                                # 运行基准测试并获取结果
                                                logical_error_rate = benchmark.run(is_unique_decode=False)[0]
                                                total_time_seconds = benchmark.total_time_seconds
                                                logger.info(f"Logical Error Rate for {code_task}, d={d}, p={p}, noise_model={noise_model}, decoder={decoder_method}: {logical_error_rate}, total_time_seconds: {total_time_seconds}")

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
                                                    'total_running_time': total_time_seconds
                                                })

    logger.info(f"Finish the benchmark process.")

if __name__ == "__main__":
    main()

# 后台运行命令
# nohup python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/overall_performance_surface_code_acc.py > /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/log/overall_performance_surface_code_acc_output.log 2>&1 &
# 1053585