import os
import logging
from datetime import datetime
import csv
import hamld
import pymatching
# from stimbposd import BPOSD
from hamld.benchmark import generate_detector_error_model, DecoderSpeedBenchmark
# 设置 logging 配置，放在模块级别
from hamld.logging_config import setup_logger

logger = setup_logger("epmld_paper_experiment/noisy_varying_surface_code_speed", log_level=logging.INFO)

def main():
    # 设置输入数据的相关路径
    related_path = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/noise_varying/surface_code"

    # 设置输出目录
    output_dir="/home/normaluser/ck/epmld/experiment/epmld_paper_experiment/result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"noisy_varying_surface_code_speed_results_{timestamp}.csv")
    # output_file = os.path.join(output_dir, f"noisy_varying_qldpc_code_acc_results.csv")

    # 设置Surface code的相关初始化参数
    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    # distances = [3, 5, 7, 9]
    distances = [9]
    # probabilities = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 范围值偏大
    # probabilities = [0.01, 0.1, 0.5, 1, 5, 10, 20, 30]
    probabilities = [0.1, 1, 10, 100]
    # probabilities = [0.001, 0.1, 1, 10, 100]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]

    # 设置待测试的解码方法：
    # decoder_methods = ["MWPM", "HAMLD", "EMLD"]  # 在r=1的情况下，MWPM, 近似的MLD方法（HAMLD）, 无损的HAMLD，理论上是能够跑完的，10**5。
    # 在surface code中，只有在r=d的情况下，才会比较明显的看出HAMLD的优势。
    decoder_methods = ["MWPM", "HAMLD"]  # r=d的情况下，无损的HAMLD，也是一个非常大的指数，只有在d=3的时候才能跑，因为这里我们只考虑HAMLD，理论上是能够跑完的，10**5。
    # decoder_methods = ["HAMLD"]
    
    # 近似的MLD方法的初始化参数：
    # 注意，approximatestrategy 与 approximate_param 是HAMLD的参数，与BP+OSD无关。
    approximatestrategy = "hyperedge_topk"
    approximate_param = 100
    # 超图截取的优先级，超图截取的最大数量
    priority = -2
    priority_topk = 150


    # 创建存储结果的CSV文件并写入表头
    fieldnames = ['code_task', 'd', 'round', 'probability', 'noise_model', 'decoder_method', 'average_per_round_time', 'have_stabilizer']
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
                                    logger.info(f"Running for {code_task}, d={d}, r={r},probability={p}, noise_model={noise_model}, decoder_method={decoder_method}")

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
                                    elif decoder_method == "EMLD":
                                        # 使用HAMLD解码器，这是是无损的HAMLD解码器。
                                        # cpp
                                        decoder = hamld.HAMLD(detector_error_model=dem,
                                                              order_method='greedy',
                                                              slice_method='no_slice',
                                                              contraction_code = 'cpp',
                                                              accuracy = "float64")
                                        
                                    elif decoder_method == "HAMLD":
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
                                    
                                    # 创建基准测试对象
                                    benchmark = DecoderSpeedBenchmark(
                                        decoder_function=decoder,
                                        d=d,
                                        nkd=None,
                                        r=r,
                                        p=p,
                                        noise_model=noise_model,
                                        error_type=error_type,
                                        num_runs=10,
                                        data_path=related_path,
                                        code_name="surface code",
                                        have_stabilizer = have_stabilizer
                                    )

                                    # 运行基准测试并获取结果
                                    average_per_round_time = benchmark.run(shots=10)[0]
                                    logger.info(f"Per syndrome decoding time for {code_task}, d={d}, p={p}, noise_model={noise_model}, decoder={decoder_method}: {average_per_round_time}")

                                    # 将结果写入CSV文件
                                    writer.writerow({
                                        'code_task': code_task,
                                        'd': d,
                                        'round': r,
                                        'probability': p,
                                        'noise_model': noise_model,
                                        'decoder_method': decoder_method,
                                        'average_per_round_time': average_per_round_time,
                                        'have_stabilizer': have_stabilizer
                                    })

    logger.info(f"Finish the benchmark process.")

if __name__ == "__main__":
    main()

# 后台运行命令
# nohup python /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/code/noisy_varying_surface_code_speed.py > /home/normaluser/ck/epmld/experiment/epmld_paper_experiment/log/noisy_varying_surface_code_speed_output.log 2>&1 &
# 545382