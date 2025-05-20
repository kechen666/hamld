import stim
import os
# import argparse

# # 创建命令行参数解析器
# parser = argparse.ArgumentParser(description="Generate a surface code circuit with given parameters.")
# parser.add_argument("--r", type=int, default=10, help="Number of rounds (default: 10)")
# parser.add_argument("--d", type=int, default=5, help="Distance of the code (default: 5)")
# # parser.add_argument("--acdp", type=float, default=0.001, help="After Clifford depolarization probability (default: 0.001)")
# # parser.add_argument("--arfp", type=float, default=0.0001, help="After reset flip probability (default: 0.0001)")
# # parser.add_argument("--bmfp", type=float, default=0.005, help="Before measure flip probability (default: 0.005)")
# # parser.add_argument("--brddp", type=float, default=0.001, help="Before round data depolarization probability (default: 0.001)")
# parser.add_argument("--p", type=float, default=0.001, help="physical error rate (default: 0.001)")
# parser.add_argument("--code_task", type=str, default="surface_code:rotated_memory_z", choices=[
#     "repetition_code:memory",
#     "surface_code:rotated_memory_x",
#     "surface_code:rotated_memory_z",
#     "surface_code:unrotated_memory_x",
#     "surface_code:unrotated_memory_z",
#     "color_code:memory_xyz"
# ], help="The code task to generate (default: surface_code:rotated_memory_z)")

# # 解析命令行参数
# args = parser.parse_args()

def generate_experiment_noisy_sd_circuit(code_task, d, r, p, have_stabilizer = False):
    # 生成 surface_code 电路
    surface_code_circuit = stim.Circuit.generated(
        code_task,
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p
    )

    if have_stabilizer:
        pass
    else:
        # 从倒数第三行开始，删除（d**2-1）/2行
        for i in range(int((d**2-1)/2)):
            surface_code_circuit.pop(-2)

    # 将电路转换为字符串
    circuit_string = str(surface_code_circuit)

    # 设置文件保存路径
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    if have_stabilizer:
        file_name = f"circuit_noisy_depolarization_p{int(p*10000)}.stim"
    else:
        file_name = f"circuit_noisy_depolarization_p{int(p*10000)}_no_stabilizer.stim"

    # 根据 code_task 决定文件夹
    if "memory_z" in code_task:
        folder = "Z"
    elif "memory_x" in code_task:
        folder = "X"
    else:
        folder = "other"

    # 构建文件路径
    file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", file_name)

    # 检查文件夹是否存在
    if not os.path.exists(os.path.dirname(file_path)):
        raise FileNotFoundError(f"The directory {os.path.dirname(file_path)} does not exist.")
    
    # 保存电路字符串到文件
    with open(file_path, 'w') as file:
        file.write(circuit_string)

    print(f"Circuit saved to: {file_path}")

if __name__ == "__main__":
    # probabilities = range(10, 201, 10) # v3
    # probabilities = range(200, 1000, 100) # v4
    # probabilities = [1,500, 1000] # v5，添加一个p=1的小值和大值500，1000的情况。
    probabilities = [10] # v6 添加100,200,300,400的情况。
    have_stabilizers = [False, True]
    for code_task in ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]:
        for d in [3,5,7,9]:
            for p in probabilities:
                for have_stabilizer in have_stabilizers:
                    generate_experiment_noisy_sd_circuit(code_task = code_task, d=d, r=1, p=float(p/10000), have_stabilizer=have_stabilizer)