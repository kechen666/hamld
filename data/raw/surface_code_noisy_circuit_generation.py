import stim
import os
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Generate a surface code circuit with given parameters.")
parser.add_argument("--r", type=int, default=10, help="Number of rounds (default: 10)")
parser.add_argument("--d", type=int, default=5, help="Distance of the code (default: 5)")
parser.add_argument("--acdp", type=float, default=0.001, help="After Clifford depolarization probability (default: 0.001)")
parser.add_argument("--arfp", type=float, default=0.0001, help="After reset flip probability (default: 0.0001)")
parser.add_argument("--bmfp", type=float, default=0.005, help="Before measure flip probability (default: 0.005)")
parser.add_argument("--brddp", type=float, default=0.001, help="Before round data depolarization probability (default: 0.001)")
parser.add_argument("--code_task", type=str, default="surface_code:rotated_memory_z", choices=[
    "repetition_code:memory",
    "surface_code:rotated_memory_x",
    "surface_code:rotated_memory_z",
    "surface_code:unrotated_memory_x",
    "surface_code:unrotated_memory_z",
    "color_code:memory_xyz"
], help="The code task to generate (default: surface_code:rotated_memory_z)")

# 解析命令行参数
args = parser.parse_args()

# 生成 surface_code 电路
surface_code_circuit = stim.Circuit.generated(
    args.code_task,
    rounds=args.r,
    distance=args.d,
    after_clifford_depolarization=args.acdp,
    after_reset_flip_probability=args.arfp,
    before_measure_flip_probability=args.bmfp,
    before_round_data_depolarization=args.brddp
)

# 将电路转换为字符串
circuit_string = str(surface_code_circuit)

# 设置文件保存路径
current_file_directory = os.path.dirname(os.path.abspath(__file__))
task_name = args.code_task.replace(":", "_")

# 文件命名，包括相关参数的值
file_name = f"{task_name}_d{args.d}_r{args.r}_acdp{args.acdp}_arfp{args.arfp}_bmfp{args.bmfp}_brddp{args.brddp}.stim"
# file_name = f"{task_name}_d{args.d}_r{args.r}_acdp{args.acdp}_arfp{args.arfp}_bmfp{args.bmfp}_brddp{args.brddp}.stim"

# 构建文件路径
file_path = os.path.join(current_file_directory, "decoder_circuit" , file_name)

# 保存电路字符串到文件
with open(file_path, 'w') as file:
    file.write(circuit_string)

print(f"Circuit saved to: {file_path}")
