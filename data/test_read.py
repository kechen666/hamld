import stim
import os

code_task: str = "surface_code:rotated_memory_z"
d: int = 5
r: int = 10
acdp: float = 0.001
brddp: float = 0.001
bmfp: float = 0.005
arfp: float = 0.0001

current_file_directory = os.path.dirname(os.path.abspath(__file__))
task_name = code_task.replace(":", "_")
file_name = f"{task_name}_d{d}_r{r}_acdp{acdp}_arfp{arfp}_bmfp{bmfp}_brddp{brddp}.stim"
file_path = os.path.join(current_file_directory, "raw", file_name)

print(file_path)

# 读取文件内容
with open(file_path, 'r') as file:
    noisy_circuit = file.read()

# print(f"noisy_circuit: {noisy_circuit}")

stim_noisy_circuit = stim.Circuit(noisy_circuit)
detector_error_model = stim_noisy_circuit.detector_error_model()

# print(f"detector_error_model: {detector_error_model}")

svg_content = stim_noisy_circuit.diagram('timeline-svg-html')

# 将SVG图形保存到文件
print(type(svg_content))
print(svg_content)
# output_path = "quantum_circuit.txt"
# with open(output_path, 'w') as file:
#     file.write(str(svg_content))