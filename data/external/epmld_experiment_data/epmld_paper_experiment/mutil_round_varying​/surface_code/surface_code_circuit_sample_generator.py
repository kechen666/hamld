import os
import pathlib
import stim
import tempfile

# 目前的si1000通过替换的方式进行生成，初始的数据参考google24的数据。

def generate_sample_values(input_file_path, output_file_path, shots):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    circ = stim.Circuit(content)
    with tempfile.TemporaryDirectory() as d:
        path = output_file_path
        circ.compile_detector_sampler().sample_write(shots=shots, filepath=path, format="b8", append_observables=True)

def main():
    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    # distances = [3, 5, 7, 9]
    distances = [3]
    # 固定p为10/10000，这是目前最先进超导硬件中数量级。
    probabilities = [10]
    noise_models = ["si1000"]
    have_stabilizers = [False]

    # 采样次数，与逻辑错误率相关，至少要大于逻辑错误率的负一次方，这里取10**5
    shots = 10**5

    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for have_stabilizer in have_stabilizers:
            for d in distances:
                rounds = [1, d, int(d*2), int(d*3), int(d*30)]
                for r in rounds:
                    for p in probabilities:
                        for noise_model in noise_models:

                            # 输入为stim文件，输出存储为dat文件。
                            if have_stabilizer:
                                input_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p{p}.stim")
                                output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_sample_{noise_model}_p{p}.dat")
                            else:
                                input_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                                output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_sample_{noise_model}_p{p}_no_stabilizer.dat")
                        
                            generate_sample_values(input_file_path, output_file_path, shots)
                            
                            print(f"Circuit sample saved to: {output_file_path}")

if __name__ == "__main__":
    main()