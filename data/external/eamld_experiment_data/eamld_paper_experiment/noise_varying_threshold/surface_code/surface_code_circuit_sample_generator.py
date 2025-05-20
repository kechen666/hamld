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
    # distances = [3, 5, 7]
    # probabilities = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 范围值偏大
    # probabilities = [0.01, 0.1, 0.5, 1, 5, 10, 20, 30]
    # probabilities = [0.01, 0.1, 0.5 , 1, 10, 20, 40, 60, 80, 100, 200, 500, 1000] #更新更大范围，研究噪声对于方法的影响
    
    # probabilities = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1] # 减小噪声参数值，寻找阈值
    
    distances = [3, 5, 7]
    probabilities = [1,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150]
    have_stabilizers = [True]
    
    noise_models = ["si1000"]
    # have_stabilizers = [False]

    # 采样次数，与逻辑错误率相关，至少要大于逻辑错误率的负一次方，这里取10**5
    shots = 10**7

    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for have_stabilizer in have_stabilizers:
            for d in distances:
                # rounds = [1, d]
                rounds = [1]
                # rounds = [1]
                # rounds = [d]
                for p in probabilities:
                    for r in rounds:
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