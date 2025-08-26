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
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    # 固定p为10/10000，这是目前最先进超导硬件中数量级。
    # probabilities = [10]
    # noise_models = ["si1000"]
    # probabilities = [10, 30]
    probabilities = [10]
    noise_models = ["si1000"]
    # 只考虑没有数据比特信息的场景
    have_stabilizers = [False]
    
    measurement_params = [0, 1, 5, 10]
    
    # 采样次数，与逻辑错误率相关，至少要大于逻辑错误率的负一次方，这里取10**5
    shots = 10**5

    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for have_stabilizer in have_stabilizers:
            for nkd in nkds:
                current_file_directory = os.path.dirname(os.path.abspath(__file__))
                for p in probabilities:
                    d = nkd[2]
                    for r in [1]:
                        for noise_model in noise_models:
                            for measurement_param in measurement_params:
                                current_file_directory = os.path.dirname(os.path.abspath(__file__))
                                if have_stabilizer:
                                    input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}_m{measurement_param}.stim")
                                    output_file_path = os.path.join(current_file_directory, folder,  f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_sample_{noise_model}_p{p}_m{measurement_param}.dat")
                                else:
                                    input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}_m{measurement_param}_no_stabilizer.stim")
                                    output_file_path = os.path.join(current_file_directory, folder,  f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_sample_{noise_model}_p{p}_m{measurement_param}_no_stabilizer.dat")
                            
                                generate_sample_values(input_file_path, output_file_path, shots = shots)
                                
                                print(f"Circuit sample saved to: {output_file_path}")

if __name__ == "__main__":
    main()