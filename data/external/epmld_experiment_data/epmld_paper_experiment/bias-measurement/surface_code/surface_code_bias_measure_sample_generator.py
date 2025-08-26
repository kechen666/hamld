import os
from hamld.benchmark.utility import generate_bias_measure_syndrome_and_observables

# 目前的si1000通过替换的方式进行生成，初始的数据参考google24的数据。

def save_sample_values(output_file_path,b8_output):
    # 将b8_output写入文件
    with open(output_file_path, 'wb') as f:
        f.write(b8_output)

def main():
    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    distances = [3, 5, 7, 9]
    # 固定p为10/10000，这是目前最先进超导硬件中数量级。
    # probabilities = [10]
    # noise_models = ["si1000"]
    # noise_models = ["si1000", "depolarization"]
    # probabilities = [10, 30]
    probabilities = [10]
    noise_models = ["si1000"]
    have_stabilizers = [False]

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    is_biased = True
    biased_params = [0.25, 0.5, 1.0, 2.0]
    
    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for have_stabilizer in have_stabilizers:
            for d in distances:
                rounds = [1]
                for p in probabilities:
                    for r in rounds:
                        for noise_model in noise_models:
                            for biased_param in biased_params:
                                # 输入为stim文件，输出存储为dat文件。
                                if is_biased:
                                    if have_stabilizer:
                                        output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_sample_{noise_model}_p{p}_bias{biased_param}.dat")
                                    else:
                                        output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_sample_{noise_model}_p{p}_no_stabilizer_bias{biased_param}.dat")
                                else:
                                    if have_stabilizer:
                                        output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_sample_{noise_model}_p{p}.dat")
                                    else:
                                        output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_sample_{noise_model}_p{p}_no_stabilizer.dat")
                                error_type = folder
                                
                                if is_biased:
                                    # 生成偏置的测量结果
                                    p_00 = 1 - float(p/10000) * 5
                                    p_11 = 1 - float(p/10000) * 5 * biased_param
                                else:
                                    p_00 = 1
                                    p_11 = 1
                                
                                print(f"p_00: {p_00},  p_11:{p_11}")
                                b8_output = generate_bias_measure_syndrome_and_observables(d, r, p, noise_model, error_type, current_file_directory, have_stabilizer,p_00=p_00, p_11=p_11)
                                save_sample_values(output_file_path, b8_output)
                                
                                print(f"Circuit sample saved to: {output_file_path}")

if __name__ == "__main__":
    main()