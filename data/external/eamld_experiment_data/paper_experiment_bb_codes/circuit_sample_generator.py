import os
import pathlib
import stim
import tempfile

# 目前的si1000通过替换的方式进行生成，初始的数据参考google24的数据。

def generate_sample_values(input_file_path, output_file_path, p):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    circ = stim.Circuit(content)
    with tempfile.TemporaryDirectory() as d:
        path = output_file_path
        circ.compile_detector_sampler().sample_write(shots=10**6, filepath=path, format="b8", append_observables=True)

def main():
    # 示例用法
    # code_tasks = ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z"]
    # # d_list = [3, 5, 7, 9, 11]
    # # probabilities = probabilities = [1, 10, 20]
    # probabilities = [5, 10, 20]
    # d_list = [3, 5, 7, 9]
    # noise_models = ["si1000"]

    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    # distances = [3, 5]
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    probabilities = [5, 10, 20]
    noise_models = ["si1000"]
    have_stabilizer = False
    
    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for nkd in nkds:
            for p in probabilities:
                r = 1
                
                # d = nkd[2]
                # r = d
                for noise_model in noise_models:
                    # p=10，此时为输入。
                    current_file_directory = os.path.dirname(os.path.abspath(__file__))
                    if have_stabilizer:
                        input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}.stim")
                        output_file_path = os.path.join(current_file_directory, folder,  f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_sample_{noise_model}_p{p}.dat")
                    else:
                        input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                        output_file_path = os.path.join(current_file_directory, folder,  f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_sample_{noise_model}_p{p}_no_stabilizer.dat")
                
                    p_value = float(p / 10000)
                    generate_sample_values(input_file_path, output_file_path, p_value)
                    
                    print(f"Circuit sample saved to: {output_file_path}")

if __name__ == "__main__":
    main()