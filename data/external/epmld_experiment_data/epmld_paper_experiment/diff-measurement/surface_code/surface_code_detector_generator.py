import os
import pathlib
import stim
import tempfile
import os

# 目前的si1000通过替换的方式进行生成，初始的数据参考google24的数据。

def generate_detector_error_model(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    circ = stim.Circuit(content)
    with tempfile.TemporaryDirectory() as d:
        path = output_file_path
        dem = circ.detector_error_model(flatten_loops=True, decompose_errors= False)
        with open(path, 'w') as f:
            dem.to_file(f)

def generate_decomposed_detector_error_model(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    circ = stim.Circuit(content)
    with tempfile.TemporaryDirectory() as d:
        path = output_file_path
        dem = circ.detector_error_model(flatten_loops=True, decompose_errors= True)
        with open(path, 'w') as f:
            dem.to_file(f)

def main():
    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    distances = [3, 5, 7, 9]
    # 固定p为10/10000，这是目前最先进超导硬件中数量级。
    # probabilities = [10]
    # noise_models = ["si1000", "depolarization"]
    # probabilities = [10, 30]
    probabilities = [10]
    noise_models = ["si1000"]
    have_stabilizers = [False]
    
    measurement_params = [0, 1, 5, 10]
    
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for have_stabilizer in have_stabilizers:
            for d in distances:
                rounds = [1]
                for p in probabilities:
                    for r in rounds:
                        for noise_model in noise_models:
                            for measurement_param in measurement_params:
                                print("d,r,p,noise_model", d, r, p, noise_model)
                                
                                # 对应stim线路作为输入，进行生成对应的dem文件，其中包含了是否decomposed两种detector。
                                if have_stabilizer:
                                    input_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p{p}_m{measurement_param}.stim")

                                    output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"detector_error_model_{noise_model}_p{p}_m{measurement_param}.dem")
                                    decomposed_output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"decomposed_detector_error_model_{noise_model}_p{p}_m{measurement_param}.dem")
                                else:
                                    input_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p{p}_m{measurement_param}_no_stabilizer.stim")

                                    output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"detector_error_model_{noise_model}_p{p}_m{measurement_param}_no_stabilizer.dem")
                                    decomposed_output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"decomposed_detector_error_model_{noise_model}_p{p}_m{measurement_param}_no_stabilizer.dem")

                                # for MLD
                                generate_detector_error_model(input_file_path, output_file_path)
                                # for MWPM
                                generate_decomposed_detector_error_model(input_file_path, decomposed_output_file_path)

if __name__ == "__main__":
    main()