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
    # code_tasks = ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z"]
    # # probabilities = [5, 10, 20]
    # # d_list = [3, 5, 7, 9]
    # probabilities = [5, 10, 20]
    # d_list = [3, 5, 7, 9, 11]
    # noise_models = ["si1000"]

    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    # distances = [3, 5]
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    probabilities = [5, 10, 20]
    noise_models = ["si1000"]
    have_stabilizer = False
    # rounds = [1, d]
    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for nkd in nkds:
            for p in probabilities:
                # for r in [1, d]:
                r = 1
                # d = nkd[2]
                # r = d
                # if p!=10:
                for noise_model in noise_models:
                    print("nkd, r, p, noise_model", nkd, r, p, noise_model)
                    # p=10，此时为输入。
                    current_file_directory = os.path.dirname(os.path.abspath(__file__))
                    # input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}.stim")
                    if have_stabilizer:
                        input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}.stim")
                        output_file_path = os.path.join(current_file_directory, folder,  f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"detector_error_model_{noise_model}_p{p}.dem")
                    else:
                        input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                        output_file_path = os.path.join(current_file_directory, folder,  f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"detector_error_model_{noise_model}_p{p}_no_stabilizer.dem")
                    # decomposed_output_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"decomposed_detector_error_model_{noise_model}_p{p}.dem")
                
                    generate_detector_error_model(input_file_path, output_file_path)
                    # generate_decomposed_detector_error_model(input_file_path, decomposed_output_file_path)

if __name__ == "__main__":
    main()