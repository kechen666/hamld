import os


if __name__ == "__main__":
    # Base directory paths
    current_file_directory = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/surface_code"
    tesseract_path = "/home/normaluser/ck/tesseract-decoder/bazel-bin/src/tesseract"
    result_dir = "/home/normaluser/ck/epmld/experiment/tesseract_experiment/surface_code_result"
    # Configuration parameters

    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    # distances = [3, 5, 7, 9]
    # distances = [11, 13]
    distances = [15, 17, 19]
    # 固定p为10/10000，这是目前最先进超导硬件中数量级。
    probabilities = [10]
    noise_models = ["si1000"]
    have_stabilizers = [False]

    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        
        for have_stabilizer in have_stabilizers:
            for d in distances:
                rounds = [1, d]
                for p in probabilities:
                    for r in rounds:
                        for noise_model in noise_models:
                            
                            output_folder = os.path.join(result_dir, folder)
                            os.makedirs(output_folder, exist_ok=True)
                            
                            # Construct file paths
                            if have_stabilizer:
                                dat_file = f"{current_file_directory}/{folder}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}.dat"
                                dem_file = f"{current_file_directory}/{folder}/d{d}_r{r}/detector_error_model_{noise_model}_p{p}.dem"
                                # dat_file = os.path.join(current_file_directory, f"circuit_sample_{noise_model}_p{p}.dat")
                                output_file = f"{result_dir}/{folder}/decoded_surface_{noise_model}_p{p}_d{d}_r{r}_no_stabilizer.json"
                                
                                
                                output_file_pre = f"{result_dir}/{folder}/decoded_surface_{noise_model}_p{p}_d{d}_r{r}_no_stabilizer.01"
                            else:
                                dat_file = f"{current_file_directory}/{folder}/d{d}_r{r}/circuit_sample_{noise_model}_p{p}_no_stabilizer.dat"
                                dem_file = f"{current_file_directory}/{folder}/d{d}_r{r}/detector_error_model_{noise_model}_p{p}_no_stabilizer.dem"
                                # dat_file = os.path.join(current_file_directory, f"circuit_sample_{noise_model}_p{p}_no_stabilizer.dat")
                                output_file = f"{result_dir}/{folder}/decoded_surface_{noise_model}_p{p}_d{d}_r{r}_no_stabilizer.json"
                                
                                output_file_pre = f"{result_dir}/{folder}/decoded_surface_{noise_model}_p{p}_d{d}_r{r}_no_stabilizer.01"
                            
                            # Construct the command
                            command = (
                                f"{tesseract_path} --pqlimit 1000000 --dem {dem_file} "
                                f"--in {dat_file} --in-format b8 --in-includes-appended-observables --out {output_file_pre} --out-format 01 "
                                f"--stats-out {output_file} --threads 128 --num-det-orders 1 --beam 23 --no-merge-errors"
                            )
                            

                            # Print or execute the command
                            print(f"Running: {command}")
                            
                            # Uncomment the following line to actually execute the command
                            os.system(command)
                            
    print("decoding finsihed")
                        
# nohup python /home/normaluser/ck/epmld/experiment/tesseract_experiment/surface_code.py >/home/normaluser/ck/epmld/experiment/tesseract_experiment/log/surface_code.log 2>&1 &
# 1458071