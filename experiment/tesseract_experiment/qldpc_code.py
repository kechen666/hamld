import os

if __name__ == "__main__":
    # Configuration parameters
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    have_stabilizers = [False]  # Only considering no stabilizer case
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    probabilities = [10]
    noise_models = ["si1000"]
    base_dir = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/qldpc_code"
    tesseract_path = "/home/normaluser/ck/tesseract-decoder/bazel-bin/src/tesseract"

    result_dir = "/home/normaluser/ck/epmld/experiment/tesseract_experiment/qldpc_code_result"

    for code_task in code_tasks:
        # Determine folder (X or Z) based on code task
        folder = "X" if "memory_x" in code_task else "Z"
        
        for have_stabilizer in have_stabilizers:
            for nkd in nkds:
                n, k, d = nkd
                for p in probabilities:
                    for r in [1, d]:
                        current_file_directory = os.path.join(base_dir, folder, f"nkd_{n}_{k}_{d}_r{r}")
                        for noise_model in noise_models:
                            
                            output_folder = os.path.join(result_dir, folder)
                            os.makedirs(output_folder, exist_ok=True)
                            
                            # # Construct file paths
                            # if have_stabilizer:
                            #     # This branch won't execute since have_stabilizers = [False]
                            #     circuit_file = os.path.join(current_file_directory, f"circuit_noisy_{noise_model}_p{p}.stim")
                            #     dem_file = os.path.join(current_file_directory, f"detector_error_model_{noise_model}_p{p}.dem")
                            #     dat_file = os.path.join(current_file_directory, f"circuit_sample_{noise_model}_p{p}.dat")
                            #     output_file = f"{result_dir}/{folder}/decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_with_stabilizer.json"
                                
                            #     output_file_pre = f"{result_dir}/{folder}/decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_with_stabilizer_pre.01"
                            # else:
                            #     circuit_file = os.path.join(current_file_directory, f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                            #     dem_file = os.path.join(current_file_directory, f"detector_error_model_{noise_model}_p{p}_no_stabilizer.dem")
                            #     dat_file = os.path.join(current_file_directory, f"circuit_sample_{noise_model}_p{p}_no_stabilizer.dat")
                            #     output_file = f"{result_dir}/{folder}/decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_no_stabilizer.json"

                            #     output_file_pre = f"{result_dir}/{folder}/decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_no_stabilizer_pre.01"
                            # # Construct the command
                            # command = (
                            #     f"{tesseract_path} --pqlimit 1000000 --circuit {circuit_file} --dem {dem_file} "
                            #     f"--in {dat_file} --in-format b8 --in-includes-appended-observables --out {output_file_pre} --out-format 01 "
                            #     f"--stats-out {output_file} --threads 128 "
                            #     f"--beam 23 --beam_climbing --no-merge-errors"
                            # )
                            if have_stabilizer:
                                circuit_file = os.path.join(current_file_directory, f"circuit_noisy_{noise_model}_p{p}.stim")
                                dem_file = os.path.join(current_file_directory, f"detector_error_model_{noise_model}_p{p}.dem")
                                dat_file = os.path.join(current_file_directory, f"circuit_sample_{noise_model}_p{p}.dat")
                                output_file = os.path.join(output_folder, f"decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_with_stabilizer.json")
                                output_file_pre = os.path.join(output_folder, f"decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_with_stabilizer_pre.01")
                            else:
                                circuit_file = os.path.join(current_file_directory, f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                                dem_file = os.path.join(current_file_directory, f"detector_error_model_{noise_model}_p{p}_no_stabilizer.dem")
                                dat_file = os.path.join(current_file_directory, f"circuit_sample_{noise_model}_p{p}_no_stabilizer.dat")
                                output_file = os.path.join(output_folder, f"decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_no_stabilizer.json")
                                output_file_pre = os.path.join(output_folder, f"decoded_qldpc_{noise_model}_p{p}_n{n}_k{k}_d{d}_r{r}_no_stabilizer_pre.01")

                            # 构造命令并执行
                            # command = (
                            #     f"{tesseract_path} --pqlimit 1000000 --circuit {circuit_file} --dem {dem_file} "
                            #     f"--in {dat_file} --in-format b8 --in-includes-appended-observables --out {output_file_pre} --out-format 01 "
                            #     f"--stats-out {output_file} --threads 128 "
                            #     f"--beam 23 --beam-climbing --no-merge-errors --num-det-orders 1"
                            # )
                            command = (
                                f"{tesseract_path} --pqlimit 1000000 --circuit {circuit_file} --dem {dem_file} "
                                f"--in {dat_file} --in-format b8 --in-includes-appended-observables --out {output_file_pre} --out-format 01 "
                                f"--stats-out {output_file} --threads 128 --beam 23 --num-det-orders 1 "
                                f"--no-merge-errors"
                            )
                            
                            # Print or execute the command
                            print(f"Running: {command}")
                            
                            # Uncomment the following line to actually execute the command
                            os.system(command)
                            
                            print(f"finsihed, nkd: {nkd}, p: {p}, noise_model: {noise_model}, r: {r}, have_stabilizer: {have_stabilizer}")

    print("decoding finsihed")
# nohup python /home/normaluser/ck/epmld/experiment/tesseract_experiment/qldpc_code.py >/home/normaluser/ck/epmld/experiment/tesseract_experiment/log/qldpc_code.log 2>&1 &
# 939213