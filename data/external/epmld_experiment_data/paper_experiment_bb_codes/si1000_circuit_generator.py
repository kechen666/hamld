import os
import stim

def replace_values_to_generator_circuit(input_file_path, output_file_path, p, r, have_stabilizer, nkd):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 设置噪声参数
    content = content.replace('X_ERROR(0.002)', f'DEPOLARIZE1({p*2})')
    content = content.replace('DEPOLARIZE1(0.002)', f'DEPOLARIZE1({p*2})')
    content = content.replace('DEPOLARIZE1(0.0001)', f'DEPOLARIZE1({p/10})')
    content = content.replace('M(0.005)', f'M({p*5})')
    content = content.replace('DEPOLARIZE2(0.001)', f'DEPOLARIZE2({p})')
    # 设置轮次
    stim_circuit = stim.Circuit(content)
    n = r - 1
    stim_circuit_r = stim_circuit.copy()
    if n == 0:
        for i in range(len(stim_circuit)):
            if isinstance(stim_circuit[i], stim.CircuitRepeatBlock):
                stim_circuit_r.pop(i)
    elif n >= 1:
        for i in range(len(stim_circuit)):
            if isinstance(stim_circuit[i], stim.CircuitRepeatBlock):
                REPEAT_part = stim_circuit_r.pop(i).body_copy()
                REPEAT_part_n = REPEAT_part * n
                stim_circuit_r.insert(i, REPEAT_part_n)

    if have_stabilizer:
        pass
    else:
        for i in range(int(nkd[0]/2)):
            stim_circuit_r.pop(-(nkd[1]+1))

    new_content = str(stim_circuit_r)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

def main():
    code_tasks = ["bivariate_bicycle_code:rotated_memory_x", "bivariate_bicycle_code:rotated_memory_z"]
    # distances = [3, 5]
    nkds = [[72, 12, 6], [90, 8, 10], [108, 8, 10], [144, 12, 12]]
    probabilities = [5, 10, 20]
    noise_models = ["si1000"]
    have_stabilizer = False
    
    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for nkd in nkds:
            current_file_directory = os.path.dirname(os.path.abspath(__file__))
            for p in probabilities:
                r = 1
                # d = nkd[2]
                # r = d
                for noise_model in noise_models:
                    if p!=10 or r != nkd[2]:
                        # p=10，此时为输入。
                        # 使用d=r的进行生成。
                        input_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{nkd[2]}", f"circuit_noisy_{noise_model}_p10.stim")
                        if have_stabilizer:
                            output_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}.stim")
                        else:
                            output_file_path = os.path.join(current_file_directory, folder, f"nkd_{nkd[0]}_{nkd[1]}_{nkd[2]}_r{r}", f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        
                        p_value = float(p / 10000)
                        replace_values_to_generator_circuit(input_file_path, output_file_path, p_value, r, have_stabilizer, nkd)
                        print(f"Circuit saved to: {output_file_path}")
                    
if __name__ == "__main__":
    main()