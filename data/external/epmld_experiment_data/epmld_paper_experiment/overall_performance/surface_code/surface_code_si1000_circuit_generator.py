import os
import stim
import re

def replace_values_to_generator_circuit(input_file_path, output_file_path, p, r, have_stabilizer, d):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 设置噪声参数
    content = content.replace('X_ERROR(0.002)', f'DEPOLARIZE1({p*2})')
    content = content.replace('DEPOLARIZE1(0.002)', f'DEPOLARIZE1({p*2})')
    content = content.replace('DEPOLARIZE1(0.0001)', f'DEPOLARIZE1({p/10})')
    # content = content.replace('M(0.005)', f'M({p*5})')
    # 更新辅助比特测量参数，数据比特不进行测量。
    m_pattern = 'M(0.005)'
    if m_pattern in content:
        # 分割字符串为最后一个匹配前的部分、匹配部分和匹配后的部分
        before_last, sep, after_last = content.rpartition(m_pattern)
        
        # 替换前面所有的M(0.005)
        before_last = before_last.replace(m_pattern, f'M({p*5})')
        
        # 组合字符串，将最后一个匹配替换为M
        content = before_last + 'M' + after_last
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

    # surface code 中的数据比特的stabilizer信息，进行删除
    if have_stabilizer:
        pass
    else:
        # 从倒数第三行开始，删除（d**2-1）/2行
        for i in range(int((d**2-1)/2)):
            stim_circuit_r.pop(-3)

    new_content = str(stim_circuit_r)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

def main():
    code_tasks = ["surface_code:rotated_memory_x","surface_code:rotated_memory_z"]
    # distances = [3, 5, 7, 9, 11, 13]
    
    # 固定p为10/10000，这是目前最先进超导硬件中数量级。
    probabilities = [10]
    noise_models = ["si1000"]
    have_stabilizers = [False]
    distances = [15, 17, 19]
    
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    for code_task in code_tasks:
        folder = "Z" if "memory_z" in code_task else "X" if "memory_x" in code_task else "other"
        for have_stabilizer in have_stabilizers:
            for d in distances:
                rounds = [1, d]
                for p in probabilities:
                    for r in rounds:
                        for noise_model in noise_models:
                            if p == 10 and r == d and have_stabilizer==True:
                                # p=10, r=d，have_stabilizer = True，这个文件是作为输入，进行生成其他文件。
                                continue

                            input_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p10.stim")
                            if have_stabilizer:
                                output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p{p}.stim")
                            else:
                                output_file_path = os.path.join(current_file_directory, folder, f"d{d}_r{r}", f"circuit_noisy_{noise_model}_p{p}_no_stabilizer.stim")
                            
                            # Ensure the output directory exists
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                            
                            # 替换，得到对应的stim文件
                            p_value = float(p / 10000)
                            replace_values_to_generator_circuit(input_file_path, output_file_path, p_value, r, have_stabilizer, d)
                            print(f"Circuit saved to: {output_file_path}")
                    
if __name__ == "__main__":
    main()