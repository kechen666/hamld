from typing import List, Optional

class CircuitGenerator:
    """
    A class to generate quantum error correction circuits based on various parameters.
    """
    
    def __init__(self):
        """
        Initialize the CircuitGenerator with specific parameters.
        
        :param code_task: The type of quantum error correction code.
        :param distance: The code distance.
        :param rounds: The number of repetition rounds.
        :param noise_model: The noise model to be used (e.g., SI1000, uniform(sd6)).
        :param p: The noise parameter.
        :param gates: The type of gates used (e.g., cz, all).
        """
        self.code_task: str = "surface_code:rotated_memory_x"  # Default quantum error correction code type
        self.distance: int = 3  # Default code distance (number of physical qubits)
        self.rounds: int = 1  # Default number of error correction rounds
        self.noise_model: str = "si1000"  # Default noise model (silicon-based qubit model)
        self.p: float = 0.001  # Default error probability per gate
        self.q: int = 17  # Default number of qubits (specific to certain code types)
        self.gates: str = "cz"  # Default gate type for operations
        self.nkd: List[int] = None  # Parameters for bivariate bicycle codes (n, k, d)
        self.iscolored: bool = None  # Flag for color code implementations
        self.A_poly: str = None  # Polynomial A for bivariate bicycle codes
        self.B_poly: str = None  # Polynomial B for bivariate bicycle codes
        self.circuit: str = ""  # String representation of the generated quantum circuit

    def _validate_parameters(self):
        """
        验证所有输入参数是否有效
        
        :raises ValueError: 如果任何参数不合法
        """
        # 验证code task是否支持
        if self.code_task not in ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z",
                                "surface_code_trans_cx:rotated_memory_x", "surface_code_trans_cx:rotated_memory_z",
                                "surface_code_trans_cx:magicEPR", "midout_color_code:memory_x",
                                "midout_color_code:memory_z", "superdense_color_code:memory_x",
                                "superdense_color_code:memory_z", "bivariate_bicycle_code:memory_x",
                                "bivariate_bicycle_code:memory_z", "repetition_code:memory"]:
            raise ValueError("Unsupported code task")
        
        # 验证噪声模型是否支持
        if self.noise_model not in ["si1000", "uniform"]:
            raise ValueError("Unsupported noise model")
        
        # 验证门类型是否支持
        if self.gates not in ["cz", "cx", "all"]:
            raise ValueError("Unsupported gates")
        
        # 验证bivariate_bicycle_code所需的特殊参数
        if "bivariate_bicycle_code" in self.code_task and (self.nkd is None or self.iscolored is None or self.A_poly is None or self.B_poly is None):
            raise ValueError("Missing parameters for bivariate_bicycle_code")

    def generate_circuit(self, code_task: str, distance: int, rounds: int,
                         noise_model: str, p: float, gates: Optional[str] = None,
                         nkd: Optional[List[int]] = None, iscolored: Optional[bool]= None,
                         A_poly:  Optional[str] = "x^3+y+y^2", B_poly:  Optional[str] = "y^3+x+x^2") -> str:
        """
        生成量子纠错电路的核心方法
        
        参数:
        code_task (str): 量子纠错码类型，支持以下格式：
            - 表面码: surface_code:rotated_memory_x（默认） / surface_code:rotated_memory_z
            - 横向CX表面码: surface_code_trans_cx:rotated_memory_x / surface_code_trans_cx:rotated_memory_z / surface_code_trans_cx:magicEPR
            - 颜色码: midout_color_code:memory_x / midout_color_code:memory_z
            - 超密颜色码: superdense_color_code:memory_x / superdense_color_code:memory_z
            - 双变量自行车码: bivariate_bicycle_code:memory_x（需nkd/iscolored/多项式参数） / bivariate_bicycle_code:memory_z
            - 重复码: repetition_code:memory (Z基底)
            
        distance (int): 代码距离，决定纠错能力（物理量子比特数）
        rounds (int): 纠错轮次
        noise_model (str): 噪声模型，支持 si1000（默认超导启发模型）
        p (float): 单量子门错误概率，范围 [0,1]
        gates (str): 使用的量子门类型，支持 cz（默认）、cx 或 all
        nkd (List[int], optional): BB codes专用参数，格式 [[n, k, d]]，n为比特数，k为逻辑比特数，d为码距
        iscolored (bool, optional): BB codes，是否使用彩色编码
        A_poly (str, optional): BB codes对应的多项式A，默认为x^3+y+y^2。
        B_poly (str, optional): BB codes对应的多项式B，默认为y^3+x+x^2。

        返回:
        str: 生成的量子电路字符串表示，包含量子位坐标和操作指令

        异常:
        ValueError: 当输入参数不合法时抛出
        """
        circuit:str = ""  # 初始化空电路字符串

        # 设置实例参数
        self.code_task = code_task
        self.d = distance    # 代码距离赋值
        self.r = rounds      # 纠错轮次赋值
        self.noise_model = noise_model.lower()  # 噪声模型转为小写
        self.p = p           # 错误概率赋值
        self.gates = gates    # 量子门类型赋值
        self.nkd = nkd       # 自行车码参数
        self.iscolored = iscolored  # 彩色编码标志
        self.A_poly = A_poly  # 多项式参数A
        self.B_poly = B_poly  # 多项式参数B

        # 执行参数合法性检查
        self._validate_parameters()

        # 根据代码类型选择生成器
        if "surface_code_trans_cx" in self.code_task:
            circuit = self._generate_surface_code_trans_cx()
        elif "surface_code" in self.code_task:
            circuit = self._generate_surface_code()
        elif "color_code" in self.code_task:
            circuit = self._generate_color_code()
        elif "bivariate_bicycle_code" in self.code_task:
            circuit = self._generate_bivariate_bicycle_code()
        elif "repetition_code" in self.code_task:
            circuit = self._generate_repetition_code()
        else:
            raise ValueError("不支持的代码类型")

        # 缓存生成的电路
        self.circuit = circuit
        
        return circuit
    
    def _generate_surface_code(self):
        """
        生成一个Surface code的stim电路。
        其中参考Google 24年的电路，为ZXXZ线路。
        参考论文：Quantum error correction below the surface code threshold
        
        :return: A str representing the generated surface code circuit.
        # TODO: 3月21日，完成单轮的线路生成，后续补充已实现考虑实现多轮的线路生成。
        """
        # Initialize empty circuit string
        circuit:str = ""
        
        # Data qubit initialization (d x d grid)
        data_qubits = list(range(self.d ** 2))
        data_coords = [(x, y) for y in range(self.d) for x in range(self.d)]
        
        # Measurement qubit setup (d^2-1 measure qubits)
        measure_qubits = list(range(len(data_qubits), len(data_qubits) + self.d **2-1))
        measure_coords = self._surface_code_calculate_measurement_coords()
        
         # Classify measure qubits into MX(4/2) and MZ(4/2) types
        mx_4_qubits, mx_2_qubits, mz_4_qubits, mz_2_qubits = self._surface_code_classify_measurement_qubits(measure_qubits, len(data_qubits), measure_coords)

        # Combine all qubits and create coordinate mappings
        all_qubits  = data_qubits + measure_qubits
        all_coords  = data_coords + measure_coords
        
        # Create qubit coordinate dictionaries
        qubit_coords = {qubit: coord for qubit, coord in zip(all_qubits, all_coords)}
        coords_qubit = {coord: qubit for qubit, coord in qubit_coords.items()}
        
        # Write qubit coordinates to circuit
        for qubit, (y, x) in qubit_coords.items():
            circuit = circuit + f"QUBIT_COORDS({y}, {x}) {qubit}\n"
        
         # Get stabilizer relationships between measure and data qubits
        stablizer_qubits = self._surface_code_get_adjacent_data_qubits(mx_4_qubits, mx_2_qubits, mz_4_qubits, mz_2_qubits, qubit_coords, coords_qubit)
        
        # Apply surface code operations
        even_coords = [(x, y) for x, y in data_coords if (x + y) % 2 == 0]
        odd_coords = [(x, y) for x, y in data_coords if (x + y) % 2 == 1]
        
        MX_qubits = sorted(mx_2_qubits+mx_4_qubits)
        MZ_qubits = sorted(mz_2_qubits+mz_4_qubits)
        MX_qubits_coords = [all_coords[i] for i in MX_qubits]
        MZ_qubits_coords = [all_coords[i] for i in MZ_qubits]
        measure_len = len(measure_qubits)
        mx_len = int(measure_len // 2)
        data_len  = len(data_qubits)
        if "rotated_memory_z" in self.code_task and self.gates == "cz":
            for round_num in range(self.r):
                circuit = self._handle_round_operations(round_num, circuit, all_qubits, all_coords, measure_qubits,
                                 mx_4_qubits, mz_4_qubits,mx_2_qubits, mz_2_qubits, 
                                 data_qubits, MX_qubits, MZ_qubits, MX_qubits_coords, MZ_qubits_coords,
                                 stablizer_qubits, data_len, mx_len, measure_len, even_coords, code_type = "Z")
        else:
            for round_num in range(self.r):
                circuit = self._handle_round_operations(round_num, circuit, all_qubits, all_coords, measure_qubits,
                                 mx_4_qubits, mz_4_qubits,mx_2_qubits, mz_2_qubits, 
                                 data_qubits, MX_qubits, MZ_qubits, MX_qubits_coords, MZ_qubits_coords,
                                 stablizer_qubits, data_len, mx_len, measure_len, odd_coords, code_type = "X")

        return circuit

    def _handle_round_operations(self, round_num, circuit, all_qubits, all_coords, measure_qubits,
                                 mx_4_qubits, mz_4_qubits,mx_2_qubits, mz_2_qubits, 
                                 data_qubits, MX_qubits, MZ_qubits, MX_qubits_coords, MZ_qubits_coords,
                                 stablizer_qubits, data_len, mx_len, measure_len, parity_coords, code_type = "Z"):
        # Z is even_coords
        # X is odd_coords

        if round_num == 0:
            # first rounds
            # Reset all
            # circuit = circuit + "R" +" " + " ".join(map(str, all_qubits)) + "\n"
            # circuit = circuit + "X_ERROR(" + str(self.p * 2) + ") " + " ".join(map(str, all_qubits)) + "\n"
            
            # First round: Initialize all qubits with depolarizing noise
            circuit = circuit + f"DEPOLARIZE1({self.p/10})" +" " + " ".join(map(str, all_qubits)) + "\n"
            circuit = circuit + "TICK\n"
            
            # CX sweep
            circuit = circuit + f"DEPOLARIZE1({self.p/10})" +" " + " ".join(map(str, all_qubits)) + "\n"
            circuit = circuit + "TICK\n"
            
            # Apply Hadamard gates to qubits with odd-parity coordinates
            H_qubits = [i for i in all_qubits if all_coords[i] not in parity_coords]
            circuit =  self._apply_single_operations(circuit, "H", H_qubits, all_qubits)
        else:
            # Subsequent rounds: Reset measurement qubits
            circuit = circuit + "R" +" " + " ".join(map(str, measure_qubits)) + "\n"
            circuit = circuit + "X_ERROR(" + str(self.p * 2) + ") " + " ".join(map(str, measure_qubits)) + "\n"
            
            # Apply noise to data qubits during idle periods
            circuit = circuit + f"DEPOLARIZE1({self.p/10})" +" " + " ".join(map(str, data_qubits)) + "\n"
            circuit = circuit + f"DEPOLARIZE1({self.p*2})" +" " + " ".join(map(str, data_qubits)) + "\n"
            circuit = circuit + "TICK\n"
            
                # Prepare measurement qubits with Hadamard gates
            circuit =  self._apply_single_operations(circuit, "H", measure_qubits, all_qubits)

        # 第一l轮 CZ
        # CZ Gate Application Sequence (4 stages per round)
        for cz_stage in range(4):
            # Stage 0-1: Apply CZ gates between data and measure qubits
            circuit = self._apply_cz_operations(circuit, mx_4_qubits, mz_4_qubits, 
                                mx_2_qubits, mz_2_qubits, stablizer_qubits,
                                all_coords, all_qubits, cz_round=cz_stage)
            if cz_stage == 0:
                circuit =  self._apply_single_operations(circuit, "H", data_qubits, all_qubits)
            elif cz_stage == 1:
                circuit =  self._apply_single_operations(circuit, "X", all_qubits, all_qubits)
            elif cz_stage == 2:
                circuit =  self._apply_single_operations(circuit, "H", data_qubits, all_qubits)
            elif cz_stage == 3:
                pass
            else:
                raise ValueError("CZ stage should be in range [0, 3]")
        
        if self.r == 1:
            # 为最后一轮，并且为第一轮
            # 第一轮QEC的第一轮H，除了坐标之和为偶数的数据比特之外，其他都作用H。最后一轮QEC的最后一轮H也是这样。
            H_qubits = [i for i in all_qubits if all_coords[i] not in parity_coords]
            circuit =  self._apply_single_operations(circuit, "H", H_qubits, all_qubits)
            
            if code_type == "Z":
                # 测量所有测量比特
                circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, MZ_qubits)) + " " + " ".join(map(str, MX_qubits)) + "\n"
                # 添加MX的检测器
                for i, (x, y) in enumerate(MX_qubits_coords):
                    rec_index = mx_len - i
                    circuit = circuit + f"DETECTOR({x}, {y}, {round_num})" + " " + f"rec[{-rec_index}]" + "\n"
            elif code_type == "X":
                # 测量所有测量比特
                circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, MZ_qubits)) + " " + " ".join(map(str, MX_qubits)) + "\n"
                # 添加MX的检测器
                for i, (x, y) in enumerate(MZ_qubits_coords):
                    rec_index = 2*mx_len - i
                    circuit = circuit + f"DETECTOR({x}, {y}, {round_num})" + " " + f"rec[{-rec_index}]" + "\n"

            # 测量所有的数据比特
            circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, data_qubits)) + "\n"
            if code_type == "Z":
                # Detector data qubits and MX qubits
                for i, (x, y) in enumerate(MX_qubits_coords):
                    # 添加MX以及其对应数据比特的detector
                    mx_qubit = MX_qubits[i]
                    rec_index = mx_len - i
                    stablizer_data_qubits = stablizer_qubits[mx_qubit]
                    detector_str = ""
                    rec_str = ""
                    for data_qubit in stablizer_data_qubits:
                        data_rec_index = data_len - data_qubit 
                        data_coord = all_coords[data_qubit]
                        detector_str = detector_str + f"{data_coord[0]}, {data_coord[1]}, {round_num+1}, "
                        rec_str = rec_str + f"rec[{-data_rec_index}] "
                    circuit = circuit + f"DETECTOR({detector_str}{x}, {y}, {round_num}) {rec_str}rec[{-(rec_index+data_len)}]\n"
            elif code_type == "X":
                for i, (x, y) in enumerate(MZ_qubits_coords):
                    # 添加MZ以及其对应数据比特的detector
                    mz_qubit = MZ_qubits[i]
                    rec_index = 2*mx_len - i
                    stablizer_data_qubits = stablizer_qubits[mz_qubit]
                    detector_str = ""
                    rec_str = ""
                    for data_qubit in stablizer_data_qubits:
                        data_rec_index = data_len - data_qubit 
                        data_coord = all_coords[data_qubit]
                        detector_str = detector_str + f"{data_coord[0]}, {data_coord[1]}, {round_num+1}, "
                        rec_str = rec_str + f"rec[{-data_rec_index}] "
                    circuit = circuit + f"DETECTOR({detector_str}{x}, {y}, {round_num}) {rec_str}rec[{-(rec_index+data_len)}]\n"
            
            # observable
            rec_str = ""
            if code_type == "Z":
                for i in range(0, data_len, self.d):
                    rec_index = data_len - i
                    rec_str = rec_str + f"rec[{-rec_index}] "
            elif code_type == "X":
                for i in range(0, self.d, 1):
                    rec_index = data_len - i
                    rec_str = rec_str + f"rec[{-rec_index}] "
            circuit = circuit + f"OBSERVABLE_INCLUDE(0)" + " " + f"{rec_str}"+ "\n"
            # 测量后
            circuit = circuit + f"DEPOLARIZE1({self.p})" +" " + " ".join(map(str, all_qubits)) + "\n"
        elif self.r > 1 and round_num == 0:
            # 第一轮，单只利用4个stablizer
            circuit =  self._apply_single_operations(circuit, "H", measure_qubits, all_qubits)
            if code_type == "Z":
                # 测量所有测量比特
                circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, MZ_qubits)) + " " + " ".join(map(str, MX_qubits)) + "\n"
                # 添加MX的检测器
                for i, (x, y) in enumerate(MX_qubits_coords):
                    rec_index = mx_len - i
                    circuit = circuit + f"DETECTOR({x}, {y}, {round_num})" + " " + f"rec[{-rec_index}]" + "\n"
            elif code_type == "X":
                # 测量所有测量比特
                circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, MZ_qubits)) + " " + " ".join(map(str, MX_qubits)) + "\n"
                # 添加MZ的检测器
                for i, (x, y) in enumerate(MZ_qubits_coords):
                    rec_index = 2*mx_len - i
                    circuit = circuit + f"DETECTOR({x}, {y}, {round_num})" + " " + f"rec[{-rec_index}]" + "\n"
            
            # TODO: 这个地方有多种实现方式，暂时参考Google24的实现方案。
            # 其他数据比特作用Y
            Y_qubits = data_qubits
            circuit =  self._apply_single_operations(circuit, "Y", Y_qubits, all_qubits)
            
            # 给执行测量操作后，Y门作用添加噪声，初始化测量比特添加去极化噪声。
            # circuit = circuit + f"DEPOLARIZE1({self.p/10})" +" " + " ".join(map(str, data_qubits)) + "\n"
            circuit = circuit + f"DEPOLARIZE1({self.p})" +" " + " ".join(map(str, measure_qubits)) + "\n"
            circuit = circuit + "TICK\n"
        elif self.r > 1 and round_num > 0 and round_num < self.r-1:
            # 第一轮，单只利用4个stablizer
            circuit =  self._apply_single_operations(circuit, "H", measure_qubits, all_qubits)

            # 测量所有测量比特
            circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, MZ_qubits)) + " " + " ".join(map(str, MX_qubits)) + "\n"

            # Detector measure qubit and before measure qubits
            for i, (x, y) in enumerate(MZ_qubits_coords):
                rec_index = 2 * mx_len - i
                last_rec_index = rec_index + measure_len
                circuit = circuit + f"DETECTOR({x}, {y}, {round_num}, {x}, {y}, {round_num-1})" + " " + f"rec[{-rec_index}]" +" "+ f"rec[{-last_rec_index}]" + "\n"
            # Detector measure qubit and before measure qubits
            for i, (x, y) in enumerate(MX_qubits_coords):
                rec_index = mx_len - i
                last_rec_index = rec_index + measure_len
                circuit = circuit + f"DETECTOR({x}, {y}, {round_num}, {x}, {y}, {round_num-1})" + " " + f"rec[{-rec_index}]" + " "+ f"rec[{-last_rec_index}]" + "\n"
            
            # TODO: 这个地方有多种实现方式。
            # 其他数据比特作用Y
            Y_qubits = data_qubits
            circuit =  self._apply_single_operations(circuit, "Y", Y_qubits, all_qubits)
            
            # 给执行测量操作后，静置比特添加去极化噪声。
            # circuit = circuit + f"DEPOLARIZE1({self.p/10})" +" " + " ".join(map(str, data_qubits)) + "\n"
            circuit = circuit + f"DEPOLARIZE1({self.p})" +" " + " ".join(map(str, measure_qubits)) + "\n"
            circuit = circuit + "TICK\n"
            
        elif self.r > 1 and round_num == self.r-1:
            H_qubits = [i for i in all_qubits if all_coords[i] not in parity_coords]
            circuit =  self._apply_single_operations(circuit, "H", H_qubits, all_qubits)
            
            # 测量所有测量比特
            circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, MZ_qubits)) + " " + " ".join(map(str, MX_qubits)) + "\n"
            
            for i, (x, y) in enumerate(MZ_qubits_coords):
                rec_index = 2 * mx_len - i
                last_rec_index = rec_index + measure_len
                circuit = circuit + f"DETECTOR({x}, {y}, {round_num}, {x}, {y}, {round_num-1})" + " " + f"rec[{-rec_index}]" +" "+ f"rec[{-last_rec_index}]" + "\n"
            # Detector measure qubit and before measure qubits
            for i, (x, y) in enumerate(MX_qubits_coords):
                rec_index = mx_len - i
                last_rec_index = rec_index + measure_len
                circuit = circuit + f"DETECTOR({x}, {y}, {round_num}, {x}, {y}, {round_num-1})" + " " + f"rec[{-rec_index}]" + " "+ f"rec[{-last_rec_index}]" + "\n"

            
            # 测量所有的数据比特
            circuit = circuit + f"M({self.p*5})" +" " + " ".join(map(str, data_qubits)) + "\n"
            if code_type == "Z":
                # Detector data qubits and MX qubits
                for i, (x, y) in enumerate(MX_qubits_coords):
                    # 是什么数据比特
                    mx_qubit = MX_qubits[i]
                    rec_index = mx_len - i
                    stablizer_data_qubits = stablizer_qubits[mx_qubit]
                    detector_str = ""
                    rec_str = ""
                    for data_qubit in stablizer_data_qubits:
                        data_rec_index = data_len - data_qubit 
                        data_coord = all_coords[data_qubit]
                        detector_str = detector_str + f"{data_coord[0]}, {data_coord[1]}, {round_num+1}, "
                        rec_str = rec_str + f"rec[{-data_rec_index}] "
                    circuit = circuit + f"DETECTOR({detector_str}{x}, {y}, {round_num}) {rec_str}rec[{-(rec_index+data_len)}]\n"
            elif code_type == "X":
                for i, (x, y) in enumerate(MZ_qubits_coords):
                    # 是什么数据比特
                    mz_qubit = MZ_qubits[i]
                    rec_index = 2*mx_len - i
                    stablizer_data_qubits = stablizer_qubits[mz_qubit]
                    detector_str = ""
                    rec_str = ""
                    for data_qubit in stablizer_data_qubits:
                        data_rec_index = data_len - data_qubit 
                        data_coord = all_coords[data_qubit]
                        detector_str = detector_str + f"{data_coord[0]}, {data_coord[1]}, {round_num+1}, "
                        rec_str = rec_str + f"rec[{-data_rec_index}] "
                    circuit = circuit + f"DETECTOR({detector_str}{x}, {y}, {round_num}) {rec_str}rec[{-(rec_index+data_len)}]\n"
                    
            # observable
            rec_str = ""
            if code_type == "Z":
                for i in range(0, data_len, self.d):
                    rec_index = data_len - i
                    rec_str = rec_str + f"rec[{-rec_index}] "
            elif code_type == "X":
                for i in range(0, self.d, 1):
                    rec_index = data_len - i
                    rec_str = rec_str + f"rec[{-rec_index}] "
            circuit = circuit + f"OBSERVABLE_INCLUDE(0)" + " " + f"{rec_str}"+ "\n"
            # 测量后
            circuit = circuit + f"DEPOLARIZE1({self.p})" +" " + " ".join(map(str, all_qubits)) + "\n"
        else:
            raise ValueError("不支持的QEC轮次")
        
        
        return circuit
    def _surface_code_calculate_measurement_coords(self):
        """Calculate surface code coordinates for measurement qubits."""
        ## XZZX code
        coords = []
        for row in range(self.d + 1):
            if row == 0:
                # 第一列
                coords.extend((1.5 + x, -0.5) for x in range(0, self.d-1, 2))
            elif row == self.d:
                coords.extend((x + 0.5, self.d - 0.5) for x in range(0, self.d-1, 2))
            elif row % 2 == 1:
                coords.extend((x - 0.5, -0.5 + row) for x in range(self.d))
            else:
                coords.extend((x + 0.5, -0.5 + row) for x in range(self.d))
        # print("coords:",coords)
        return coords
    
    def _surface_code_classify_measurement_qubits(self, measure_qubits, data_count, measure_coords):
        """Classify surface code measurement qubits into mx and mz types."""
        mx_4_qubits, mx_2_qubits, mz_4_qubits, mz_2_qubits = [], [], [], []
        first_col = self.d // 2
        
        for qubit in measure_qubits:
            idx = qubit - data_count
            row = (idx - first_col) % self.d
            col = (idx - first_col) // self.d
            is_first_col = idx < first_col
            is_last_col = idx >= len(measure_qubits) - first_col
            
            if is_first_col or is_last_col:
                # mz_2_qubits.append(qubit)
                mx_2_qubits.append(qubit)
            elif row % 2 == 1:
                # mz_4_qubits.append(qubit)
                mx_4_qubits.append(qubit)
            elif (row  == 0 and col % 2 == 0) or (row == self.d - 1 and col % 2 == 1):
                # mx_2_qubits.append(qubit)
                mz_2_qubits.append(qubit)
            else:
                # mx_4_qubits.append(qubit)
                mz_4_qubits.append(qubit)
                
        return mx_4_qubits, mx_2_qubits, mz_4_qubits, mz_2_qubits

    def _surface_code_get_adjacent_data_qubits(self, mx_4_qubits, mx_2_qubits, mz_4_qubits, mz_2_qubits, qubit_coords, coords_qubit):
        """Get adjacent data qubits for surface code measurement qubits."""
        data_qubits_neighbors = {}
        left_top = (-0.5, -0.5)
        right_top = (0.5, -0.5)
        left_bottom = (-0.5, 0.5)
        right_bottom = (0.5, 0.5)
        for qubit in mx_4_qubits:
            x, y = qubit_coords[qubit]
            # 左上，右上，左下，右下
            neighbors = [(x + dx, y + dy) for (dx, dy) in [left_top, right_top, left_bottom, right_bottom]]
            data_qubits_neighbors[qubit] = [coords_qubit[(nx, ny)] for (nx, ny) in neighbors]
        for qubit in mx_2_qubits:
            x, y = qubit_coords[qubit]
            # y < 0，左下，右下，x > self.d-1，左上，右上
            if y < 0:
                neighbors = [(x + dx, y + dy) for (dx, dy) in [left_bottom, right_bottom]]
            elif y > self.d-1:
                neighbors = [(x + dx, y + dy) for (dx, dy) in [left_top, right_top]]
            data_qubits_neighbors[qubit] = [coords_qubit[(nx, ny)] for (nx, ny) in neighbors]
        for qubit in mz_4_qubits:
            x, y = qubit_coords[qubit]
            # 左上，左下， 右上，右下
            neighbors = [(x + dx, y + dy) for (dx, dy) in [left_top, left_bottom, right_top, right_bottom]]
            data_qubits_neighbors[qubit] = [coords_qubit[(nx, ny)] for (nx, ny) in neighbors]
        for qubit in mz_2_qubits:
            x, y = qubit_coords[qubit]
            # x < 0，右上，右下，y > self.d-1，左上，左下
            if x < 0:
                neighbors = [(x + dx, y + dy) for (dx, dy) in [right_top, right_bottom]]
            elif x > self.d-1:
                neighbors = [(x + dx, y + dy) for (dx, dy) in [left_top, left_bottom]]
            data_qubits_neighbors[qubit] = [coords_qubit[(nx, ny)] for (nx, ny) in neighbors]
            
        return data_qubits_neighbors

    def _apply_cz_operations(self, circuit: str, mx_4_qubits: list, mz_4_qubits: list, 
                          mx_2_qubits: list, mz_2_qubits: list, stablizer_qubits: dict,
                          all_coords: list, all_qubits: list, cz_round: int) -> str:
        """
        应用CZ操作到指定的量子比特
        
        参数:
        circuit (str): 当前电路字符串
        mx_4_qubits (list): MX4类型的测量量子比特
        mz_4_qubits (list): MZ4类型的测量量子比特
        mx_2_qubits (list): MX2类型的测量量子比特
        mz_2_qubits (list): MZ2类型的测量量子比特
        stablizer_qubits (dict): 稳定子量子比特的邻接关系
        all_coords (list): 所有量子比特的坐标
        all_qubits (list): 所有量子比特的索引
        cz_round: 当前CZ的作用轮次
        
        返回:
        str: 更新后的电路字符串
        """
        cz_use_qubits = []
        
        # 其中测量比特在前，数据比特在后。如果反的话，生成detector会出错。
        for mx4 in mx_4_qubits:
            cz_use_qubits.append(mx4)
            cz_use_qubits.append(stablizer_qubits[mx4][cz_round])
        for mz4 in mz_4_qubits:
            cz_use_qubits.append(mz4)
            cz_use_qubits.append(stablizer_qubits[mz4][cz_round])

        if cz_round == 0 or cz_round == 1:
            for mx2 in mx_2_qubits:
                if all_coords[mx2][1] > self.d-1:
                    cz_use_qubits.append(mx2)
                    cz_use_qubits.append(stablizer_qubits[mx2][cz_round])
            for mz2 in mz_2_qubits:
                if all_coords[mz2][0] > self.d-1:
                    cz_use_qubits.append(mz2)
                    cz_use_qubits.append(stablizer_qubits[mz2][cz_round])
        elif cz_round == 2 or cz_round == 3:
            for mx2 in mx_2_qubits:
                if all_coords[mx2][1] < 0:
                    cz_use_qubits.append(mx2)
                    cz_use_qubits.append(stablizer_qubits[mx2][cz_round-2])
            for mz2 in mz_2_qubits:
                if all_coords[mz2][0] < 0:
                    cz_use_qubits.append(mz2)
                    cz_use_qubits.append(stablizer_qubits[mz2][cz_round-2])

        circuit = circuit + f"CZ" + " " + " ".join(map(str,cz_use_qubits)) + "\n"
        circuit = circuit + f"DEPOLARIZE2({self.p})" + " " + " ".join(map(str,cz_use_qubits)) + "\n"
        no_in_cz_use_qubits = [i for i in all_qubits if i not in cz_use_qubits]
        circuit = circuit + f"DEPOLARIZE1({self.p/10})" + " " + " ".join(map(str, no_in_cz_use_qubits)) + "\n"
        circuit = circuit + "TICK\n"
        
        return circuit
    def _apply_single_operations(self, circuit: str, gate_type: str, gate_qubits: List, noise_qubits: List) -> str:
        circuit = circuit + f"{gate_type}" +" " + " ".join(map(str, gate_qubits)) + "\n"
        circuit = circuit + f"DEPOLARIZE1({self.p/10})" +" " + " ".join(map(str, noise_qubits)) + "\n"
        if gate_type != "Y":
            circuit = circuit + "TICK\n"
        return circuit
    def _generate_surface_code_trans_cx(self):
        """
        Generate a circuit for transversal CNOT operations between rotated surface codes.
        
        :return: A str representing the generated surface_code_trans_cx circuit.
        """
        # Placeholder logic for generating surface code circuit
        return f"Transversal CNOT operations between Rotated Surface Codes circuit with d={self.distance}, r={self.rounds}, noise={self.noise_model}, p={self.p}, gates={self.gates}"

    def _generate_color_code(self):
        """
        Generate a circuit for color code variations.
        
        :return: A str representing the generated color code circuit.
        """
        return f"Color code circuit with d={self.distance}, r={self.rounds}, noise={self.noise_model}, p={self.p}, gates={self.gates}"
    
    def _generate_bivariate_bicycle_code(self):
        """
        Generate a circuit for bivariate bicycle code.
        
        :return: A str representing the generated bivariate bicycle code circuit.
        """
        return f"Bivariate bicycle code circuit with d={self.distance}, r={self.rounds}, noise={self.noise_model}, p={self.p}, gates={self.gates}"
    
    def _generate_repetition_code(self):
        """
        Generate a circuit for repetition code.
        
        :return: A str representing the generated repetition code circuit.
        """
        return f"Repetition code circuit with d={self.distance}, r={self.rounds}, noise={self.noise_model}, p={self.p}, gates={self.gates}"



if __name__ == "__main__":
    # Example usage:
    circuit_gen = CircuitGenerator()
    print(circuit_gen.generate_circuit("surface_code:rotated_memory_z", 3, 1, "SI1000", 0.001, "cz"))
