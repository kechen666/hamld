# HAMLD (Efficient Approximate Maximum Likelihood Decoding)

- [What is HAMLD?](#1?)

- [How do I use HAMLD?](#2)
- [How does HAMLD work?](#3)
- [How do I cite HAMLD?](#4)

<a id="1"></a>

## What is HAMLD?
HAMLD (Efficient Approximate Maximum Likelihood Decoding) 是一种针对量子纠错码(QEC)的创新型解码器，专为处理电路级噪声模型的高精度解码而设计。该解码器通过以下关键技术实现高效解码：

核心技术创新:
- 支持任意纠错码和噪声。支持任意可通过stim含噪线路构建的纠错码，包括stim尚未实现的偏置测量噪声等特殊噪声模型
- 近似加速。引入超图近似、收缩近似方法, 移除部分解码无关的计算代价, 在保证精度的前提下大幅提升速度
- 优化收缩顺序加速。采用优化的收缩顺序算法，无损降低计算复杂度

性能优势分析：
- 复杂度更低。将传统MLD解码器的指数级计算复杂度降至多项式级别
- 更高解码精度。在非破坏性纠错（仅使用辅助比特测量）场景下，逻辑错误率显著优于MWPM和BP+OSD等传统方法

当前技术限制：
- 在存在数据比特测量信息作为syndrome的情况下，Surface code中，相较于MWPM具备更好的精度，但是解码精度提升有限。
- 在存在数据比特测量信息作为syndrome的情况下，QLDPC code中，相较于BP+OSD难以体现精度优势，但是解码复杂度更低。
- 目前解码速度尚未达到超导量子计算实时解码的要求

<a id="2"></a>

## How do I use HAMLD?

### 安装说明
当前 HAMLD 仅支持本地安装，请按照以下步骤进行操作：

1. 创建 Python 虚拟环境（推荐使用 Anaconda）
```bash
conda create -n hamld python=3.10
conda activate hamld
```

2. 安装构建工具
我们采用现代化的 pyproject.toml 构建方案，使用 hatch 作为构建工具：
```bash
pip install hatch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 项目构建与安装
```bash
hatch build  # 生成构建包，输出至 dist/ 目录
```

安装方式任选其一：
```bash
pip install -e .  # 开发模式安装
# 或
pip install ./dist/hamld-0.0.1-py3-none-any.whl  # 直接安装构建包
```

### 验证安装
安装完成后，您可以通过以下方式验证：
```python
import hamld  # 确认可正常导入
```

### 示例代码
以下是一个简单的使用示例：
```python
import numpy as np
import stim
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", 
                                 distance=3, 
                                 rounds=1, 
                                 after_clifford_depolarization=0.05)
num_shots = 1000
model = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)

import hamld
mld_decoder = hamld.HAMLD(detector_error_model=model, order_method='mld', slice_method='no_slice')
predicted_observables = mld_decoder.decode_batch(syndrome)
num_mistakes = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print(f"{num_mistakes}/{num_shots}")
```

### 快速上手
我们提供了详细的教程指南：
- 运行 `HAMLD_Tutorial.ipynb` 教程文件
- 参考教程中的示例代码进行功能测试
- 探索 `hamld` 包的README.md文档，了解更多功能和用法

<a id="3"></a>

## How does HAMLD work?
基于HAMLD的研究论文，该解码器相较于传统MLD解码器及其他解码方案具有以下显著优势：

1. 近似策略的高扩展性
HAMLD采用创新的近似策略实现高效解码，其时间复杂度为O(rd²+C)，其中：
- r表示解码重复轮次
- d为码距参数
- C是与近似噪声相关的常数项
通过优化参数配置，特别是调整C值，可以进一步显著提升解码速度。

2. 逼近MLD的解码精度
在基于辅助比特的syndrome解码场景中，HAMLD表现出：
- 解码精度显著优于MWPM和BP+OSD等传统方法
- 在surface code解码任务中，达到接近MLD的最优解码性能
（注：在QLDPC code场景下，由于无法精确定义MLD基准，故不进行直接比较）

3. 测量偏置噪声的适应性
针对实际量子系统中存在的测量偏置噪声（即0/1测量错误率不对称现象），HAMLD设计了专门的噪声处理机制。这一特性在当前主流的stim模拟器及多数解码器中尚未实现。

### 其他工具
我们还撰写了相关代码：

- 超图分析DEM解码
创新性地采用超图及连通图数据结构，为DEM解码任务提供可视化分析工具

- 含噪量子电路生成器
已实现surface code的stim噪声电路生成，未来计划扩展支持更多量子纠错码类型

<a id="4"></a>

## How do I cite HAMLD?
使用 HAMLD 进行研究时，请引用：
```bibtex
xxxxx
```

## Other Good Open-Source for QEC decoding

* [Stim](https://github.com/quantumlib/Stim.git)
* [PyMatching](https://github.com/oscarhiggott/PyMatching/tree/master)
* [LDPC](https://github.com/quantumgizmos/ldpc)
* [stimbposd](https://github.com/oscarhiggott/stimbposd)
* [tesseract-decoder](https://github.com/quantumlib/tesseract-decoder)
* [qtcodes](https://github.com/yaleqc/qtcodes)