# Data

在data中, external主要是外部提供的数据源。

其中外部数据主要来源为23年google发表的Suppressing quantum errors by scaling a surface code logical qubit论文中的开源数据：https://zenodo.org/records/6804040。
以及24年google发表的Quantum error correction below the surface code threshold论文中的开源数据：https://zenodo.org/records/13273331。


在data中，raw表示自己构建的一些基础数据，通过stim，来设置码矩和重复轮次以及surface code的电路级噪声模型参数。

```python
import stim
circuit_noisy = stim.Circuit.from_file("./data/surface_code_bZ_d3_r01_center_3_5/circuit_noisy.stim")
```
可以读取对应函数的噪声模型。