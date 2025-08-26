# Experimental Data Sources

Each directory contains the complete experimental code and analysis scripts for the respective study.

Note: Since a wider range of parameter configurations was tested in the experiments, the .stim files provided in the generated folders may not correspond exactly to the parameter settings used in the paper. If you need to use the relevant data, we strongly recommend referring to our data generation code. (注意：由于在实验中测试了更多种参数的配置，因此生成的文件夹中提供的.stim文件可能不一定是按论文实验参数配置生成的文件。如果你需要使用相关的数据，我们强烈建议你阅读我们的数据生成代码。)


## Surface Code Implementation
Our surface code implementation is based on:  
**Google Quantum AI and Collaborators**  
"Quantum error correction below the surface code threshold"  
*Nature* 638, 920–926 (2025)  
https://doi.org/10.1038/s41586-024-08449-y  

We implemented a ZXXZ Surface code SI1000 quantum circuit using the stim framework.

## QLDPC Code Implementation
Our quantum low-density parity-check (QLDPC) code implementation draws from two key references:  

1. **Bravyi, S., Cross, A.W., Gambetta, J.M. et al.**  
"High-threshold and low-overhead fault-tolerant quantum memory"  
*Nature* 627, 778–782 (2024)  
https://doi.org/10.1038/s41586-024-07107-7  

2. **Beni L.A., Higgott O., Shutty N.**  
"Tesseract: A search-based decoder for quantum error correction"  
*arXiv preprint* arXiv:2503.10988 (2025)  
https://doi.org/10.48550/arXiv.2503.10988  

These references informed our implementation of bivariate bicycle codes using stim circuits.

## Experimental Directory Structure
The repository is organized into the following experimental directories:

1. `overall_performance/` - Baseline performance comparisons
2. `bias_measurement/` - Biased noise measurement analysis
3. `noise_varying/` - Noise parameter variation studies
4. `approx_param_varying/` - Approximation parameter analysis
5. `noise_varying_threshold/` - Threshold determination experiments
6. Other experimental studies
