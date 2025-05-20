# Experimental Data Sources

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

Each directory contains the complete experimental code and analysis scripts for the respective study.