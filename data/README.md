# Data  

The `data` directory contains two main types of datasets:  

1. **External datasets** – Provided by third-party sources:  
   - Surface code experimental data from Google's 2023 paper *"Suppressing quantum errors by scaling a surface code logical qubit"* ([Zenodo](https://zenodo.org/records/6804040)).  
   - Additional surface code data from Google's 2024 paper *"Quantum error correction below the surface code threshold"* ([Zenodo](https://zenodo.org/records/13273331)).  
   - Tesseract decoder datasets ([GitHub](https://github.com/quantumlib/tesseract-decoder)), based on the paper:  
     *Bravyi, S., Cross, A.W., Gambetta, J.M. et al. High-threshold and low-overhead fault-tolerant quantum memory. Nature 627, 778–782 (2024).* ([DOI](https://doi.org/10.1038/s41586-024-07107-7)).  

   Example usage (loading noisy circuits):  
   ```python  
   import stim  
   circuit_noisy = stim.Circuit.from_file("./data/surface_code_bZ_d3_r01_center_3_5/circuit_noisy.stim")  
   ```  

2. **EAMLD-generated datasets (`eamld_experiment_data`)** – Contains:  
   - `xxx_detector_generator`: Tools for constructing detector error models (DEM).  
   - `xxx_circuit_sample_generator`: Utilities for sampling quantum circuits.  

For more details, refer to the respective subdirectories.