# Experimental Code Structure

This repository contains the following experimental code files organized into different studies:

## Experiment 1: Baseline Performance Comparison
- `overall_performance_qldpc_code_acc.py` - Accuracy evaluation of qLDPC codes
- `overall_performance_qldpc_code_speed.py` - Speed evaluation of qLDPC codes  
- `overall_performance_surface_code_acc.py` - Accuracy evaluation of surface codes
- `overall_performance_surface_code_speed.py` - Speed evaluation of surface codes

## Experiment 2: Biased Measurement Analysis
- `biased_measurement_qldpc_code_acc.py` - qLDPC code accuracy under biased noise
- `biased_measurement_surface_code_acc.py` - Surface code accuracy under biased noise

## Experiment 3: Noise Parameter Studies
### Noise Level Variation
- `noisy_varying_surface_code_acc.py` - Surface code accuracy vs noise
- `noisy_varying_surface_code_speed.py` - Surface code speed vs noise  
- `noisy_varying_qldpc_code_acc.py` - qLDPC code accuracy vs noise
- `noisy_varying_qldpc_code_speed.py` - qLDPC code speed vs noise

### Approximation Parameter Variation
- `approx_param_varying_qldpc_code_acc.py` - qLDPC accuracy vs approximation parameter
- `approx_param_varying_qldpc_code_speed.py` - qLDPC speed vs approximation parameter
- `approx_param_varying_surface_code_acc.py` - Surface code accuracy vs approximation parameter  
- `approx_param_varying_surface_code_speed.py` - Surface code speed vs approximation parameter

## Appendix Experiments: Threshold Analysis
- `noisy_varying_surface_code_acc_threshold_with_data_qubits.py` - Surface code threshold with data qubit noise
- `noisy_varying_surface_code_acc_threshold.py` - Standard surface code threshold analysis

# Data Preparation Note  

**Important:** The data paths in the code need to be modified according to your local directory structure.  

The repository does **not** include pre-generated data files such as:  
- Detector error models (`.dem`)  
- Sampling data (`.dat`)  

To generate these files, please run the following scripts located in the corresponding `data/` directory:  
- `detector_generator.py` - Generates detector error models  
- `circuit_sample_generator.py` - Produces sampling data  

Ensure you execute these scripts before running the main experiments to generate the required input data. You may need to adjust parameters in these generators depending on your experimental needs.