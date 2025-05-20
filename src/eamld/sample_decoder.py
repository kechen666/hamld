import numpy as np
import stim

class Sample_Decoder:
    def __init__(self, noisy_circuit: stim.Circuit, shots: int = 1000):
        self.num_shots = shots
        # Compile the detector sampler and perform the sampling
        sampler = noisy_circuit.compile_detector_sampler()
        # Sample the syndromes and actual observables separately
        self.syndromes, self.actual_observables = sampler.sample(shots=self.num_shots, separate_observables=True)
    
    def decode(self, syndrome: np.ndarray):
        # Find the indices where self.syndromes matches the input syndrome
        matching_indices = np.all(self.syndromes == syndrome, axis=1)
        
        # Get the corresponding actual observables for the matching syndromes
        matching_observables = self.actual_observables[matching_indices]
        
        # Count the True and False values in matching_observables
        true_count = np.sum(matching_observables)
        false_count = len(matching_observables) - true_count
        
        # Calculate the ratio of True and False counts
        total_count = true_count + false_count
        true_ratio = true_count / total_count if total_count > 0 else 0
        false_ratio = false_count / total_count if total_count > 0 else 0
        
        # Return the result: the most likely value and its confidence ratio
        if true_count > false_count:
            return np.array([True]), true_ratio
        else:
            return np.array([False]), false_ratio
