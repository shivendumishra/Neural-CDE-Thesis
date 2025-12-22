import numpy as np
import scipy.stats as stats

def extract_acc_features(acc_magnitude):
    """
    Extracts statistical features from the accelerometer magnitude sequence.
    
    Features:
    - Mean
    - Variance
    - Energy
    - Entropy
    """
    
    # Mean
    mean_val = np.mean(acc_magnitude)
    
    # Variance
    var_val = np.var(acc_magnitude)
    
    # Energy: sum of squares
    energy = np.sum(acc_magnitude ** 2)
    
    # Entropy: Shannon entropy of the distribution of values
    # We can use histogram limits to estimate probability density
    # Or spectral entropy. Prompt just says "Entropy". 
    # Usually time-domain entropy -> Shannon entropy of binned values
    
    hist, bin_edges = np.histogram(acc_magnitude, bins=10, density=True)
    # Filter zeros for log
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    
    return {
        'acc_mean': mean_val,
        'acc_var': var_val,
        'acc_energy': energy,
        'acc_entropy': entropy
    }
