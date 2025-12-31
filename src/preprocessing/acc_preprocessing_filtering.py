import numpy as np
import scipy.signal as signal

def preprocess_accelerometer(acc_data, fs=32):
    """
    Preprocessing for 3-axis input accelerometer data.
    
    Args:
        acc_data: (N, 3) array [x, y, z]
        fs: sampling rate
        
    Returns:
        magnitude: (N,) array of resultant magnitude with gravity removed
    """
    
    # 1. High-pass filter (remove gravity)
    # Gravity is steady component (DC) => Cutoff ~0.2-0.5 Hz
    cutoff = 0.5
    nyquist = 0.5 * fs
    b, a = signal.butter(4, cutoff / nyquist, btype='high')
    
    # Apply to each axis
    acc_filtered = np.zeros_like(acc_data)
    for i in range(3):
        acc_filtered[:, i] = signal.filtfilt(b, a, acc_data[:, i])
        
    # 2. Resultant Magnitude Computation
    # sqrt(x^2 + y^2 + z^2)
    magnitude = np.sqrt(np.sum(acc_filtered**2, axis=1))
    
    return magnitude
