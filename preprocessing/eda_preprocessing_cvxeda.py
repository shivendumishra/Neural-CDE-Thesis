import numpy as np
from scipy import signal
from utils.config import EDA_LOW_PASS_CUT

def cvxeda_decompose(eda_signal, fs):
    """
    Placeholder for cvxEDA decomposition.
    In a full implementation, this would use cvxopt to solve the sparse 
    optimization problem for tonic/phasic components.
    """
    n = len(eda_signal)
    tonic = np.zeros(n)
    phasic = np.zeros(n)
    # Placeholder: return zeros to be filled by the real algorithm or fallback
    return tonic, phasic

def eda_preprocessing(eda_signal, fs):
    """
    Preprocesses EDA signal and decomposes it into tonic and phasic components.
    """
    # 1. Low-pass Filter (1Hz)
    nyquist = 0.5 * fs
    low = EDA_LOW_PASS_CUT / nyquist
    b, a = signal.butter(4, low, btype='low')
    filtered_eda = signal.filtfilt(b, a, eda_signal)
    
    # 2. Decomposition
    try:
        # Attempt cvxEDA (or your specific decomposition)
        tonic, phasic = cvxeda_decompose(filtered_eda, fs)
    except:
        # Fallback if cvxopt is not available or fails
        # Using a simple high-pass filter for phasic component extraction
        # Tonic = Low-pass (< 0.05 Hz)
        # Phasic = Original - Tonic
        b_t, a_t = signal.butter(4, 0.05 / nyquist, btype='low')
        tonic = signal.filtfilt(b_t, a_t, filtered_eda)
        phasic = filtered_eda - tonic
        
    return filtered_eda, phasic, tonic
