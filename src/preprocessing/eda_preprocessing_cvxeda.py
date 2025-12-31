import numpy as np
from scipy import signal
from utils.config import EDA_LOW_PASS_CUT

def cvxeda_decompose(eda_signal, fs):
    """
    Spectral filtering approach for EDA decomposition.
    Tonic: Low-pass filter (< 0.05 Hz)
    Phasic: Original - Tonic, then cleaned with 1Hz LP.
    """
    nyquist = 0.5 * fs
    
    # 1. Extract Tonic (Lower than 0.05 Hz)
    b_t, a_t = signal.butter(4, 0.05 / nyquist, btype='low')
    tonic = signal.filtfilt(b_t, a_t, eda_signal)
    
    # 2. Extract Phasic
    # Initial phasic is signal - tonic
    phasic_raw = eda_signal - tonic
    
    # Clean phasic with a 1Hz low-pass (most SCR events are slow)
    b_p, a_p = signal.butter(4, 1.0 / nyquist, btype='low')
    phasic = signal.filtfilt(b_p, a_p, phasic_raw)
    
    # Ensure phasic is non-negative (biological property of Skin Conductance Responses)
    # and add a tiny epsilon to prevent numerical issues with flat lines
    phasic = np.maximum(phasic, 0) + np.random.normal(0, 1e-6, size=len(phasic))
    
    return tonic, phasic

def eda_preprocessing(eda_signal, fs):
    """
    Preprocesses EDA signal and decomposes it into tonic and phasic components.
    """
    # 1. Low-pass Filter (1Hz) for general noise removal
    nyquist = 0.5 * fs
    low = EDA_LOW_PASS_CUT / nyquist
    b, a = signal.butter(4, low, btype='low')
    filtered_eda = signal.filtfilt(b, a, eda_signal)
    
    # 2. Decomposition
    tonic, phasic = cvxeda_decompose(filtered_eda, fs)
        
    return filtered_eda, phasic, tonic
