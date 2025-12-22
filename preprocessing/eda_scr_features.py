import numpy as np
import scipy.signal as signal

def extract_scr_features(phasic_component, fs=4):
    """
    Extracts Skin Conductance Response (SCR) features from the phasic component.
    
    Features:
    - Number of peaks (SCRs)
    - Mean amplitude of peaks
    - Mean phasic value
    - Std phasic value
    """
    
    # detect peaks in phasic component
    # Threshold: e.g., 0.01 muS
    threshold = 0.01 
    
    peaks, properties = signal.find_peaks(phasic_component, height=threshold, distance=1.0*fs)
    
    num_peaks = len(peaks)
    
    if num_peaks > 0:
        mean_peak_amp = np.mean(properties['peak_heights'])
    else:
        mean_peak_amp = 0.0
        
    mean_phasic = np.mean(phasic_component)
    std_phasic = np.std(phasic_component)
    
    # Area under the curve (integral of absolute phasic)?? 
    # Prompt asks for: Statistical Features (mean, std, number of peaks)
    
    features = {
        'scr_num_peaks': num_peaks,
        'scr_mean_peak_amp': mean_peak_amp,
        'scr_mean_phasic': mean_phasic,
        'scr_std_phasic': std_phasic
    }
    
    return features
