import numpy as np

def compute_rr_intervals(r_peaks, fs):
    """
    Computes RR intervals in milliseconds from R-peak indices.
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    # Difference in indices
    r_peak_diffs = np.diff(r_peaks)
    
    # Convert samples to ms
    rr_intervals_ms = (r_peak_diffs / fs) * 1000.0
    
    return rr_intervals_ms

def extract_hrv_features(rr_intervals):
    """
    Extracts time-domain HRV features.
    
    Returns:
        dict: {
            'mean_rr': float,
            'sdnn': float,
            'rmssd': float
        }
    """
    if len(rr_intervals) < 1:
        return {
            'mean_rr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0
        }
    
    # Mean RR
    mean_rr = np.mean(rr_intervals)
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_intervals, ddof=1) if len(rr_intervals) > 1 else 0.0
    
    # RMSSD: Root mean square of successive differences
    diff_rr = np.diff(rr_intervals)
    if len(diff_rr) > 0:
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
    else:
        rmssd = 0.0
        
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd
    }

if __name__ == "__main__":
    # Test stub
    dummy_rr = np.array([800, 810, 790, 805, 800])
    features = extract_hrv_features(dummy_rr)
    print("Features:", features)
