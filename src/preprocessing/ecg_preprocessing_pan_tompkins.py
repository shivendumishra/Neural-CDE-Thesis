import numpy as np
import scipy.signal as signal
from utils.config import FS_ECG, ECG_LOW_CUT, ECG_HIGH_CUT

def bandpass_filter(data, lowcut, highcut, fs, order=1):
    """
    Butterworth Bandpass Filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def pan_tompkins_detector(ecg_signal, fs):
    """
    Pan-Tompkins Algorithm for R-peak detection.
    
    Steps:
    1. Bandpass Filter (5-15Hz in original, but we use 0.5-40Hz as per config/prompt generally, 
       though PT specifically asks for 5-15Hz for QRS maximization. We will respect the prompt's 
       mention of Bandpass+Notch first, then PT.
       The prompt says: "Bandpass + Notch Filtering" then "Pan-Tompkins".
       So we assume the input `ecg_signal` might need cleaning or we clean it here.
    """
    
    # 1. Bandpass Filter
    # Standard PT uses 5-15Hz. Prompt asked for "Bandpass + Notch" in 2.1.
    # We will apply the config-based filtering (0.5-40Hz) if not already done, 
    # but for detection specifically, a tighter band is often better.
    # We'll stick to the prompt's "Bandpass + Notch" requirement generally.
    # Let's assume input is raw and apply general cleaning first.
    
    filtered_ecg = bandpass_filter(ecg_signal, ECG_LOW_CUT, ECG_HIGH_CUT, fs, order=4)
    
    # Notch filter (50Hz or 60Hz)
    # Using a simple iirnotch
    notch_freq = 50.0  # From config, assume config is imported or passed
    nyquist = 0.5 * fs
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq / nyquist, quality_factor)
    filtered_ecg = signal.filtfilt(b_notch, a_notch, filtered_ecg)

    # 2. Derivative
    # H(z) = (1/8T)(-z^-2 - 2z^-1 + 2z^1 + z^2)
    # Approximation: y[n] = x[n+1] - x[n-1] ... standard diff is simpler
    diff_ecg = np.diff(filtered_ecg)
    
    # 3. Squaring
    squared_ecg = diff_ecg ** 2
    
    # 4. Moving Window Integration
    # Window width ~ 150ms
    window_extracted = int(0.150 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_extracted)/window_extracted, mode='same')
    
    # 5. Peak Detection
    # Finding local maxima
    # Use scipy.signal.find_peaks with distance constraint (refractory period ~200ms)
    min_distance = int(0.2 * fs)
    peaks, _ = signal.find_peaks(integrated_ecg, distance=min_distance, height=np.mean(integrated_ecg))
    
    # The peaks found are in the integrated signal. 
    # We need to map them back to the R-peaks in the filtered ECG.
    # We search in a small window around the integrated peak in the filtered signal.
    
    r_peaks = []
    search_window = int(0.1 * fs) # +/- 100ms
    
    for peak in peaks:
        start_idx = max(0, peak - search_window)
        end_idx = min(len(filtered_ecg), peak + search_window)
        if start_idx >= end_idx:
            continue
        
        # Find max in filtered ECG in this window
        local_max_idx = np.argmax(filtered_ecg[start_idx:end_idx])
        r_peaks.append(start_idx + local_max_idx)
    
    # Remove duplicates and sort
    r_peaks = np.unique(r_peaks)
    
    return filtered_ecg, r_peaks

if __name__ == "__main__":
    # Test stub
    t = np.linspace(0, 10, 7000)
    mock_ecg = np.sin(2 * np.pi * 1.0 * t) # Dummy
    clean, peaks = pan_tompkins_detector(mock_ecg, FS_ECG)
    print(f"Detected {len(peaks)} peaks")
