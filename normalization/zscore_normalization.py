import numpy as np

def zscore_normalize(data, mean=None, std=None):
    """
    Applies Z-score normalization to the data.
    
    Args:
        data: input array (N, C) or (N,)
        mean: pre-computed mean (optional). If None, computed from data.
        std: pre-computed std (optional). If None, computed from data.
        
    Returns:
        normalized_data: (data - mean) / std
        mean: computed or used mean
        std: computed or used std
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    
    if std is None:
        std = np.std(data, axis=0)
        
    # Prevent division by zero
    if np.ndim(std) == 0:
        if std == 0: std = 1.0
    else:
        std[std == 0] = 1.0
        
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std

def inverse_zscore(normalized_data, mean, std):
    """
    Reverts Z-score normalization.
    """
    return normalized_data * std + mean
