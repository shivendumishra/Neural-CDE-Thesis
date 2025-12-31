import numpy as np
import torch
import sys

def zscore_normalize(data, mean=None, std=None):
    """
    Applies Z-score normalization to the data. Supports both NumPy and Torch.
    """
    is_torch = torch.is_tensor(data) if 'torch' in sys.modules else False
    
    if mean is None:
        if is_torch:
            # For (Batch, Length, Channels), compute mean across (Batch, Length)
            if data.dim() == 3:
                mean = torch.mean(data, dim=(0, 1), keepdim=True)
            else:
                mean = torch.mean(data)
        else:
            mean = np.mean(data, axis=(0, 1) if np.ndim(data) == 3 else 0, keepdims=True)
    
    if std is None:
        if is_torch:
            if data.dim() == 3:
                std = torch.std(data, dim=(0, 1), keepdim=True)
            else:
                std = torch.std(data)
        else:
            std = np.std(data, axis=(0, 1) if np.ndim(data) == 3 else 0, keepdims=True)
            
    # Prevent division by zero
    eps = 1e-8
    if is_torch:
        std = torch.clamp(std, min=eps)
    else:
        std = np.clip(std, a_min=eps, a_max=None)
        
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std

def inverse_zscore(normalized_data, mean, std):
    """
    Reverts Z-score normalization.
    """
    return normalized_data * std + mean
