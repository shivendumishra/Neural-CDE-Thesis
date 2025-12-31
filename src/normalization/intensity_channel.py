import numpy as np
import torch

def add_intensity_channel(data, time_points=None):
    """
    Adds an intensity channel (cumulative observation counter or time) to the data.
    
    For Neural CDEs, it's often useful to augment the state with time or an 
    observational intensity counter.
    
    Args:
        data: (N, C) tensor or array
        time_points: (N,) tensor or array of timestamps. 
                     If None, assumes regular sampling and uses index.
                     
    Returns:
        augmented_data: (N, C+1)
    """
    if isinstance(data, np.ndarray):
        is_numpy = True
        N = data.shape[0]
        if data.ndim == 1:
            data = data[:, None]
    else:
        is_numpy = False
        N = data.size(0)
        if data.dim() == 1:
            data = data.unsqueeze(1)
            
    if time_points is None:
        if is_numpy:
            # Normalized time 0 to 1 or just cumulative index
            time_channel = np.arange(N, dtype=np.float32)[:, None]
        else:
            time_channel = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    else:
        if is_numpy:
            if time_points.ndim == 1:
                time_channel = time_points[:, None]
            else:
                time_channel = time_points
        else:
            if time_points.dim() == 1:
                time_channel = time_points.unsqueeze(1)
            else:
                time_channel = time_points
                
    # Reshape time_channel to match data dimensions
    if is_numpy:
        if data.ndim == 3:
            # (Batch, Length, Channels)
            if time_channel.ndim == 2:
                time_channel = time_channel[:, :, None]
            concatenated = np.concatenate([data, time_channel], axis=2)
        else:
            # (Length, Channels)
            if time_channel.ndim == 1:
                time_channel = time_channel[:, None]
            concatenated = np.concatenate([data, time_channel], axis=1)
    else:
        if data.dim() == 3:
            # (Batch, Length, Channels)
            if time_channel.dim() == 2:
                time_channel = time_channel.unsqueeze(-1)
            concatenated = torch.cat([data, time_channel], dim=2)
        else:
            # (Length, Channels)
            if time_channel.dim() == 1:
                time_channel = time_channel.unsqueeze(1)
            concatenated = torch.cat([data, time_channel], dim=1)
            
    return concatenated
