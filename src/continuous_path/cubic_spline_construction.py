import torch
import torchcde

class ContinuousPathBuilder:
    """
    Constructs continuous-time paths from discrete time-series data using 
    Natural Cubic Spline Interpolation.
    """
    
    def __init__(self):
        pass
        
    def build_path(self, data, times=None):
        """
        Computes the natural cubic spline coefficients for the given data.
        
        Args:
            data: Tensor of shape (batch, length, channels)
            times: Tensor of shape (batch, length) representing observation times.
                   If None, assumes regular sampling (handled by torchcde usually requiring coeffs).
                   Note: torchcde.natural_cubic_coeffs usually expects fixed time grid or explicit times.
                   If data is (B, L, C), and we want X(t), we need coeffs.
                   
        Returns:
            coeffs: Spline coefficients suitable for torchcde.NaturalCubicSpline
        """
        
        # Ensure data is a tensor
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
            
        if times is not None and not torch.is_tensor(times):
            times = torch.tensor(times, dtype=torch.float32)
            
        # torchcde.natural_cubic_coeffs expects input (batch, seq_len, channels)
        # It natively handles the interpolation construction.
        # If 'times' acts as the explicit time channel included in 'data', 
        # then we just pass 'data'. 
        
        # If 'times' are separate and irregular, we normally augment data with time 
        # (which we did in normalization/intensity_channel.py).
        # So 'data' should already have the time channel or we treat indices as time.
        
        # According to torchcde docs:
        # coeffs = torchcde.natural_cubic_coeffs(X, t=t)
        # returns the coefficients.
        
        if times is None:
            # Assumes values are observed at t=0, 1, ..., L-1
            # Or t must be passed.
            # We will assume 't' is implicitly the index scaled or provided.
            coeffs = torchcde.natural_cubic_coeffs(data)
        else:
            coeffs = torchcde.natural_cubic_coeffs(data, t=times)
            
        return coeffs

    def get_spline(self, coeffs):
        """
        Returns a NaturalCubicSpline object that can be queried at any t.
        """
        return torchcde.CubicSpline(coeffs)

def build_spline(data, times=None):
    """
    Functional wrapper.
    """
    builder = ContinuousPathBuilder()
    coeffs = builder.build_path(data, times)
    return builder.get_spline(coeffs)
