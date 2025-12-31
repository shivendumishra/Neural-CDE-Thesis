import torch
import numpy as np

def generate_fixed_timeline(start_time, end_time, sampling_rate):
    """
    Generates a fixed timeline for sampling latent trajectories.
    
    Args:
        start_time: Start time (scalar)
        end_time: End time (scalar)
        sampling_rate: Hz
        
    Returns:
        timeline: Tensor of times [t_0, t_1, ..., t_N]
    """
    # Calculate step size
    dt = 1.0 / sampling_rate
    
    # Generate points
    # Use torch.arange or linspace
    # We want to ensure we cover the range
    
    # Check if inputs are tensors
    if torch.is_tensor(start_time):
        start_time = start_time.item()
    if torch.is_tensor(end_time):
        end_time = end_time.item()
        
    timeline = torch.arange(start_time, end_time, dt, dtype=torch.float32)
    
    # If end_time is not exactly hit due to float precision or step, 
    # we might want to include it or not. 
    # Usually strictly < end is fine, or <=. 
    # We'll stick to arange which is [start, end).
    
    return timeline

def sample_latent_trajectory(neural_cde_model, X_path, global_timeline):
    """
    Evaluates the Neural CDE model at the specified global timeline.
    
    This acts as the 'Controlled Latent Discretization' step.
    
    Args:
        neural_cde_model: The trained NeuralCDE instance.
        X_path: The continuous path object (Spline).
        global_timeline: Tensor of times to sample.
        
    Returns:
        z_discretized: (batch, num_steps, hidden_dim)
    """
    
    z_discretized = neural_cde_model(X_path, timeline=global_timeline)
    return z_discretized
