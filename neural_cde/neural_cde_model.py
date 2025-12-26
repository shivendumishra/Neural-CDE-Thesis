import torch
import torch.nn as nn
from neural_cde.cde_vector_field import CDEVectorField
from neural_cde.adjoint_solver import cde_solver

class NeuralCDE(nn.Module):
    
    def __init__(self, input_channels, hidden_dim, output_channels=None):
        super(NeuralCDE, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Vector field f_theta
        self.func = CDEVectorField(input_channels, hidden_dim)
        
        # Layer to map initial data point X0 to Initial Latent State z0
        # z0 = h(x0)
        self.initial_network = nn.Linear(input_channels, hidden_dim)
        
    def forward(self, X_path, timeline=None):
        """
        Args:
            X_path: NaturalCubicSpline object representing X(t).
            timeline: Tensor of times at which to evaluate z(t). 
                      Crucial for 'Controlled Latent Discretization' stage.
        
        Returns:
            z_trajectory: Latent states at requested times (batch, len, hidden_dim)
        """
        
        # 1. Compute initial state z0
        # Evaluate path at t=start (usually t=0)
        # Using X_path.evaluate(0) or similar.
        # torchcde splines usually have .evaluate(t).
        
        # We need the start time. 
        # Typically 0, but if timeline is provided, use timeline[0] or X_path.interval[0]
        
        # Wait, 'interval' attr exists in torchcde CubicSpline
        start_time = X_path.interval[0] if hasattr(X_path, 'interval') else torch.tensor(0.)
        
        x0 = X_path.evaluate(start_time)
        z0 = self.initial_network(x0)
        
        # 2. Integrate CDE
        # Output shape: (batch, time_steps, hidden_dim) if timeline is vector
        # If timeline is Non, usually returns value at t_end.
        # But we want trajectory for discretization.
        
        z_trajectory = cde_solver(X=X_path, 
                                  func=self.func, 
                                  z0=z0, 
                                  t=timeline)
        
        return z_trajectory
