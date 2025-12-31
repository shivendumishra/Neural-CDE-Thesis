import torch
import torch.nn as nn

class CDEVectorField(nn.Module):
    """
    Learns the vector field f_theta(z) for the CDE:
    dz(t) = f_theta(z(t)) dX(t)
    
    The function returns a matrix of size (batch, hidden_dim, input_channels).
    """
    
    def __init__(self, input_channels, hidden_dim):
        super(CDEVectorField, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # A standard MLP parameterization
        # z -> hidden -> .. -> hidden * input_channels
        
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, hidden_dim * input_channels)
        
        # Initialize weights carefully (often near zero for stability in ODEs)
        # But standard initialization is usually fine for short sequences.
        # Tanh is sometimes preferred for ODE vector fields to bound derivatives.
        self.activation = nn.Tanh() 

    def forward(self, t, z):
        # t is provided by the solver but often unused in autonomous ODEs/CDEs 
        # unless specifically needed.
        
        # z: (batch, hidden_dim)
        batch_size = z.shape[0]
        
        x = self.linear1(z)
        x = self.activation(x)
        x = self.linear2(x)
        
        # Reshape to (batch, hidden_dim, input_channels)
        # This matrix acts on dX (which is (batch, input_channels))
        # Result is (batch, hidden_dim) update to z.
        
        return x.view(batch_size, self.hidden_dim, self.input_channels)
