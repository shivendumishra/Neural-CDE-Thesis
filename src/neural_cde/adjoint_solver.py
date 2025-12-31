import torch
import torchcde

def cde_solver(X, func, z0, t=None, adjoint=True, backend='torchdiffeq', **kwargs):
    """
    Solves the CDE dz(t) = f(z(t)) dX(t).
    
    Args:
        X: The control path (NaturalCubicSpline).
        func: The vector field module.
        z0: Initial latent state.
        t: Times to evaluate the solution at. If None, evaluates at all step points 
           or endpoints depending on usage (usually CDE returns the path).
           
        adjoint: Whether to use the adjoint sensitivity method for O(1) memory.
        backend: 'torchdiffeq' is standard.
        
    Returns:
        z_t: Latent trajectory.
    """
    
    # Switched to 'dopri5' (adaptive) for faster integration
    z = torchcde.cdeint(X=X,
                        func=func,
                        z0=z0,
                        t=t,
                        adjoint=False,
                        method='dopri5',
                        **kwargs)
                        
    return z
