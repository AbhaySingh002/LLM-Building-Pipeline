import torch
import torch.nn as nn

class RmsNorm(nn.Module):
    '''
    Root Mean Square Layer Normalization
    
    formula for RMSNorm:
        RMSNorm(x) = x * (gamma / rms(x))
    where rms(x) = sqrt(mean(x^2) + epsilon)
    
    epsilon is a small constant to prevent division by zero.
    gamma is a learnable scaling parameter.
    
    RMSNorm normalizes the inputs based on the root mean square of the elements,
    without centering them around the mean. This can lead to more stable training in architectures like Transformers.
    '''
    def __init__(self, dim: int, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.epsilon).rsqrt()
        return x * rms * self.gamma