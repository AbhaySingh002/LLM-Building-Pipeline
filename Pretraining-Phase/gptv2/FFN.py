import torch
import torch.nn as nn

class SwishFFN(nn.Module):
    '''
    Feed-Forward Network with SiLU (Swish) activation , more smoother than ReLU, is differentiable.
    '''
    def __init__(self, d_model: int, hidden_times: int = 3, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * hidden_times)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)