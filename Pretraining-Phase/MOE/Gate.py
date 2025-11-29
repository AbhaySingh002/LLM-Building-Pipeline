import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, num_experts: int, d_model: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.route = nn.Linear(d_model, num_experts)
    
    def forward(self, x:torch.Tensor):
        _,_,C = x.shape
        x_flat = x.view(-1, C)
        probs  = F.softmax(self.route(x_flat), dim=-1)
        exp_v, exp_id  = torch.topk(probs, k=self.top_k, dim=-1)
        
        # a sparse routing matrix of shape (B*T, num_experts)
        routing_weights = torch.zeros_like(probs)
        
        # Scatter the top-k values back into the full matrix size at their respective indices
        # This puts the probability in the column of the selected expert, and leaves 0 elsewhere.
        routing_weights.scatter_(1, exp_id, exp_v)
        return routing_weights, probs
        