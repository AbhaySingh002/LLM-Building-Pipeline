import torch
import torch.nn as nn
import torch.nn.functional as F

from Gate import Router

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
    

class Experts(nn.Module):
    def __init__(self, num_experts: int , d_model: int, hidden_times: int = 3, dropout: float = 0.0, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            SwishFFN(d_model, hidden_times, dropout) for _ in range(num_experts)
        ])
        self.router = Router(self.num_experts, d_model, self.top_k)
        
        
    def forward(self, x):
        B, T, C = x.shape
        routing_weight, all_probs = self.router(x)  # exp_id: (B, T, top_k)
        
        x_reshaped = x.view(-1, C)
        
        output = torch.zeros_like(x_reshaped)
        for ids in range(self.num_experts):
            # getting the each expert weight column
            expert_weight_col = routing_weight[:, ids]
            
            # creating a boolean mask for the choosing the tokens for the current expert
            expert_mask = expert_weight_col > 0
            
            if not torch.any(expert_mask):
                continue
            
            selected_tokens = x_reshaped[expert_mask]
            
            expert_output = self.experts[ids](selected_tokens)
            weight_output = expert_weight_col[expert_mask].unsqueeze(-1) * expert_output
            
            output[expert_mask] += weight_output
            
        
        output = output.view(B, T, C)
        aux_loss = self._compute_load_balancing_loss(routing_weight, all_probs)
        return output , aux_loss


    def _compute_load_balancing_loss(self, routing_weights, all_probs, lambda_val: float = 0.01):
        '''
        Compute load balancing loss to encourage even distribution of tokens across experts.
        routing_weights: (B*T, num_experts) , sparse matrix with top-k weights
        all_probs: (B*T, num_experts) , softmax probabilities
        '''
        if routing_weights is None or all_probs is None:
            return 0.0
        # the fraction of tokens assigned to each expert
        total_tokens = routing_weights.shape[0]
        fi = (routing_weights > 0).float().sum(0) / total_tokens
        fi = fi.detach()

        # the average probability assigned to each expert
        pi = all_probs.mean(0)
        
        # Switch Transformers load balancing loss
        aux_loss = lambda_val * self.num_experts * (fi * pi).sum()

        return aux_loss