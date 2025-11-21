import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, n_head: int, d_model:int, context_length:int, dropout:float=0.0, log_shape:bool=True):
        super().__init__()
        self.log_shape = log_shape
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, "Dimension of embeddings must be divisible by number of heads"
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        mask = torch.triu(
            torch.ones(context_length, context_length, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", mask)
        
        
    def forward(self, x:torch.Tensor):
        B, T, D= x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)
        
        if self.log_shape:
            print(f'Shape of the Q, K, V for each head : {qkv.shape}')
            
        q, k, v = qkv.unbind(dim=2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.log_shape:
            print("q:", q.shape, "k:", k.shape, "v:", v.shape)
            
        
        norm = 1/math.sqrt(self.d_head)
        
        attention_weight = torch.matmul(q, k.transpose(-2,-1)) * norm
        
        # for the decoder - Masking
        attention_weight = attention_weight.masked_fill(self.causal_mask[:T, :T], float('-inf'))

        attention_weight_prob = F.softmax(attention_weight, dim=-1)
        attention_weight_prob = self.dropout(attention_weight_prob)
        context = torch.matmul(attention_weight_prob,v)
        
        if self.log_shape:
            print("weights:", attention_weight_prob.shape, "context:", context.shape)
            
        out = context.transpose(1, 2).contiguous().view(B,T,self.d_model)
        out = self.projection(out)
        
        if self.log_shape:
            print("Multi-Head Shape output :", out.shape)
        return out, attention_weight_prob
        
        