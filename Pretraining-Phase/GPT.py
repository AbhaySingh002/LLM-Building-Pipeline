import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Decoder_Multi_Head_Attention(nn.Module):
    def __init__(self, n_head: int, d_model:int, dropout:float=0.0):
        super().__init__()
        assert d_model % n_head == 0   # should be divisible (d_model % n_head = 0)
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor) -> None:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out
    
## End Feed Forward of One Block 
class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_times: int = 3, dropout: float=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_times*d_model),
            nn.GELU(),
            nn.Linear(hidden_times*d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
        
## one single tranformer block       
class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float=0.2) -> None:
        super().__init__()
        self.lr1Norm = nn.LayerNorm(d_model)
        self.multiHead_attention = Decoder_Multi_Head_Attention(n_head, d_model, dropout)
        self.lr2Norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        
    def forward(self, x: torch.Tensor):
        x = x + self.multiHead_attention(self.lr1Norm(x))
        x = x + self.ffn(self.lr2Norm(x))
        
        return x
    
    

## multi block architecture - GPT

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, n_block: int = 4, n_head: int = 4, d_model: int = 256, dropout: float = 0.0):
        super().__init__()
        self.embedings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.context_length = context_length
        self.pos_embedings = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head, dropout) for _ in range(n_block)])
        self.out_head = nn.Linear(d_model, vocab_size)
        self.lrNorm = nn.LayerNorm(d_model)
        
        self.apply(self._init_weights)
        
    # initialising the weights in normal distribution near zero - this results better than random initialising 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std= 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
    
    
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor]=None):
        B, T = idx.shape
        if T > self.context_length:
            idx = idx[:, -self.context_length:]
            T = idx.size(1)
        
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.embedings(idx) + self.pos_embedings(pos)
        x = self.blocks(x)
        x = self.lrNorm(x)
        logits = self.out_head(x)
        loss = None
        if targets is not None :
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            
        return logits, loss