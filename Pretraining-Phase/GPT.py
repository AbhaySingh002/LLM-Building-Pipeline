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
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        n_block: int = 4, 
        n_head: int = 4, 
        d_model: int = 256, 
        dropout: float = 0.0
    ):
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
            
    
    
        
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor]=None
    ):
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
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        tokenizer=None,
        stream: bool = False  
    ):
        self.eval()
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
            
        idx = tokenizer.encode(prompt)
        idx = torch.tensor([idx], dtype=torch.long, device=next(self.parameters()).device)

        # Generation loop
        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            # Sampling (Top-K / Top-P)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

            # using yeild to generate the token
            if stream:
                yield int(next_token.item())

        # 
        if not stream:
            return tokenizer.decode(idx[0].tolist())