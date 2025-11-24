import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from Cache import KVCache
from FFN import SwishFFN
from layerNorm import RmsNorm

class Decoder_Multi_Head_Attention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_inference: bool = False, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)

        # Split q, k, v
        q, k, v = qkv.unbind(dim=2)

        # Transpose for attention: (B, n_head, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV Cache Logic only during inference
        if is_inference and kv_cache is not None:
            # Update cache with new k, v and get the full history back
            k, v = kv_cache.update(k, v)

        # Scaled Dot Product Attention
        # CRITICAL: During inference decoding (T=1), we attend to ALL past keys (k),
        # During training, it is True or the coming sequence of tokens are greater than the 1
        use_causal_mask = not is_inference or (is_inference and T > 1)

        # If we are in inference and T=1 (decoding), we don't apply dropout
        dropout_p = self.dropout.p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=use_causal_mask
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.lr1Norm = RmsNorm(d_model)
        self.multiHead_attention = Decoder_Multi_Head_Attention(n_head, d_model, dropout)
        self.lr2Norm = RmsNorm(d_model)
        self.ffn = SwishFFN(d_model, 4, dropout)

    def forward(self, x: torch.Tensor, is_inference: bool = False, kv_cache: Optional['KVCache'] = None):
        # Pass inference flags down to attention
        x = x + self.multiHead_attention(self.lr1Norm(x), is_inference=is_inference, kv_cache=kv_cache)
        x = x + self.ffn(self.lr2Norm(x))
        return x