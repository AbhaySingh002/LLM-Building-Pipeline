import torch
from torch import nn

class KVCache(nn.Module):
    def __init__(self, batch_size, max_seq_len, num_heads, head_dim, dtype=torch.float32):
        super().__init__()
        self.k_cache: torch.Tensor
        self.v_cache: torch.Tensor
        self.cache_pos: torch.Tensor

        # Pre-allocate empty cache
        self.register_buffer("k_cache", torch.zeros(batch_size, num_heads, max_seq_len, head_dim, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(batch_size, num_heads, max_seq_len, head_dim, dtype=dtype))

        # Position can be a scalar tensor since batch usually generates in sync
        self.register_buffer("cache_pos", torch.tensor(0, dtype=torch.long))

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        # k_new shape: (B, H, T_new, D)
        seq_len_new = k_new.size(2)

        #  Get Scalar Integer for slicing
        # We must use .item() because we cannot use a Tensor to define a slice range like [pos : pos+n]
        pos = self.cache_pos.item()

        # Update the cache
        self.k_cache[:, :, pos : pos + seq_len_new, :] = k_new
        self.v_cache[:, :, pos : pos + seq_len_new, :] = v_new

        # Update position
        self.cache_pos += seq_len_new

        # CRITICAL
        # If we return the full buffer, Attention will see the empty zeros and mess up results.
        # We slice up to the current filled length.
        current_len = pos + seq_len_new
        return (
            self.k_cache[:, :, :current_len, :],
            self.v_cache[:, :, :current_len, :]
        )

    def reset(self):
        self.cache_pos.zero_()