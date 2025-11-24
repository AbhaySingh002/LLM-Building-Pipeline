import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformer_block import Block
from Cache import KVCache


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
        self.n_head = n_head

        self.pos_embedings = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_block)])
        self.out_head = nn.Linear(d_model, vocab_size)
        self.lrNorm = nn.RMSNorm(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_inference: bool = False,
        start_pos: int = 0
    ):
        B, T = idx.shape

        # During training/prefill, crop if needed. During decoding, we assume 1 token.
        if T > self.context_length:
            idx = idx[:, -self.context_length:]
            T = idx.size(1)

        # Create position indices based on start_pos
        pos = torch.arange(start_pos, start_pos + T, device=idx.device).unsqueeze(0) # (1, T)

        x = self.embedings(idx) + self.pos_embedings(pos)

        # Pass inference flag and kv_cache to each block
        for block in self.blocks:
            x = block(x, is_inference=is_inference, kv_cache=kv_cache)

        x = self.lrNorm(x)
        logits = self.out_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stream: bool = False
    ):
        self.eval()
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=next(self.parameters()).device)

        # Initialize KV Cache
        B = idx.shape[0]
        kv_cache = KVCache(B, self.context_length, self.n_head, self.d_model // self.n_head, dtype=next(self.parameters()).dtype).to(idx.device)
        kv_cache.reset()

        # Prefill Phase (Process the prompt) to fill the cache
        # We pass the whole prompt. T > 1, so attention acts causally.
        logits, _ = self(idx, is_inference=True, kv_cache=kv_cache, start_pos=0)

        # Get the last token's logits to predict autoprogeressively
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Sampling logits from the output distribution
        probs = self._sample_logits(logits, top_k)
        next_token = torch.multinomial(probs, num_samples=1)

        # If not streaming, we collect tokens here
        generated_tokens = [next_token.item()]
        if stream:
            yield next_token.item()

        # Autoregressive Generation Phase
        # Input is now just (B, 1). kv_cache remembers the past.
        input_token = next_token

        for i in range(max_new_tokens - 1):
            # Calculate current position in sequence (prompt len + generated so far)
            current_pos = idx.shape[1] + i

            # Stop if context limit reached
            if current_pos >= self.context_length:
                break

            # Forward pass with ONLY the new token
            logits, _ = self(input_token, is_inference=True, kv_cache=kv_cache, start_pos=current_pos)

            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = self._sample_logits(logits, top_k)
            next_token = torch.multinomial(probs, num_samples=1)

            input_token = next_token # Update input for next iteration
            generated_tokens.append(next_token.item())

            if stream:
                yield next_token.item()

        if not stream:
            # Combine prompt + generated for full output
            full_idx = idx.tolist()[0] + generated_tokens
            return tokenizer.decode(full_idx)

    def _sample_logits(self, logits, top_k):
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        return torch.softmax(logits, dim=-1)