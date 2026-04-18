"""Standard multi-head attention with optional TurboQuant KV cache."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from edgemoe.quantization.turboquant import TurboQuantKVCache, TurboQuantCacheStore


class StandardAttention(nn.Module):
    """Vanilla MHA — works everywhere, serves as the correctness reference."""

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        use_turboquant: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden, bias=False)
        self.use_turboquant = use_turboquant
        if use_turboquant:
            self.kv_compressor = TurboQuantKVCache(self.head_dim, device=device)
            self.kv_store = TurboQuantCacheStore(self.kv_compressor)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _shape(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        layer_id: int,
        past_kv: bool = True,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self._shape(self.q_proj(x), self.num_heads)
        k = self._shape(self.k_proj(x), self.num_kv_heads)
        v = self._shape(self.v_proj(x), self.num_kv_heads)

        if self.use_turboquant and past_kv:
            for h in range(self.num_kv_heads):
                self.kv_store.append(layer_id, h, k[:, h], v[:, h])
            k_full = torch.stack(
                [self.kv_store.load(layer_id, h)[0] for h in range(self.num_kv_heads)], dim=1
            )
            v_full = torch.stack(
                [self.kv_store.load(layer_id, h)[1] for h in range(self.num_kv_heads)], dim=1
            )
        else:
            k_full, v_full = k, v

        if self.num_kv_heads != self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(reps, dim=1)
            v_full = v_full.repeat_interleave(reps, dim=1)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_full) * self.scale
        mask = torch.triu(
            torch.full((s, k_full.shape[-2]), float("-inf"), device=x.device),
            diagonal=1 + k_full.shape[-2] - s,
        )
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_full)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)

    def reset_cache(self) -> None:
        if self.use_turboquant:
            self.kv_store.clear()
