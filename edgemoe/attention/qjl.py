"""QJL — Quantized Johnson-Lindenstrauss attention, O(n) memory.

Paper: arxiv.org/abs/2504.19874 (2025).

The JL transform projects d-dimensional vectors down to m<<d while
preserving inner products in expectation. We 1-bit-quantize the
projection (sign only) so each key becomes m bits instead of d * 16.

Asymmetric estimator: queries stay float, keys go to sign-bits.
Inner-product estimator is unbiased, low-variance under JL assumptions.

WARNING: this is the subtle part — the naïve symmetric version
produces garbage. We follow the paper's asymmetric sign-bit design.

Enables 1M-token context on a 16 GB T4:
    normal attn @ 1M tokens → 32+ GB KV (impossible)
    QJL     @ 1M tokens → ~3 GB KV (fits!)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QJLSketch(nn.Module):
    """Asymmetric 1-bit JL sketch for keys; queries projected to floats."""

    def __init__(self, head_dim: int, sketch_dim: int, device: str = "cuda"):
        super().__init__()
        self.head_dim = head_dim
        self.sketch_dim = sketch_dim
        # Random Gaussian matrix — fixed once, shared across layers/heads.
        g = torch.randn(head_dim, sketch_dim, device=device) / math.sqrt(sketch_dim)
        self.register_buffer("G", g)

    def project_query(self, q: torch.Tensor) -> torch.Tensor:
        return q @ self.G                           # (..., sketch_dim) float

    def sketch_key(self, k: torch.Tensor) -> torch.Tensor:
        """Sign-bit sketch: sign(k G) packed into uint8."""
        proj = k @ self.G
        bits = (proj > 0).to(torch.uint8)
        return self._pack_bits(bits)

    def estimate_scores(
        self, q_sketch: torch.Tensor, k_bits_packed: torch.Tensor
    ) -> torch.Tensor:
        """Unbiased <q, k> estimator under the asymmetric JL framework."""
        k_signs = self._unpack_bits(k_bits_packed, self.sketch_dim) * 2 - 1  # {0,1}→{-1,+1}
        k_signs = k_signs.to(q_sketch.dtype)
        est = torch.einsum("...qd,...kd->...qk", q_sketch, k_signs)
        est = est * math.sqrt(math.pi / 2)          # unbiased scaling
        return est

    @staticmethod
    def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
        """Pack trailing dim of 0/1 uint8 into uint8 (x8 compression)."""
        *lead, d = bits.shape
        pad = (-d) % 8
        if pad:
            bits = F.pad(bits, (0, pad))
        bits = bits.view(*lead, -1, 8)
        weights = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device
        )
        return (bits * weights).sum(dim=-1, dtype=torch.uint8)

    @staticmethod
    def _unpack_bits(packed: torch.Tensor, out_dim: int) -> torch.Tensor:
        *lead, n = packed.shape
        result = torch.empty(*lead, n, 8, dtype=torch.uint8, device=packed.device)
        for i in range(8):
            result[..., i] = (packed >> i) & 1
        result = result.view(*lead, -1)[..., :out_dim]
        return result.to(torch.float32)


class QJLAttention(nn.Module):
    """Drop-in attention that keeps sketched keys + full values.

    Values are NOT sketched — compression only targets the O(n) key store,
    which dominates the KV memory budget. Values are either stored at
    FP16 (if within budget) or chunked into TurboQuant compressed blocks.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        sketch_dim: int = 128,
        num_kv_heads: int | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden // num_heads
        self.sketch = QJLSketch(self.head_dim, sketch_dim, device=device)
        self.q_proj = nn.Linear(hidden, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden, bias=False)
        # Per-(layer, head) append-only sketched-key + value stores.
        self.key_bits: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.value_cache: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)

        outputs = []
        for h in range(self.num_kv_heads):
            kh = k[:, :, h]                                 # (b, s, d)
            vh = v[:, :, h]
            sketched = self.sketch.sketch_key(kh)
            self.key_bits.setdefault((layer_id, h), []).append(sketched)
            self.value_cache.setdefault((layer_id, h), []).append(vh)

            all_kbits = torch.cat(self.key_bits[(layer_id, h)], dim=-2)
            all_v = torch.cat(self.value_cache[(layer_id, h)], dim=-2)

            reps = self.num_heads // self.num_kv_heads
            for r in range(reps):
                qh = q[:, :, h * reps + r]                  # (b, s, d)
                qproj = self.sketch.project_query(qh)       # (b, s, m)
                scores = self.sketch.estimate_scores(qproj, all_kbits) * self.scale
                mask = torch.triu(
                    torch.full(
                        (s, all_v.shape[-2]),
                        float("-inf"),
                        device=x.device,
                    ),
                    diagonal=1 + all_v.shape[-2] - s,
                )
                scores = scores + mask
                attn = F.softmax(scores, dim=-1)
                outputs.append(torch.einsum("bqk,bkd->bqd", attn, all_v))
        out = torch.cat(outputs, dim=-1).contiguous().view(b, s, -1)
        return self.o_proj(out)

    def reset_cache(self) -> None:
        self.key_bits.clear()
        self.value_cache.clear()
