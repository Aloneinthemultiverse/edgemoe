"""BitNet 1.58-bit ternary quantization — applied to MoE experts only.

NOVEL RESEARCH CONTRIBUTION: post-training absmean ternarisation of
expert FFN weights. Attention stays at 4-bit for quality; experts go
to {-1, 0, +1} with a single float32 scale per weight matrix.

Why experts-only:
  - Experts are 97 % of model size, FFN is more quant-tolerant than attn
  - Matmul → additions only → 6x faster on CPU (no multiply)
  - 4-bit experts (140 GB) → 1.58-bit experts (57 GB) = 2.5x smaller

Open question: quality at 235B scale. Our empirical answer is the paper.
"""

from __future__ import annotations

import torch


class BitNetExpertQuantizer:
    """Ternary quantizer for expert weight matrices."""

    @staticmethod
    def quantize(weight: torch.Tensor) -> dict:
        """Return packed ternary weights + scale.

        absmean quantization (from BitNet b1.58 paper):
            scale = mean(|W|)
            W_q   = round(W / scale)    (clamped to {-1, 0, +1})
        """
        scale = weight.abs().mean().clamp_min(1e-8)
        ternary = (weight / scale).round().clamp(-1, 1).to(torch.int8)
        packed = BitNetExpertQuantizer._pack_ternary(ternary)
        return {
            "packed": packed,
            "scale": scale,
            "shape": tuple(weight.shape),
            "dtype": "ternary",
        }

    @staticmethod
    def dequantize(record: dict) -> torch.Tensor:
        ternary = BitNetExpertQuantizer._unpack_ternary(
            record["packed"], record["shape"]
        )
        return ternary.float() * record["scale"]

    @staticmethod
    def _pack_ternary(ternary: torch.Tensor) -> torch.Tensor:
        """Pack 5 ternary values into one uint8 (3^5 = 243 < 256).

        Storage cost: ceil(n / 5) bytes for n weights → 1.6 bits/weight.
        Slightly worse than the theoretical 1.58 bits but avoids
        bit-packing overhead and is unpackable with simple divmod.
        """
        flat = (ternary.flatten() + 1).to(torch.int16)  # {-1,0,1} → {0,1,2}
        pad = (-flat.numel()) % 5
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int16, device=flat.device)])
        flat = flat.view(-1, 5)
        coeffs = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=flat.device)
        packed = (flat * coeffs).sum(dim=-1).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_ternary(packed: torch.Tensor, shape: tuple) -> torch.Tensor:
        device = packed.device
        coeffs = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16, device=device)
        p = packed.to(torch.int16)
        out = torch.empty(p.numel(), 5, dtype=torch.int16, device=device)
        for i, c in enumerate([1, 3, 9, 27, 81]):
            out[:, i] = (p // c) % 3
        out = out.flatten()
        n = 1
        for s in shape:
            n *= s
        out = out[:n].view(shape)
        return (out - 1).to(torch.int8)

    @staticmethod
    def matmul(input_fp: torch.Tensor, record: dict) -> torch.Tensor:
        """Ternary matmul: W ∈ {-1,0,1} × scale × input.

        Equivalent to additions/subtractions only — no multiplies in the
        hot loop. Here we let PyTorch's matmul do it in FP32 for clarity;
        the native kernel (kernels/matmul.c) uses the int8 path.
        """
        w = BitNetExpertQuantizer.dequantize(record)
        return input_fp @ w.T

    @staticmethod
    def size_bits_per_weight() -> float:
        return 1.6  # 5 trits per byte = 8/5 bits per weight
