"""Per-expert adaptive quantization — hot 4-bit, warm 3-bit, cold 2-bit.

Expert temperature is tracked from activation frequency and refreshed
every `update_interval` tokens. Moving an expert across tiers is the
Quantizer's job; persistence lives in the file store.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass
class ExpertActivationStats:
    total_tokens: int = 0
    per_expert: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))

    def record(self, tokens: int, expert_ids: list[tuple[int, int]]) -> None:
        self.total_tokens += tokens
        for eid in expert_ids:
            self.per_expert[eid] += 1

    def frequency(self, layer_id: int, expert_id: int) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.per_expert[(layer_id, expert_id)] / self.total_tokens


class AdaptiveQuantizer:
    """Quantize float weights to 4 / 3 / 2 bits based on activation frequency.

    Tier thresholds (tuned from MoE literature):
        >5%   activation → hot  (4-bit, per-group asymmetric)
        1-5%  activation → warm (3-bit, per-group asymmetric)
        <1%   activation → cold (2-bit, per-channel symmetric)
    """

    HOT_THRESH = 0.05
    WARM_THRESH = 0.01

    def __init__(self, group_size: int = 128, update_interval: int = 1000):
        self.group_size = group_size
        self.update_interval = update_interval
        self.stats = ExpertActivationStats()

    def tier(self, freq: float) -> tuple[str, int]:
        if freq >= self.HOT_THRESH:
            return "hot", 4
        if freq >= self.WARM_THRESH:
            return "warm", 3
        return "cold", 2

    def quantize(self, weight: torch.Tensor, bits: int) -> dict:
        """Group-quant; returns packed dict.

        For bits ∈ {3,4}: per-group asymmetric with scale+zero_point.
        For bits=2:       per-channel symmetric (cheaper, slight quality hit).
        """
        if bits == 2:
            return self._quantize_per_channel_symmetric(weight, bits=2)
        return self._quantize_group_asymmetric(weight, bits=bits)

    def _quantize_group_asymmetric(self, w: torch.Tensor, bits: int) -> dict:
        qmax = (1 << bits) - 1
        rows, cols = w.shape
        assert cols % self.group_size == 0, "cols must be divisible by group_size"
        groups = w.view(rows, cols // self.group_size, self.group_size)
        wmin = groups.min(dim=-1, keepdim=True).values
        wmax = groups.max(dim=-1, keepdim=True).values
        scale = (wmax - wmin).clamp_min(1e-8) / qmax
        zp = (-wmin / scale).round().clamp(0, qmax)
        q = (groups / scale + zp).round().clamp(0, qmax).to(torch.uint8)
        return {
            "q": q.view(rows, cols),
            "scale": scale.squeeze(-1),
            "zp": zp.squeeze(-1).to(torch.uint8),
            "bits": bits,
            "group_size": self.group_size,
            "mode": "group_asym",
        }

    def _quantize_per_channel_symmetric(self, w: torch.Tensor, bits: int) -> dict:
        qmax = (1 << (bits - 1)) - 1
        scale = w.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8) / qmax
        q = (w / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
        return {
            "q": q,
            "scale": scale.squeeze(-1),
            "bits": bits,
            "mode": "per_channel_sym",
        }

    def dequantize(self, packed: dict) -> torch.Tensor:
        mode = packed["mode"]
        if mode == "group_asym":
            q = packed["q"].to(torch.float32)
            scale = packed["scale"].unsqueeze(-1)
            zp = packed["zp"].to(torch.float32).unsqueeze(-1)
            rows, cols = q.shape
            gs = packed["group_size"]
            q = q.view(rows, cols // gs, gs)
            w = (q - zp) * scale
            return w.view(rows, cols)
        if mode == "per_channel_sym":
            return packed["q"].to(torch.float32) * packed["scale"].unsqueeze(-1)
        raise ValueError(f"Unknown quant mode: {mode}")

    def should_update(self) -> bool:
        return self.stats.total_tokens % self.update_interval == 0
