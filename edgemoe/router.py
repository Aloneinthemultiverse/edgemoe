"""MoE router — top-K expert selection per token."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """Standard learned router: linear projection → top-K softmax gating.

    The router is part of the model backbone (tiny, always in VRAM).
    We just need it to (a) pick which experts to activate and (b) give us
    gating weights to blend their outputs.
    """

    def __init__(self, hidden: int, num_experts: int, top_k: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (expert_ids [*, K], weights [*, K], full_logits [*, E])."""
        logits = self.gate(hidden_states)
        topk_logits, topk_ids = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_logits, dim=-1)
        return topk_ids, weights, logits

    @torch.no_grad()
    def predict_experts(
        self, hidden_states: torch.Tensor, top_k: int | None = None
    ) -> torch.Tensor:
        """Cheap forward without gradients — used by prefetcher."""
        k = top_k or self.top_k
        return self.gate(hidden_states).topk(k, dim=-1).indices
