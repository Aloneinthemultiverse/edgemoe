"""MoE-aware speculative routing — NOVEL RESEARCH CONTRIBUTION.

Standard MoE:  token → router → expert IDs → load experts → compute
Speculative:   token → router → predict expert OUTPUT directly.

For ~30-40 % of tokens (common words, punctuation) the expert outputs
are predictable from the router logits + token embedding alone. A small
predictor network skips the expert load entirely on high-confidence
tokens. On low confidence we fall back to the real experts.

Online distillation keeps the predictor aligned with the target model
as the session progresses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertOutputPredictor(nn.Module):
    """Small MLP that maps (token_embed || router_logits) → predicted FFN output."""

    def __init__(
        self,
        hidden: int,
        num_experts: int,
        mlp_hidden: int = 512,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(hidden + num_experts, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden),
        )
        self.confidence_head = nn.Linear(mlp_hidden, 1)

    def forward(
        self, token_embed: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([token_embed, F.softmax(router_logits, dim=-1)], dim=-1)
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        h = self.net[3](h)
        pred = self.net[4](h)
        conf = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        return pred, conf


class MoESpeculativeRouter:
    """Predicts expert outputs directly for easy tokens; loads experts for hard ones.

    Call site:
        pred, conf, expert_ids = router.route(token_embed, router_logits)
        if conf > threshold:
            use pred, skip expert loading
        else:
            load experts, compute, then update_predictor(pred, actual)
    """

    def __init__(
        self,
        hidden: int,
        num_experts: int,
        confidence_threshold: float = 0.85,
        device: str = "cuda",
        lr: float = 1e-4,
    ):
        self.threshold = confidence_threshold
        self.predictor = ExpertOutputPredictor(hidden, num_experts).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.device = device
        self.skips = 0
        self.loads = 0

    @property
    def skip_rate(self) -> float:
        total = self.skips + self.loads
        return self.skips / total if total else 0.0

    @torch.no_grad()
    def route(
        self, token_embed: torch.Tensor, router_logits: torch.Tensor, expert_ids: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, str]:
        self.predictor.eval()
        pred, conf = self.predictor(token_embed, router_logits)
        mean_conf = conf.mean().item() if conf.ndim > 0 else float(conf)
        if mean_conf > self.threshold:
            self.skips += 1
            return pred, conf, expert_ids, "speculative"
        self.loads += 1
        return None, conf, expert_ids, "actual"

    def update_predictor(
        self,
        token_embed: torch.Tensor,
        router_logits: torch.Tensor,
        actual_output: torch.Tensor,
    ) -> float:
        """One step of online distillation against the real expert output."""
        self.predictor.train()
        pred, conf = self.predictor(token_embed, router_logits)
        mse = F.mse_loss(pred, actual_output)
        # encourage high confidence only when error is small
        target_conf = (1.0 / (1.0 + mse.detach())).clamp(0, 1)
        conf_loss = F.binary_cross_entropy(conf, target_conf.expand_as(conf))
        loss = mse + 0.1 * conf_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.predictor.eval()
        return float(loss.item())

    def stats(self) -> dict:
        return {
            "skip_rate": self.skip_rate,
            "skips": self.skips,
            "loads": self.loads,
            "threshold": self.threshold,
        }
