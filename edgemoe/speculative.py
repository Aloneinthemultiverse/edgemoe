"""Speculative decoding — draft model proposes, target model verifies.

Draft: a tiny ternary Llama3-8B on CPU.
Target: EdgeMoE-backed Qwen3-235B on the main engine.

Protocol:
  1. draft.generate(prompt, K)                 → K candidate tokens
  2. target.forward(prompt + candidates)       → distribution at each pos
  3. for t in candidates:
         accept with prob min(1, p_target(t) / p_draft(t))
         break on first reject; sample from corrected distribution
  4. always one target-model token per accepted prefix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class SpeculativeConfig:
    draft_model_id: str = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"
    gamma: int = 4                 # draft tokens per cycle
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256


class SpeculativeDecoder:
    """Coordinates draft + target for 2-3x speedup without quality loss."""

    def __init__(
        self,
        target_forward: Callable[[torch.Tensor], torch.Tensor],
        draft_forward: Callable[[torch.Tensor], torch.Tensor],
        config: SpeculativeConfig | None = None,
    ):
        self.target_forward = target_forward
        self.draft_forward = draft_forward
        self.cfg = config or SpeculativeConfig()
        self.accepted = 0
        self.proposed = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def _sample(self, logits: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = logits / max(self.cfg.temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        if 0 < self.cfg.top_p < 1:
            sorted_p, sorted_i = probs.sort(descending=True)
            cum = sorted_p.cumsum(dim=-1)
            mask = cum > self.cfg.top_p
            mask[..., 0] = False
            sorted_p = sorted_p.masked_fill(mask, 0)
            probs = torch.zeros_like(probs).scatter(-1, sorted_i, sorted_p)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        tok = int(torch.multinomial(probs, 1).item())
        return tok, probs

    def generate(
        self, input_ids: torch.Tensor, stop_token_id: int | None = None
    ) -> torch.Tensor:
        ids = input_ids.clone()
        remaining = self.cfg.max_new_tokens
        while remaining > 0:
            draft_tokens, draft_probs = [], []
            cur = ids.clone()
            for _ in range(self.cfg.gamma):
                logits = self.draft_forward(cur)[:, -1]
                tok, p = self._sample(logits)
                draft_tokens.append(tok)
                draft_probs.append(p)
                cur = torch.cat(
                    [cur, torch.tensor([[tok]], device=ids.device, dtype=ids.dtype)],
                    dim=-1,
                )

            target_logits = self.target_forward(cur)            # (1, L, V)
            accepted_this_round = 0
            for t, (tok, dprob) in enumerate(zip(draft_tokens, draft_probs)):
                tgt_logits = target_logits[:, -self.cfg.gamma - 1 + t]
                tgt_probs = F.softmax(tgt_logits / max(self.cfg.temperature, 1e-5), dim=-1)
                ratio = (tgt_probs[0, tok] / dprob[0, tok].clamp_min(1e-8)).clamp(max=1.0)
                if torch.rand(1, device=ids.device).item() < float(ratio):
                    ids = torch.cat(
                        [ids, torch.tensor([[tok]], device=ids.device, dtype=ids.dtype)],
                        dim=-1,
                    )
                    self.accepted += 1
                    self.proposed += 1
                    accepted_this_round += 1
                    if stop_token_id is not None and tok == stop_token_id:
                        return ids
                else:
                    self.proposed += 1
                    # Sample from the residual distribution
                    residual = (tgt_probs - dprob).clamp_min(0)
                    residual = residual / residual.sum().clamp_min(1e-8)
                    new_tok = int(torch.multinomial(residual, 1).item())
                    ids = torch.cat(
                        [ids, torch.tensor([[new_tok]], device=ids.device, dtype=ids.dtype)],
                        dim=-1,
                    )
                    break
            else:
                # All γ drafts accepted → one bonus sample from target.
                bonus_logits = target_logits[:, -1]
                bonus_tok, _ = self._sample(bonus_logits)
                ids = torch.cat(
                    [ids, torch.tensor([[bonus_tok]], device=ids.device, dtype=ids.dtype)],
                    dim=-1,
                )
                accepted_this_round += 1
            remaining -= accepted_this_round
            if stop_token_id is not None and int(ids[0, -1].item()) == stop_token_id:
                break
        return ids
