"""Expert VRAM cache with LSTM-predicted retention + LRU fallback."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn


@dataclass
class CachedExpert:
    layer_id: int
    expert_id: int
    payload: Any               # tensor OR dict of tensors (gate/up/down)
    size_bytes: int


def _payload_bytes(payload: Any) -> int:
    """Best-effort byte count for a cache payload (tensor or dict of tensors)."""
    if torch.is_tensor(payload):
        return payload.element_size() * payload.nelement()
    if isinstance(payload, dict):
        return sum(_payload_bytes(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_payload_bytes(v) for v in payload)
    return 0


class ExpertPredictorLSTM(nn.Module):
    """Tiny LSTM that predicts the probability each expert is needed next.

    Input: sequence of (layer_id, expert_id) pairs as one-hot / embeddings.
    Output: logits over (num_layers * num_experts).

    Kept deliberately small (~50 KB) — we care about signal, not accuracy.
    """

    def __init__(self, num_layers: int, num_experts: int, hidden: int = 64):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.total = num_layers * num_experts
        self.embed = nn.Embedding(self.total + 1, hidden)  # +1 for pad
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, self.total)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        h, _ = self.lstm(x)
        return self.head(h[:, -1])

    def encode(self, layer_id: int, expert_id: int) -> int:
        return layer_id * self.num_experts + expert_id


class MLCache:
    """VRAM-bounded expert cache with LSTM-predicted retention.

    On eviction we score every cached expert by:
        score = predicted_next_prob + recency_bonus
    and drop the lowest. When the predictor is cold (too few samples),
    we fall back to pure LRU.
    """

    def __init__(
        self,
        budget_bytes: int,
        num_layers: int,
        num_experts: int,
        device: str = "cuda",
        history_len: int = 32,
        warmup_samples: int = 500,
    ):
        self.budget = budget_bytes
        self.used = 0
        self.device = device
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.warmup_samples = warmup_samples
        self.samples_seen = 0
        self.history: deque[int] = deque(maxlen=history_len)
        self.entries: OrderedDict[tuple[int, int], CachedExpert] = OrderedDict()
        self.lock = threading.Lock()
        self.predictor = ExpertPredictorLSTM(num_layers, num_experts).to(device)
        self.predictor.eval()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def record_access(self, layer_id: int, expert_id: int) -> None:
        code = self.predictor.encode(layer_id, expert_id)
        self.history.append(code)
        self.samples_seen += 1

    def get(self, layer_id: int, expert_id: int) -> Any:
        key = (layer_id, expert_id)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return entry.payload

    def put(
        self,
        layer_id: int,
        expert_id: int,
        payload: Any,
        size_bytes: int | None = None,
    ) -> None:
        key = (layer_id, expert_id)
        if size_bytes is None:
            size_bytes = _payload_bytes(payload)
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return
            while self.used + size_bytes > self.budget and self.entries:
                self._evict_one()
            self.entries[key] = CachedExpert(layer_id, expert_id, payload, size_bytes)
            self.used += size_bytes

    def _evict_one(self) -> None:
        if self.samples_seen < self.warmup_samples:
            key, entry = self.entries.popitem(last=False)
            self.used -= entry.size_bytes
            return
        scores = self._score_cached()
        victim_key = min(scores, key=scores.get)
        victim = self.entries.pop(victim_key)
        self.used -= victim.size_bytes

    def _score_cached(self) -> dict[tuple[int, int], float]:
        if not self.history:
            return {k: i for i, k in enumerate(self.entries)}
        ids = torch.tensor(
            [list(self.history)], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            probs = torch.softmax(self.predictor(ids), dim=-1).squeeze(0)
        out: dict[tuple[int, int], float] = {}
        for i, key in enumerate(self.entries):
            code = self.predictor.encode(*key)
            recency = i / max(1, len(self.entries))
            out[key] = float(probs[code].item()) + 0.1 * recency
        return out

    def online_update(self, next_accesses: Iterable[tuple[int, int]]) -> None:
        """Teach the predictor the actual next-step access pattern."""
        if len(self.history) < 4:
            return
        ids = torch.tensor(
            [list(self.history)[:-1]], dtype=torch.long, device=self.device
        )
        targets = torch.tensor(
            [self.predictor.encode(*next(iter(next_accesses)))],
            dtype=torch.long,
            device=self.device,
        )
        self.predictor.train()
        logits = self.predictor(ids)
        loss = nn.functional.cross_entropy(logits, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.predictor.eval()

    def stats(self) -> dict:
        return {
            "used_mb": self.used / 1e6,
            "budget_mb": self.budget / 1e6,
            "entries": len(self.entries),
            "hit_rate": self.hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "samples_seen": self.samples_seen,
        }
