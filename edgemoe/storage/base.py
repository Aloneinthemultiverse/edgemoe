"""Abstract storage backend. All backends share this interface."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ExpertRecord:
    layer_id: int
    expert_id: int
    offset: int
    size: int
    cluster: int = -1
    temp: str = "cold"      # hot / warm / cold
    bits: int = 4           # 4 / 3 / 2 / 1.58


class StorageBackend(abc.ABC):
    """Unified API over local disk / GDrive / HuggingFace."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._manifest: dict | None = None

    @abc.abstractmethod
    def load_expert(self, layer_id: int, expert_id: int) -> bytes:
        """Return raw bytes for one expert. Blocking."""

    @abc.abstractmethod
    def prefetch_layer(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        """Hint: start pulling these experts into local cache. Non-blocking."""

    @abc.abstractmethod
    def load_backbone(self) -> bytes:
        """Return the full backbone (attn + embeddings). Always loaded once."""

    def get_manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = self._read_manifest()
        return self._manifest

    @abc.abstractmethod
    def _read_manifest(self) -> dict:
        """Implementation-specific manifest fetch."""

    def get_expert_record(self, layer_id: int, expert_id: int) -> ExpertRecord:
        m = self.get_manifest()["experts"][str(layer_id)][str(expert_id)]
        return ExpertRecord(
            layer_id=layer_id,
            expert_id=expert_id,
            offset=m["offset"],
            size=m["size"],
            cluster=m.get("cluster", -1),
            temp=m.get("temp", "cold"),
            bits=m.get("bits", 4),
        )

    def iter_experts(self) -> Iterable[ExpertRecord]:
        m = self.get_manifest()["experts"]
        for layer_s, experts in m.items():
            for expert_s in experts:
                yield self.get_expert_record(int(layer_s), int(expert_s))

    @staticmethod
    def _parse_manifest_bytes(data: bytes) -> dict:
        return json.loads(data.decode("utf-8"))
