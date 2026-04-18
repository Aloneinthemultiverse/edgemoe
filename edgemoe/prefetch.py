"""Proactive async prefetcher — hides storage latency behind GPU compute.

While the GPU runs layer N, we pull layers N+1..N+K into a RAM-resident
staging buffer. When the router actually picks experts for layer N+1,
the bytes are already local — cost becomes a memcpy to VRAM.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable

from edgemoe.storage.base import StorageBackend


class AsyncPrefetcher:
    """Background ring buffer of pre-fetched layer bytes."""

    def __init__(
        self,
        storage: StorageBackend,
        ram_buffer_bytes: int,
        lookahead_layers: int = 3,
        workers: int = 8,
    ):
        self.storage = storage
        self.budget = ram_buffer_bytes
        self.used = 0
        self.lookahead = lookahead_layers
        self.buffer: "OrderedDict[tuple[int, int], bytes]" = OrderedDict()
        self.in_flight: dict[tuple[int, int], Future] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="prefetch"
        )

    def hint_next_layers(
        self,
        current_layer: int,
        predicted_experts_per_layer: dict[int, Iterable[int]],
    ) -> None:
        """Kick off downloads for the next `lookahead` layers."""
        for offset in range(1, self.lookahead + 1):
            layer = current_layer + offset
            experts = predicted_experts_per_layer.get(layer, ())
            for eid in experts:
                self._schedule(layer, eid)

    def _schedule(self, layer_id: int, expert_id: int) -> None:
        key = (layer_id, expert_id)
        with self.lock:
            if key in self.buffer or key in self.in_flight:
                return
            self.in_flight[key] = self.pool.submit(self._fetch_sync, key)

    def _fetch_sync(self, key: tuple[int, int]) -> bytes:
        data = self.storage.load_expert(*key)
        with self.lock:
            self._evict_if_needed(len(data))
            self.buffer[key] = data
            self.used += len(data)
            self.in_flight.pop(key, None)
        return data

    def _evict_if_needed(self, incoming: int) -> None:
        while self.used + incoming > self.budget and self.buffer:
            _, data = self.buffer.popitem(last=False)
            self.used -= len(data)

    def get(self, layer_id: int, expert_id: int, block: bool = True) -> bytes | None:
        key = (layer_id, expert_id)
        with self.lock:
            data = self.buffer.pop(key, None)
            if data is not None:
                self.used -= len(data)
                return data
            fut = self.in_flight.get(key)
        if fut is not None:
            if not block:
                return None
            data = fut.result()
            with self.lock:
                self.buffer.pop(key, None)
            return data
        return None

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
