"""Expert clustering — groups co-activated experts into shared files.

Offline pass: run calibration data, log which experts fire together,
then cluster by co-activation frequency. At inference we fetch one
cluster-file per GPU step instead of K tiny expert files — 3-4x fewer
I/O ops, which is the dominant cost on Google Drive.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


class ExpertCoActivationLogger:
    """Records which experts fire in the same forward pass, per layer."""

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.counts = {
            L: np.zeros((num_experts, num_experts), dtype=np.int64)
            for L in range(num_layers)
        }
        self.totals = np.zeros(num_layers, dtype=np.int64)

    def log(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        ids = list(expert_ids)
        mat = self.counts[layer_id]
        for i in ids:
            for j in ids:
                mat[i, j] += 1
        self.totals[layer_id] += 1

    def save(self, path: str | Path) -> None:
        p = Path(path)
        np.savez_compressed(
            p,
            totals=self.totals,
            **{f"L{L}": self.counts[L] for L in range(self.num_layers)},
        )


def cluster_layer(
    coactivation: np.ndarray, cluster_size: int = 4, seed: int = 0
) -> list[list[int]]:
    """Greedy agglomerative clustering on a co-activation matrix.

    Each cluster = `cluster_size` experts that fire together most often.
    Not optimal, but fast and good enough to cut I/O by ~3x in practice.
    """
    rng = np.random.default_rng(seed)
    num_experts = coactivation.shape[0]
    remaining = set(range(num_experts))
    clusters: list[list[int]] = []
    sym = (coactivation + coactivation.T) / 2
    np.fill_diagonal(sym, 0)

    while remaining:
        if len(remaining) <= cluster_size:
            clusters.append(sorted(remaining))
            break
        seed_expert = int(rng.choice(list(remaining)))
        scores = sym[seed_expert]
        candidates = sorted(
            remaining,
            key=lambda e: -scores[e] if e != seed_expert else float("-inf"),
        )
        picked = [seed_expert] + candidates[1:cluster_size]
        for e in picked:
            remaining.discard(e)
        clusters.append(sorted(picked))
    return clusters


def build_cluster_manifest(
    logs_path: str | Path, cluster_size: int = 4
) -> dict:
    """Turn a saved coactivation log into a {layer: clusters} manifest."""
    logs_path = Path(logs_path)
    data = np.load(logs_path)
    totals = data["totals"]
    num_layers = len(totals)
    out = {"cluster_size": cluster_size, "layers": {}}
    for L in range(num_layers):
        mat = data[f"L{L}"]
        clusters = cluster_layer(mat, cluster_size=cluster_size, seed=L)
        out["layers"][str(L)] = clusters
    return out


def save_cluster_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2))
