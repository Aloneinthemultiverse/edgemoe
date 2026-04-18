"""Smoke tests — verify modules import and basic math works end-to-end.

These don't need a real model; they exercise tensor shapes and quant
round-trips so you catch most regressions in <5 seconds.
"""

from __future__ import annotations

import torch

from edgemoe.quantization.turboquant import TurboQuantKVCache
from edgemoe.quantization.adaptive import AdaptiveQuantizer
from edgemoe.quantization.bitnet import BitNetExpertQuantizer
from edgemoe.speculative_router import ExpertOutputPredictor
from edgemoe.cache import MLCache
from edgemoe.router import MoERouter


def test_turboquant_roundtrip():
    comp = TurboQuantKVCache(head_dim=64, device="cpu")
    kv = torch.randn(2, 4, 64)
    packed = comp.compress(kv)
    out = comp.decompress(packed)
    cos = torch.nn.functional.cosine_similarity(
        kv.flatten(0, 1), out.flatten(0, 1), dim=-1
    ).mean()
    assert cos > 0.7, f"3-bit KV round-trip cosine {cos:.3f} too low"


def test_adaptive_quant_roundtrip():
    q = AdaptiveQuantizer(group_size=128)
    w = torch.randn(256, 512)
    for bits in (4, 3, 2):
        packed = q.quantize(w, bits=bits)
        w2 = q.dequantize(packed)
        err = (w - w2).abs().mean() / w.abs().mean()
        tol = {4: 0.05, 3: 0.12, 2: 0.35}[bits]
        assert err < tol, f"{bits}-bit error {err:.3f} > {tol}"


def test_bitnet_roundtrip():
    w = torch.randn(256, 512)
    rec = BitNetExpertQuantizer.quantize(w)
    w2 = BitNetExpertQuantizer.dequantize(rec)
    assert w2.shape == w.shape
    # Ternary is lossy; only sanity-check direction is preserved.
    cos = torch.nn.functional.cosine_similarity(
        w.flatten().unsqueeze(0), w2.flatten().unsqueeze(0), dim=-1
    ).item()
    assert cos > 0.5


def test_ml_cache_evicts():
    cache = MLCache(
        budget_bytes=1024, num_layers=4, num_experts=4, device="cpu",
        warmup_samples=0,
    )
    for L in range(4):
        for E in range(4):
            cache.put(L, E, torch.zeros(64, dtype=torch.float32))
    assert cache.used <= cache.budget + 1024


def test_router_shapes():
    r = MoERouter(hidden=128, num_experts=8, top_k=2)
    x = torch.randn(1, 4, 128)
    ids, w, logits = r(x)
    assert ids.shape == (1, 4, 2)
    assert w.shape == (1, 4, 2)
    assert logits.shape == (1, 4, 8)


def test_spec_predictor_shapes():
    net = ExpertOutputPredictor(hidden=128, num_experts=8)
    tok = torch.randn(3, 128)
    logits = torch.randn(3, 8)
    pred, conf = net(tok, logits)
    assert pred.shape == (3, 128)
    assert conf.shape == (3,)


if __name__ == "__main__":
    test_turboquant_roundtrip()
    test_adaptive_quant_roundtrip()
    test_bitnet_roundtrip()
    test_ml_cache_evicts()
    test_router_shapes()
    test_spec_predictor_shapes()
    print("All smoke tests passed.")
