"""EdgeMoE — main orchestrator.

Ties everything together:
    storage   →  cache   →  prefetcher   →  router / speculative router
                                         ↓
    attention (Standard or QJL)      expert FFN compute
                                         ↓
    TurboQuant KV cache ←  shared ←  speculative decoder

This file is deliberately lean: the heavy logic lives in the submodules.
The engine only wires them together and runs the generation loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from edgemoe.cache import MLCache
from edgemoe.prefetch import AsyncPrefetcher
from edgemoe.router import MoERouter
from edgemoe.speculative_router import MoESpeculativeRouter
from edgemoe.quantization.adaptive import AdaptiveQuantizer
from edgemoe.quantization.bitnet import BitNetExpertQuantizer
from edgemoe.attention.standard import StandardAttention
from edgemoe.attention.qjl import QJLAttention
from edgemoe.storage import get_backend, StorageBackend


@dataclass
class EngineConfig:
    model: str
    backend: str = "huggingface"
    backend_kwargs: dict = field(default_factory=dict)
    vram_budget_gb: float = 11.0
    ram_buffer_gb: float = 15.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    attention: str = "standard"          # "standard" | "qjl"
    qjl_sketch_dim: int = 128
    turboquant_bits: int = 3
    adaptive_quant: bool = True
    bitnet_experts: bool = False
    expert_clustering: bool = False
    speculative_model: str | None = None
    speculative_router: bool = True
    speculative_router_threshold: float = 0.85
    prefetch_lookahead: int = 3
    top_k_experts: int = 8
    dtype: torch.dtype = torch.float16


class ExpertBank:
    """Looks up an expert's weights: cache → prefetch → storage."""

    def __init__(
        self,
        storage: StorageBackend,
        cache: MLCache,
        prefetcher: AsyncPrefetcher,
        quantizer: AdaptiveQuantizer,
        bitnet: BitNetExpertQuantizer | None,
        device: str,
        dtype: torch.dtype,
    ):
        self.storage = storage
        self.cache = cache
        self.prefetcher = prefetcher
        self.quantizer = quantizer
        self.bitnet = bitnet
        self.device = device
        self.dtype = dtype

    def get(self, layer_id: int, expert_id: int) -> torch.Tensor:
        cached = self.cache.get(layer_id, expert_id)
        if cached is not None:
            return cached
        raw = self.prefetcher.get(layer_id, expert_id, block=False)
        if raw is None:
            raw = self.storage.load_expert(layer_id, expert_id)
        record = self._deserialize(raw)
        if record["dtype"] == "ternary":
            w = BitNetExpertQuantizer.dequantize(record)
        else:
            w = self.quantizer.dequantize(record)
        w = w.to(self.device, dtype=self.dtype)
        self.cache.put(layer_id, expert_id, w)
        self.cache.record_access(layer_id, expert_id)
        return w

    @staticmethod
    def _deserialize(raw: bytes) -> dict:
        """Decode one of our on-disk expert records from bytes.

        The on-disk format is a tiny JSON header (length-prefixed) +
        raw tensor payloads. See tools/split_experts.py for the writer.
        """
        header_len = int.from_bytes(raw[:4], "little")
        header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
        payload = raw[4 + header_len :]
        return _rehydrate(header, payload)


def _rehydrate(header: dict, payload: bytes) -> dict:
    """Turn header+payload back into a tensor record dict."""
    record: dict = {"dtype": header.get("dtype", "group_asym")}
    offset = 0
    for name, meta in header["tensors"].items():
        nbytes = meta["nbytes"]
        dtype = getattr(torch, meta["torch_dtype"])
        shape = tuple(meta["shape"])
        t = torch.frombuffer(
            bytearray(payload[offset : offset + nbytes]), dtype=dtype
        ).view(*shape)
        record[name] = t
        offset += nbytes
    for k, v in header.items():
        if k != "tensors":
            record[k] = v
    if record["dtype"] == "group_asym":
        record["mode"] = "group_asym"
    elif record["dtype"] == "per_channel_sym":
        record["mode"] = "per_channel_sym"
    return record


class MoELayer(nn.Module):
    """One transformer block: attention + MoE FFN."""

    def __init__(
        self,
        layer_id: int,
        config: EngineConfig,
        hidden: int,
        num_heads: int,
        num_experts: int,
        num_kv_heads: int | None,
        bank: ExpertBank,
        router: MoERouter,
        speculative_router: MoESpeculativeRouter | None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.bank = bank
        self.router = router
        self.speculative_router = speculative_router
        if config.attention == "qjl":
            self.attn = QJLAttention(
                hidden, num_heads, config.qjl_sketch_dim,
                num_kv_heads=num_kv_heads, device=config.device,
            )
        else:
            self.attn = StandardAttention(
                hidden, num_heads, num_kv_heads=num_kv_heads, device=config.device,
            )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.norm1(x))
        x = x + self._moe(self.norm2(x))
        return x

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.attn, QJLAttention):
            return self.attn(x, self.layer_id)
        return self.attn(x, self.layer_id)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        expert_ids, weights, logits = self.router(x)       # (b, s, K)
        b, s, k = expert_ids.shape
        flat_x = x.view(-1, x.shape[-1])
        flat_ids = expert_ids.view(-1, k)
        flat_w = weights.view(-1, k)
        flat_logits = logits.view(-1, logits.shape[-1])

        out = torch.zeros_like(flat_x)
        for t in range(flat_x.shape[0]):
            if self.speculative_router is not None:
                pred, conf, eid, mode = self.speculative_router.route(
                    flat_x[t : t + 1], flat_logits[t : t + 1], flat_ids[t]
                )
                if mode == "speculative":
                    out[t] = pred.squeeze(0)
                    continue
            token_out = torch.zeros_like(flat_x[t])
            for k_idx in range(k):
                eid = int(flat_ids[t, k_idx].item())
                w_mat = self.bank.get(self.layer_id, eid)
                token_out += flat_w[t, k_idx] * (flat_x[t] @ w_mat.T)
            out[t] = token_out
            if self.speculative_router is not None:
                self.speculative_router.update_predictor(
                    flat_x[t : t + 1], flat_logits[t : t + 1], token_out.unsqueeze(0)
                )
        return out.view(b, s, -1)

    def predict_next_experts(self, x: torch.Tensor, top_k: int | None = None) -> list[int]:
        ids = self.router.predict_experts(x, top_k=top_k)
        return sorted(set(ids.flatten().tolist()))


class EdgeMoE:
    """Public entry point. Construct, then `.generate(prompt)`."""

    def __init__(self, **kwargs):
        config = EngineConfig(**kwargs) if not isinstance(
            kwargs.get("config"), EngineConfig
        ) else kwargs["config"]
        self.config = config

        self.storage = get_backend(
            config.backend, model_path=config.model, **config.backend_kwargs
        )
        self.manifest = self.storage.get_manifest()

        hidden = self.manifest.get("hidden_size", 4096)
        num_layers = self.manifest["num_layers"]
        num_experts = self.manifest["num_experts"]
        num_heads = self.manifest.get("num_heads", 32)
        num_kv_heads = self.manifest.get("num_kv_heads", num_heads)
        top_k = self.manifest.get("experts_per_token", config.top_k_experts)

        self.cache = MLCache(
            budget_bytes=int(config.vram_budget_gb * 1e9),
            num_layers=num_layers,
            num_experts=num_experts,
            device=config.device,
        )
        self.prefetcher = AsyncPrefetcher(
            self.storage,
            ram_buffer_bytes=int(config.ram_buffer_gb * 1e9),
            lookahead_layers=config.prefetch_lookahead,
        )
        self.quantizer = AdaptiveQuantizer()
        self.bitnet = BitNetExpertQuantizer() if config.bitnet_experts else None
        self.bank = ExpertBank(
            self.storage, self.cache, self.prefetcher,
            self.quantizer, self.bitnet, config.device, config.dtype,
        )

        # Shared router + optional speculative router (one per model, not per layer).
        self.router = MoERouter(hidden, num_experts, top_k=top_k).to(config.device)
        self.speculative_router = (
            MoESpeculativeRouter(
                hidden, num_experts,
                confidence_threshold=config.speculative_router_threshold,
                device=config.device,
            )
            if config.speculative_router else None
        )

        self.layers = nn.ModuleList([
            MoELayer(
                layer_id=L, config=config, hidden=hidden, num_heads=num_heads,
                num_experts=num_experts, num_kv_heads=num_kv_heads,
                bank=self.bank, router=self.router,
                speculative_router=self.speculative_router,
            )
            for L in range(num_layers)
        ]).to(config.device)

        self.embed = nn.Embedding(self.manifest.get("vocab_size", 128000), hidden).to(
            config.device
        )
        self.lm_head = nn.Linear(hidden, self.manifest.get("vocab_size", 128000), bias=False).to(
            config.device
        )
        self.norm_f = nn.LayerNorm(hidden).to(config.device)

        self._load_backbone()

    def _load_backbone(self) -> None:
        """Load backbone (attn + embed + lm_head) weights from storage.

        The backbone blob is a safetensors-compatible state_dict. We
        map by name into the modules we just created. Tolerant of missing
        keys — useful while running against partial test models.
        """
        try:
            raw = self.storage.load_backbone()
        except FileNotFoundError:
            return
        try:
            from safetensors.torch import load as safetensors_load
            state = safetensors_load(raw)
        except Exception:
            try:
                state = torch.load(__import__("io").BytesIO(raw), map_location="cpu")
            except Exception:
                return
        self._apply_state_dict(state)

    def _apply_state_dict(self, state: dict) -> None:
        own = {}
        own.update({f"embed.{k.removeprefix('embed.')}": v for k, v in state.items() if k.startswith("embed.")})
        own.update({f"lm_head.{k.removeprefix('lm_head.')}": v for k, v in state.items() if k.startswith("lm_head.")})
        own.update({f"norm_f.{k.removeprefix('norm_f.')}": v for k, v in state.items() if k.startswith("norm_f.")})
        for L, layer in enumerate(self.layers):
            prefix = f"layers.{L}."
            for k, v in state.items():
                if k.startswith(prefix):
                    own[k] = v
        try:
            missing, unexpected = self._load(state, strict=False)
            _ = missing, unexpected
        except Exception:
            pass

    def _load(self, state: dict, strict: bool = False):
        full: dict = {}
        for k, v in self.embed.state_dict().items():
            full[f"embed.{k}"] = v
        for k, v in self.lm_head.state_dict().items():
            full[f"lm_head.{k}"] = v
        for k, v in self.norm_f.state_dict().items():
            full[f"norm_f.{k}"] = v
        for i, layer in enumerate(self.layers):
            for k, v in layer.state_dict().items():
                full[f"layers.{i}.{k}"] = v
        merged = {**full, **{k: v for k, v in state.items() if k in full}}
        missing = [k for k in full if k not in state]
        unexpected = [k for k in state if k not in full]
        self.embed.load_state_dict(
            {k[len("embed."):]: v for k, v in merged.items() if k.startswith("embed.")},
            strict=False,
        )
        self.lm_head.load_state_dict(
            {k[len("lm_head."):]: v for k, v in merged.items() if k.startswith("lm_head.")},
            strict=False,
        )
        self.norm_f.load_state_dict(
            {k[len("norm_f."):]: v for k, v in merged.items() if k.startswith("norm_f.")},
            strict=False,
        )
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            layer.load_state_dict(
                {k[len(prefix):]: v for k, v in merged.items() if k.startswith(prefix)},
                strict=False,
            )
        return missing, unexpected

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for L, layer in enumerate(self.layers):
            # Predict experts for next layers + kick off prefetch
            predicted = {}
            for off in range(1, self.config.prefetch_lookahead + 1):
                if L + off < len(self.layers):
                    predicted[L + off] = layer.predict_next_experts(x)
            self.prefetcher.hint_next_layers(L, predicted)
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def generate(
        self,
        prompt: str | torch.Tensor,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_token_id: int | None = None,
    ) -> torch.Tensor:
        if isinstance(prompt, str):
            raise NotImplementedError(
                "String prompts require the tokenizer — use tools/split_experts.py "
                "which ships tokenizer.bin alongside the model, then pass pre-tokenised "
                "ids for now."
            )
        ids = prompt.to(self.config.device)
        for _ in range(max_tokens):
            logits = self._forward(ids)[:, -1]
            logits = logits / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs, 1).item())
            ids = torch.cat(
                [ids, torch.tensor([[tok]], device=ids.device, dtype=ids.dtype)],
                dim=-1,
            )
            if stop_token_id is not None and tok == stop_token_id:
                break
        return ids

    def benchmark(self, num_tokens: int = 100) -> dict:
        ids = torch.tensor([[1] * 8], device=self.config.device, dtype=torch.long)
        t0 = time.perf_counter()
        out = self.generate(ids, max_tokens=num_tokens, temperature=1.0)
        elapsed = time.perf_counter() - t0
        generated = out.shape[-1] - ids.shape[-1]
        return {
            "tokens": generated,
            "seconds": elapsed,
            "tok_s": generated / elapsed if elapsed > 0 else 0.0,
            "cache": self.cache.stats(),
            "spec_router": (
                self.speculative_router.stats() if self.speculative_router else None
            ),
        }

    def close(self) -> None:
        self.prefetcher.shutdown()
