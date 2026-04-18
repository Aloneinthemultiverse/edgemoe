"""HFEngine — the working Kaggle path.

Wraps a HuggingFace MoE model:
  - loads config only, builds an empty model on the meta device
  - replaces every `mlp.experts` ModuleList with StreamingExperts proxies
  - loads the backbone safetensors we produced via tools/split_experts.py
  - uses the HF tokenizer + `model.generate()` for real text output

Covers Qwen3-MoE, Mixtral, OLMoE, DeepSeek-MoE, Phi-MoE — anything whose
MoE block exposes `mlp.experts[i].{gate_proj, up_proj, down_proj}` and a
`mlp.gate` router.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from edgemoe.cache import MLCache
from edgemoe.prefetch import AsyncPrefetcher
from edgemoe.quantization.adaptive import AdaptiveQuantizer
from edgemoe.quantization.bitnet import BitNetExpertQuantizer
from edgemoe.storage import get_backend, StorageBackend


@dataclass
class HFEngineConfig:
    model: str
    backend: str = "local"
    backend_kwargs: dict = field(default_factory=dict)
    vram_budget_gb: float = 10.0
    ram_buffer_gb: float = 15.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    prefetch_lookahead: int = 2


def _rehydrate(header: dict, payload: bytes) -> dict:
    """Reader companion to tools.split_experts._serialize_tensor_record."""
    record: dict = {}
    offset = 0
    for name, meta in header["tensors"].items():
        nbytes = meta["nbytes"]
        dtype = getattr(torch, meta["torch_dtype"])
        shape = tuple(meta["shape"])
        buf = bytearray(payload[offset : offset + nbytes])
        t = torch.frombuffer(buf, dtype=dtype).view(*shape).clone()
        record[name] = t
        offset += nbytes
    for k, v in header.items():
        if k != "tensors":
            record[k] = v
    return record


class ExpertBank:
    """Dequantise-on-demand source of expert weight matrices."""

    def __init__(
        self,
        storage: StorageBackend,
        cache: MLCache,
        prefetcher: AsyncPrefetcher,
        device: str,
        dtype: torch.dtype,
    ):
        self.storage = storage
        self.cache = cache
        self.prefetcher = prefetcher
        self.device = device
        self.dtype = dtype
        self.quantizer = AdaptiveQuantizer()

    def get(self, layer_id: int, expert_id: int) -> dict[str, torch.Tensor]:
        cached = self.cache.get(layer_id, expert_id)
        if cached is not None:
            return cached
        raw = self.prefetcher.get(layer_id, expert_id, block=False)
        if raw is None:
            raw = self.storage.load_expert(layer_id, expert_id)
        record = self._deserialize(raw)
        projections = self._dequantize_all(record)
        projections = {
            k: v.to(self.device, dtype=self.dtype, non_blocking=True)
            for k, v in projections.items()
        }
        self.cache.put(layer_id, expert_id, projections)
        self.cache.record_access(layer_id, expert_id)
        return projections

    @staticmethod
    def _deserialize(raw: bytes) -> dict:
        header_len = struct.unpack("<I", raw[:4])[0]
        header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
        payload = raw[4 + header_len :]
        return _rehydrate(header, payload)

    def _dequantize_all(self, record: dict) -> dict[str, torch.Tensor]:
        """Group `.q/.scale/.zp` (or `.packed/.scale/.shape`) keys by base name."""
        dtype = record.get("dtype", "group_asym")
        suffixes = (".q", ".scale", ".zp", ".packed", ".shape")
        bases: set[str] = set()
        for key in record:
            for suf in suffixes:
                if key.endswith(suf):
                    bases.add(key[: -len(suf)])
                    break

        out: dict[str, torch.Tensor] = {}
        for base in bases:
            if dtype == "ternary":
                scale_field = record[f"{base}.scale"]
                scale = float(
                    scale_field.item() if torch.is_tensor(scale_field) and scale_field.numel() == 1
                    else scale_field[0].item() if torch.is_tensor(scale_field) else scale_field
                )
                shape_field = record.get(f"{base}.shape", ())
                shape = tuple(shape_field) if not torch.is_tensor(shape_field) else tuple(shape_field.tolist())
                out[base] = BitNetExpertQuantizer.dequantize({
                    "packed": record[f"{base}.packed"],
                    "scale": scale,
                    "shape": shape,
                })
            else:
                q_rec = {
                    "q": record[f"{base}.q"],
                    "scale": record[f"{base}.scale"],
                    "bits": 4,
                    "group_size": 128,
                    "mode": "group_asym",
                }
                if f"{base}.zp" in record:
                    q_rec["zp"] = record[f"{base}.zp"]
                out[base] = self.quantizer.dequantize(q_rec)
        return out


class StreamingExpert(nn.Module):
    """One HF expert, but weights come from ExpertBank on every call.

    Mirrors `Qwen3MoeMlpBlock` / `MixtralBlockSparseTop2MLP`:
        out = down_proj( silu(gate_proj(x)) * up_proj(x) )
    """

    def __init__(
        self,
        bank: ExpertBank,
        layer_id: int,
        expert_id: int,
    ):
        super().__init__()
        self.bank = bank
        self.layer_id = layer_id
        self.expert_id = expert_id

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.bank.get(self.layer_id, self.expert_id)
        gate_w = _find(proj, "gate_proj")
        up_w = _find(proj, "up_proj")
        down_w = _find(proj, "down_proj")
        gate = F.silu(hidden_states @ gate_w.T)
        up = hidden_states @ up_w.T
        return (gate * up) @ down_w.T


def _find(proj: dict[str, torch.Tensor], needle: str) -> torch.Tensor:
    for k, v in proj.items():
        if needle in k:
            return v
    raise KeyError(f"{needle} missing from expert projections: {list(proj)}")


class StreamingExperts(nn.ModuleList):
    """Drop-in replacement for `mlp.experts`: indexable, iterable, but empty."""

    def __init__(self, bank: ExpertBank, layer_id: int, num_experts: int):
        super().__init__(
            [StreamingExpert(bank, layer_id, i) for i in range(num_experts)]
        )


class PrefetchHook:
    """Forward hook on `mlp.gate` that peeks at router output and kicks off
    async prefetch for the *next* layer's top-K experts."""

    def __init__(
        self,
        prefetcher: AsyncPrefetcher,
        next_layer_id: int | None,
        top_k: int,
    ):
        self.prefetcher = prefetcher
        self.next_layer_id = next_layer_id
        self.top_k = top_k

    def __call__(self, module, inputs, output):
        if self.next_layer_id is None:
            return
        logits = output if torch.is_tensor(output) else output[0]
        ids = logits.topk(self.top_k, dim=-1).indices
        eids = sorted(set(ids.flatten().tolist()))
        self.prefetcher.hint_next_layers(
            self.next_layer_id - 1, {self.next_layer_id: eids}
        )


class HFEngine:
    """Public API. `HFEngine(model=..., backend=...).generate("hello")`."""

    def __init__(self, **kwargs):
        if isinstance(kwargs.get("config"), HFEngineConfig):
            self.config = kwargs["config"]
        else:
            self.config = HFEngineConfig(**kwargs)

        self.storage = get_backend(
            self.config.backend,
            model_path=self.config.model,
            **self.config.backend_kwargs,
        )
        self.manifest = self.storage.get_manifest()
        num_layers = self.manifest["num_layers"]
        num_experts = self.manifest["num_experts"]
        top_k = self.manifest.get("experts_per_token", 8)

        self.cache = MLCache(
            budget_bytes=int(self.config.vram_budget_gb * 1e9),
            num_layers=num_layers,
            num_experts=num_experts,
            device=self.config.device,
        )
        self.prefetcher = AsyncPrefetcher(
            self.storage,
            ram_buffer_bytes=int(self.config.ram_buffer_gb * 1e9),
            lookahead_layers=self.config.prefetch_lookahead,
        )
        self.bank = ExpertBank(
            self.storage, self.cache, self.prefetcher,
            self.config.device, self.config.dtype,
        )

        self.model, self.tokenizer = self._build_hf_model(top_k)

    def _model_source(self) -> str:
        """Where HF config + tokenizer + backbone live.

        For `backend=local`: `config.model` is already a path with
        `config.json`, tokenizer files, and `backbone.bin` on disk.
        For `backend=huggingface`: we snapshot-download just the small
        metadata + backbone, experts still stream on demand.
        """
        if self.config.backend == "local":
            return str(self.config.model)
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=self.config.model,
            allow_patterns=[
                "config.json", "tokenizer*", "*.model",
                "special_tokens_map.json", "generation_config.json",
                "backbone.bin",
            ],
        )

    def _build_hf_model(self, top_k: int):
        from transformers import (
            AutoConfig, AutoModelForCausalLM, AutoTokenizer,
        )
        src = self._model_source()
        cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

        # Build the model shell on meta — no expert memory allocated.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                cfg, trust_remote_code=True, torch_dtype=self.config.dtype,
            )

        # Swap every `mlp.experts` with streaming proxies.
        num_layers = self.manifest["num_layers"]
        num_experts = self.manifest["num_experts"]
        for L, layer in enumerate(_get_layers(model)):
            mlp = _get_mlp(layer)
            if mlp is None:
                continue
            next_layer_id = L + 1 if L + 1 < num_layers else None
            for attr in ("experts",):
                if hasattr(mlp, attr):
                    setattr(mlp, attr, StreamingExperts(self.bank, L, num_experts))
            # Prefetch hook on the router
            for gate_attr in ("gate", "router"):
                gate = getattr(mlp, gate_attr, None)
                if isinstance(gate, nn.Module):
                    gate.register_forward_hook(
                        PrefetchHook(self.prefetcher, next_layer_id, top_k)
                    )
                    break

        # Load backbone weights — `assign=True` swaps meta tensors for real ones.
        state = self._load_backbone_state()
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        # Any param still on meta that wasn't in state → zero-init on device.
        # (Typically only the router gate would be unexpected-case; backbone covers the rest.)
        for name, param in model.named_parameters():
            if param.is_meta:
                real = torch.zeros(
                    param.shape, dtype=self.config.dtype, device=self.config.device,
                )
                _set_attr_tensor(model, name, real)

        model.to(self.config.device, dtype=self.config.dtype)
        model.eval()
        return model, tok

    def _load_backbone_state(self) -> dict[str, torch.Tensor]:
        raw = self.storage.load_backbone()
        try:
            from safetensors.torch import load as safetensors_load
            return safetensors_load(raw)
        except Exception:
            import io
            return torch.load(io.BytesIO(raw), map_location="cpu")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | torch.Tensor,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        if isinstance(prompt, str):
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        else:
            ids = prompt
        ids = ids.to(self.config.device)
        out = self.model.generate(
            ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)

    def benchmark(self, prompt: str = "Hello world.", num_tokens: int = 64) -> dict:
        import time
        t0 = time.perf_counter()
        text = self.generate(prompt, max_tokens=num_tokens, temperature=0.0)
        elapsed = time.perf_counter() - t0
        return {
            "text": text,
            "tokens": num_tokens,
            "seconds": round(elapsed, 3),
            "tok_s": round(num_tokens / elapsed, 2) if elapsed > 0 else 0.0,
            "cache": self.cache.stats(),
        }

    def close(self):
        self.prefetcher.shutdown()


def _get_layers(model: nn.Module):
    for attr in ("model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "layers"):
            return sub.layers
    return model.layers  # fallback


def _get_mlp(layer: nn.Module):
    for attr in ("mlp", "block_sparse_moe", "moe"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    return None


def _set_attr_tensor(root: nn.Module, dotted: str, tensor: torch.Tensor) -> None:
    *path, leaf = dotted.split(".")
    obj = root
    for p in path:
        obj = getattr(obj, p)
    if isinstance(getattr(obj, leaf, None), nn.Parameter):
        setattr(obj, leaf, nn.Parameter(tensor, requires_grad=False))
    else:
        obj.register_buffer(leaf, tensor)
