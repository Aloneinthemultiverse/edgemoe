"""HFEngine — the working Kaggle path.

Wraps a HuggingFace MoE model:
  - loads config only, builds the model on CPU (so RoPE + other buffers
    are properly initialised)
  - replaces every `mlp.experts` ModuleList with StreamingExperts proxies
    and collects to free the expert weights we just allocated
  - loads the backbone safetensors produced via tools/split_experts.py
  - moves the now-slim backbone onto `device` and runs `model.generate()`

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
        """Group `.q/.scale/.zp/.bits/.group_size/.shape` (or `.packed/.scale/.shape`) keys by base name."""
        dtype = record.get("dtype", "group_asym")
        suffixes = (".q", ".scale", ".zp", ".packed", ".shape", ".bits", ".group_size")
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
                bits_field = record.get(f"{base}.bits", 4)
                bits = int(bits_field.item() if torch.is_tensor(bits_field) else bits_field)
                gs_field = record.get(f"{base}.group_size", 128)
                group_size = int(gs_field.item() if torch.is_tensor(gs_field) else gs_field)
                q_rec = {
                    "q": record[f"{base}.q"],
                    "scale": record[f"{base}.scale"],
                    "bits": bits,
                    "group_size": group_size,
                    "mode": "group_asym",
                }
                shape_field = record.get(f"{base}.shape")
                if shape_field is not None:
                    q_rec["shape"] = (
                        tuple(shape_field.tolist()) if torch.is_tensor(shape_field)
                        else tuple(shape_field)
                    )
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
    """Drop-in replacement for `mlp.experts`.

    Two dispatch conventions exist in HF MoE models:
      - Mixtral / Qwen3-MoE / DeepSeek-MoE iterate `self.experts[i]` and
        call the expert directly. Iteration + indexing already works via
        `nn.ModuleList`.
      - OLMoE / Phi-MoE call `self.experts(hidden_states, top_k_index,
        top_k_weights)` as a single fused callable. That path needs the
        `forward` below.
    """

    def __init__(self, bank: ExpertBank, layer_id: int, num_experts: int):
        super().__init__(
            [StreamingExpert(bank, layer_id, i) for i in range(num_experts)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[-1]
        flat_experts = top_k_index.reshape(-1)
        flat_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
        row_idx = torch.arange(
            num_tokens, device=hidden_states.device
        ).repeat_interleave(top_k)

        final = torch.zeros_like(hidden_states)
        for eid in torch.unique(flat_experts).tolist():
            mask = flat_experts == eid
            tok_idx = row_idx[mask]
            weights = flat_weights[mask].unsqueeze(-1)
            y = self[eid](hidden_states[tok_idx]) * weights
            final.index_add_(0, tok_idx, y)
        return final


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
        import gc
        from transformers import (
            AutoConfig, AutoModelForCausalLM, AutoTokenizer,
        )
        src = self._model_source()
        cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

        # Build on meta → zero CPU allocation, even for a 30B model.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                cfg, trust_remote_code=True, torch_dtype=self.config.dtype,
            )

        # Swap expert ModuleLists FIRST. The children hold no tensors of
        # their own (they route through `ExpertBank`), so `to_empty` below
        # won't allocate the experts we're streaming.
        num_layers = self.manifest["num_layers"]
        num_experts = self.manifest["num_experts"]
        for L, layer in enumerate(_get_layers(model)):
            mlp = _get_mlp(layer)
            if mlp is None:
                continue
            next_layer_id = L + 1 if L + 1 < num_layers else None
            if hasattr(mlp, "experts"):
                setattr(mlp, "experts", StreamingExperts(self.bank, L, num_experts))
            for gate_attr in ("gate", "router"):
                gate = getattr(mlp, gate_attr, None)
                if isinstance(gate, nn.Module):
                    gate.register_forward_hook(
                        PrefetchHook(self.prefetcher, next_layer_id, top_k)
                    )
                    break

        # Materialise the *remaining* meta tensors (backbone + router
        # gates + layernorms + embeddings) with empty CPU memory. For a
        # Qwen3-30B-A3B that's ~3 GB; for OLMoE-1B it's ~1 GB.
        model = model.to_empty(device="cpu")

        # `to_empty` leaves non-persistent buffers (RoPE `inv_freq`, etc.)
        # filled with garbage. Re-run their init so attention works.
        _reinit_rope_buffers(model)

        # Load real backbone weights into the empty tensors (in-place).
        state = self._load_backbone_state()
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Anything in `missing` that is NOT an expert tensor is uninitialised
        # garbage from `to_empty` → that breaks the forward. Surface it loudly.
        leaked = [
            k for k in missing
            if ".experts." not in k and "rotary_emb" not in k
        ]
        if leaked:
            print(
                f"[hf_engine] WARNING: {len(leaked)} non-expert params missing "
                f"from backbone state — forward will NaN. First 10: {leaked[:10]}"
            )
        if unexpected:
            print(f"[hf_engine] unexpected keys in backbone: {unexpected[:10]}")

        model = model.to(self.config.device, dtype=self.config.dtype)
        model.eval()
        gc.collect()
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            enc = self.tokenizer(prompt, return_tensors="pt")
            ids = enc.input_ids
            attn = enc.attention_mask
        else:
            ids = prompt
            attn = torch.ones_like(ids)
        ids = ids.to(self.config.device)
        attn = attn.to(self.config.device)
        out = self.model.generate(
            ids,
            attention_mask=attn,
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


def _reinit_rope_buffers(model: nn.Module) -> None:
    """Recompute `inv_freq` on every RotaryEmbedding module.

    After `to_empty()`, non-persistent buffers (like RoPE's `inv_freq`)
    hold uninitialised memory — we've seen them come back as `[-inf..0]`,
    which saturates attention and produces NaN logits.

    We try `ROPE_INIT_FUNCTIONS` first (handles yarn / linear / ntk /
    longrope rope_scaling variants), and fall back to the plain
    `inv_freq = 1 / theta^(2i/d)` formula from the default rope.
    """
    root_cfg = getattr(model, "config", None)
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:  # noqa: BLE001
        ROPE_INIT_FUNCTIONS = {}

    patched = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Rotary" not in cls_name:
            continue
        cfg = getattr(module, "config", None) or root_cfg
        if cfg is None:
            print(f"[hf_engine] skip rotary {name}: no config")
            continue

        inv_freq = None
        scaling = 1.0
        # Preferred: use transformers' own init so rope_scaling is honoured.
        try:
            rope_scaling = getattr(cfg, "rope_scaling", None) or {}
            rope_type = (
                rope_scaling.get("rope_type")
                or rope_scaling.get("type")
                or "default"
            )
            fn = ROPE_INIT_FUNCTIONS.get(rope_type)
            if fn is not None:
                inv_freq, scaling = fn(cfg, device=None)
        except Exception as exc:  # noqa: BLE001
            print(f"[hf_engine] rope_init_fn failed for {name}: {exc}")

        # Fallback: default RoPE formula from config fields.
        if inv_freq is None:
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None:
                hidden = getattr(cfg, "hidden_size", None)
                n_heads = getattr(cfg, "num_attention_heads", None)
                if hidden and n_heads:
                    head_dim = hidden // n_heads
            if head_dim is None:
                print(f"[hf_engine] skip rotary {name}: cannot infer head_dim")
                continue
            theta = float(getattr(cfg, "rope_theta", 10000.0))
            inv_freq = 1.0 / (
                theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

        module.register_buffer("inv_freq", inv_freq, persistent=False)
        if hasattr(module, "attention_scaling"):
            module.attention_scaling = scaling
        patched += 1
        print(
            f"[hf_engine] reinit rotary {name or '<root>'} ({cls_name}) "
            f"inv_freq[{tuple(inv_freq.shape)}] "
            f"min={inv_freq.min().item():.3e} max={inv_freq.max().item():.3e}"
        )

    if patched == 0:
        print("[hf_engine] WARNING: no RotaryEmbedding modules found to reinit")
