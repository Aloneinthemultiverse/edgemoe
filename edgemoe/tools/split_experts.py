"""Split a HuggingFace MoE model into per-expert files — STREAMING.

Unlike a naive `AutoModelForCausalLM.from_pretrained` load (which needs
the whole model in RAM), this version:

  1. Downloads shard safetensors files to local disk via `snapshot_download`.
  2. Reads the header of every shard (no tensor data in RAM) and builds
     an index: tensor_name → shard_path.
  3. Walks (layer, expert) pairs in order, loading JUST that expert's few
     tensors from its shard(s), quantising them, flushing to disk, freeing
     memory before the next iteration.
  4. Saves a small backbone safetensors (attention + router + embeddings).

Memory footprint during split: ~O(single expert), typically < 200 MB.

Output layout:
    out/
      config.json, tokenizer*, special_tokens_map.json
      backbone.bin                  (safetensors of non-expert state)
      experts/manifest.json
      experts/L00_E000.bin          (our packed record format)
      ...
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Iterable

import torch


_EXPERT_RE = re.compile(
    r"^(?:model\.)?(?:layers|transformer\.h)\.(\d+)"
    r"\.(?:mlp|block_sparse_moe|moe)"
    r"\.experts\.(\d+)\.(.+)$"
)


def _serialize_tensor_record(record: dict) -> bytes:
    """Pack header-JSON + raw payloads into our on-disk format."""
    tensors = {k: v for k, v in record.items() if torch.is_tensor(v)}
    meta = {k: v for k, v in record.items() if not torch.is_tensor(v)}
    header = {"tensors": {}}
    header.update(meta)
    payload_parts: list[bytes] = []
    for name, t in tensors.items():
        t = t.contiguous().cpu()
        if t.dtype == torch.bfloat16:
            raw = t.view(torch.uint8).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        header["tensors"][name] = {
            "shape": list(t.shape),
            "torch_dtype": str(t.dtype).removeprefix("torch."),
            "nbytes": len(raw),
        }
        payload_parts.append(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(payload_parts)


def _build_tensor_index(shard_paths: Iterable[Path]) -> dict[str, Path]:
    from safetensors import safe_open
    idx: dict[str, Path] = {}
    for shard in shard_paths:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                idx[key] = shard
    return idx


def split_model(
    hf_model: str,
    output_dir: str | Path,
    quant_mode: str = "q4",
    revision: str = "main",
) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save_file

    from edgemoe.quantization.adaptive import AdaptiveQuantizer
    from edgemoe.quantization.bitnet import BitNetExpertQuantizer

    out = Path(output_dir)
    (out / "experts").mkdir(parents=True, exist_ok=True)

    print(f"[split] Resolving {hf_model} …")
    cfg = AutoConfig.from_pretrained(hf_model, trust_remote_code=True, revision=revision)
    tok = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True, revision=revision)

    print("[split] Downloading shards (safetensors only) …")
    snapshot = Path(snapshot_download(
        hf_model,
        revision=revision,
        allow_patterns=[
            "*.safetensors", "*.json", "tokenizer*", "*.model",
            "special_tokens_map.json", "generation_config.json",
        ],
    ))
    shard_paths = sorted(snapshot.glob("*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"No safetensors shards at {snapshot}")
    print(f"[split] Indexing {len(shard_paths)} shards …")
    tensor_index = _build_tensor_index(shard_paths)

    num_layers = getattr(cfg, "num_hidden_layers", None) or cfg.num_layers
    num_experts = (
        getattr(cfg, "num_experts", None)
        or getattr(cfg, "num_local_experts", None)
        or 0
    )
    top_k = (
        getattr(cfg, "num_experts_per_tok", None)
        or getattr(cfg, "top_k", None)
        or 8
    )
    if num_experts == 0:
        raise RuntimeError(
            f"Can't find expert count on config for {hf_model}. "
            f"Is this really a MoE model?"
        )

    adapt = AdaptiveQuantizer()
    bitnet = BitNetExpertQuantizer()
    quant_bits = {"q4": 4, "q3": 3, "q2": 2}.get(quant_mode, 4)

    manifest = {
        "model": hf_model,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "experts_per_token": top_k,
        "hidden_size": cfg.hidden_size,
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "vocab_size": cfg.vocab_size,
        "quant_mode": quant_mode,
        "experts": {},
    }

    # Group expert keys by (layer, expert) using a single regex pass.
    expert_keys: dict[tuple[int, int], dict[str, str]] = {}
    for key in tensor_index:
        m = _EXPERT_RE.match(key)
        if m:
            L = int(m.group(1)); E = int(m.group(2)); sub = m.group(3)
            expert_keys.setdefault((L, E), {})[sub] = key

    if not expert_keys:
        raise RuntimeError(
            "No expert keys matched the pattern. Check that this is a "
            "supported MoE family (Qwen3 / Mixtral / OLMoE / DeepSeek / Phi)."
        )

    print(f"[split] {len(expert_keys)} experts across {num_layers} layers "
          f"→ quant={quant_mode}, out={out}")

    total = len(expert_keys)
    for i, ((L, E), subkeys) in enumerate(sorted(expert_keys.items())):
        tensors: dict[str, torch.Tensor] = {}
        # Open each shard lazily; usually all subkeys sit in the same shard.
        shard_to_keys: dict[Path, list[tuple[str, str]]] = {}
        for sub, full in subkeys.items():
            shard_to_keys.setdefault(tensor_index[full], []).append((sub, full))
        for shard, items in shard_to_keys.items():
            with safe_open(str(shard), framework="pt") as f:
                for sub, full in items:
                    tensors[sub] = f.get_tensor(full)

        if quant_mode == "bitnet":
            rec: dict = {"dtype": "ternary"}
            for name, t in tensors.items():
                q = bitnet.quantize(t.float())
                rec[f"{name}.packed"] = q["packed"]
                rec[f"{name}.scale"] = torch.tensor([float(q["scale"])])
                rec[f"{name}.shape"] = list(q["shape"])
        else:
            rec = {"dtype": "group_asym", "mode": "group_asym"}
            for name, t in tensors.items():
                q = adapt.quantize(t.float(), bits=quant_bits)
                rec[f"{name}.q"] = q["q"]
                rec[f"{name}.scale"] = q["scale"]
                if "zp" in q:
                    rec[f"{name}.zp"] = q["zp"]

        payload = _serialize_tensor_record(rec)
        path = out / "experts" / f"L{L:02d}_E{E:03d}.bin"
        path.write_bytes(payload)
        manifest["experts"].setdefault(str(L), {})[str(E)] = {
            "offset": 0,
            "size": len(payload),
            "cluster": -1,
            "temp": "cold",
            "bits": 1.6 if quant_mode == "bitnet" else quant_bits,
        }
        del tensors
        if (i + 1) % 64 == 0 or (i + 1) == total:
            print(f"[split] experts {i + 1}/{total}")

    (out / "experts" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[split] Manifest written.")

    # Backbone: every non-expert key, loaded one at a time.
    print("[split] Writing backbone …")
    backbone_state: dict[str, torch.Tensor] = {}
    for key, shard in tensor_index.items():
        if _EXPERT_RE.match(key):
            continue
        with safe_open(str(shard), framework="pt") as f:
            backbone_state[key] = f.get_tensor(key)

    safetensors_save_file(backbone_state, str(out / "backbone.bin"))
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    tok.save_pretrained(out)
    bb_params = sum(v.numel() for v in backbone_state.values()) / 1e9
    print(f"[split] Done. Backbone: {bb_params:.2f}B params. Output: {out.resolve()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--output", required=True)
    p.add_argument("--quantize", default="q4",
                   choices=["q4", "q3", "q2", "bitnet"])
    p.add_argument("--revision", default="main")
    args = p.parse_args()
    split_model(args.model, args.output, quant_mode=args.quantize, revision=args.revision)
